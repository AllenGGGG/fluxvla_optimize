import math

import torch

from fluxvla.engines.utils.root import VLAS
from .pi05_flowmatching_inference import (PI05FlowMatchingInference,
                                          transformer_encoder, vision_encoder)
from .pi05_flowmatching_speed_modulated_inference import (
    PI05FlowMatchingSpeedModulatedInference)


def transformer_decoder_time_schedule(weights,
                                      buffers,
                                      encoder_seq_len,
                                      num_decoder_layers=18,
                                      num_steps=10,
                                      use_ultra_fusion=True,
                                      time_deltas=None):
    """Local decoder variant with per-step Euler deltas."""
    if time_deltas is None:
        time_deltas = [-1.0 / num_steps] * num_steps
    if len(time_deltas) != num_steps:
        raise ValueError(
            f'time_deltas length must match num_steps: {len(time_deltas)} vs '
            f'{num_steps}')

    for step in range(num_steps):
        # Time MLP produces a single [1, hidden] conditioning row that must be
        # broadcast to ALL action tokens (matching the reference scalar-time
        # GemmaRMSNorm broadcast), not just written to row 0. Compute one row
        # in decoder_time_scratch, then broadcast across the full decoder seq.
        if use_ultra_fusion and 'decoder_speed_emb' in weights:
            from fluxvla.ops.atomic_ops import time_mlp_with_speed_optimized
            time_mlp_with_speed_optimized(
                weights['decoder_time_embeds'][step].view(1, -1),
                weights['decoder_time_mlp_in_w'],
                weights['decoder_time_mlp_in_b'],
                weights['decoder_time_mlp_out_w'],
                weights['decoder_time_mlp_out_b'],
                weights['decoder_speed_emb'],
                buffers['decoder_time_scratch'],
                hidden_dim=1024)
        else:
            from fluxvla.ops.atomic_ops import matmul_bias_silu
            matmul_bias_silu(
                weights['decoder_time_embeds'][step].view(1, -1),
                weights['decoder_time_mlp_in_w'],
                weights['decoder_time_mlp_in_b'],
                buffers['decoder_x_buf'][:1],
                in_features=1024,
                out_features=1024)
            matmul_bias_silu(
                buffers['decoder_x_buf'][:1],
                weights['decoder_time_mlp_out_w'],
                weights['decoder_time_mlp_out_b'],
                buffers['decoder_time_scratch'],
                in_features=1024,
                out_features=1024)
            if 'decoder_speed_emb' in weights:
                buffers['decoder_time_scratch'].add_(
                    weights['decoder_speed_emb'])
        # Broadcast the single time-conditioning row across all action tokens.
        buffers['decoder_time_emb'].copy_(buffers['decoder_time_scratch'])

        from fluxvla.ops.atomic_ops import (adarms_norm_style_proj,
                                            matmul_attn_v, matmul_bias_small,
                                            matmul_gate, matmul_qkv_rope,
                                            matmul_res_gate)
        from fluxvla.ops.triton.attention_triton_ops import (
            matmul_abT_scale, softmax_kernel_prefix_suffix)

        matmul_bias_small(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x'],
            in_features=32,
            out_features=1024,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_K=32)
        seq_len = buffers['decoder_x'].shape[0]

        for i in range(num_decoder_layers):
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'],
                weights['decoder_pre_attn_norm_mod_w'][i],
                weights['decoder_pre_attn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style'],
                hidden_dim=1024,
                style_dim=3072)

            matmul_qkv_rope(
                buffers['x_normed_buf'],
                weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len +
                                     seq_len],
                hidden_dim=1024,
                head_dim=256,
                num_kv_heads=8)

            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                                  buffers['decoder_q_buf'],
                                  buffers['encoder_K'][i, :encoder_seq_len +
                                                       seq_len],
                                  buffers['decoder_logits_buf'],
                                  total_queries,
                                  total_keys,
                                  256,
                                  256**-0.5,
                                  BLOCK_SIZE_M=32,
                                  BLOCK_SIZE_N=32,
                                  BLOCK_SIZE_K=64)

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4, )](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024)

            matmul_attn_v(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
                head_dim=256)

            if use_ultra_fusion:
                from fluxvla.ops.atomic_ops import matmul_res_gate_optimized
                matmul_res_gate_optimized(
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'],
                    buffers['decoder_x'],
                    buffers['gate_buf'],
                    in_features=2048,
                    out_features=1024,
                    BLOCK_SIZE_N=32,
                    BLOCK_SIZE_M=32,
                    BLOCK_SIZE_K=128)
            else:
                matmul_res_gate(
                    buffers['decoder_q_buf'].view(-1, 2048),
                    weights['decoder_attn_o_w'][i],
                    buffers['decoder_x'],
                    buffers['gate_buf'],
                    in_features=2048,
                    out_features=1024,
                    BLOCK_SIZE_N=32,
                    BLOCK_SIZE_M=32,
                    BLOCK_SIZE_K=128)

            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'],
                weights['decoder_pre_ffn_norm_mod_w'][i],
                weights['decoder_pre_ffn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style'],
                hidden_dim=1024,
                style_dim=3072)

            matmul_gate(
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                in_features=1024,
                intermediate_dim=4096)

            if use_ultra_fusion:
                from fluxvla.ops.atomic_ops import matmul_split_k_res_gate
                matmul_split_k_res_gate(
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'],
                    buffers['decoder_x'],
                    buffers['gate_buf'],
                    buffers['decode_split_k_buf'],
                    in_features=4096,
                    out_features=1024,
                    split_k=8)
            else:
                matmul_res_gate(
                    buffers['decoder_hidden'],
                    weights['decoder_ffn_down_w'][i],
                    buffers['decoder_x'],
                    buffers['gate_buf'],
                    in_features=4096,
                    out_features=1024,
                    BLOCK_SIZE_N=16,
                    BLOCK_SIZE_M=32,
                    BLOCK_SIZE_K=256)

        adarms_norm_style_proj(
            buffers['decoder_x'],
            buffers['decoder_time_emb'],
            weights['decoder_final_norm_mod_w'],
            weights['decoder_final_norm_mod_b'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style'],
            hidden_dim=1024,
            style_dim=3072)

        matmul_bias_small(
            buffers['x_normed_buf'],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['decoder_action_buf'],
            in_features=1024,
            out_features=32,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_K=256)

        buffers['diffusion_noise'].add_(
            buffers['decoder_action_buf'], alpha=float(time_deltas[step]))


def pi05_model_time_schedule(weights,
                             buffers,
                             num_views,
                             encoder_seq_len,
                             num_vit_layers=27,
                             num_encoder_layers=18,
                             num_decoder_layers=18,
                             num_steps=10,
                             visual_tokens_per_view=256,
                             visual_grid_size=16,
                             visual_token_downscale_factor=1,
                             time_deltas=None,
                             use_ultra_fusion=True):
    vision_encoder(weights, buffers, num_views, num_vit_layers)
    transformer_encoder(weights, buffers, encoder_seq_len, num_encoder_layers,
                        visual_tokens_per_view, visual_grid_size,
                        visual_token_downscale_factor)
    transformer_decoder_time_schedule(weights, buffers, encoder_seq_len,
                                      num_decoder_layers, num_steps,
                                      time_deltas=time_deltas,
                                      use_ultra_fusion=use_ultra_fusion)


@VLAS.register_module()
class PI05FlowMatchingTimeScheduleInference(PI05FlowMatchingInference):
    """PI0.5 inference variant for custom denoising time schedules."""

    def __init__(self, time_schedule=None, time_deltas=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_schedule = time_schedule
        self.time_deltas = time_deltas

    @staticmethod
    def _as_float_list(values, name):
        if values is None:
            return None
        if isinstance(values, str):
            values = [item.strip() for item in values.split(',') if item.strip()]
        result = [float(value) for value in values]
        if not result:
            raise ValueError(f'{name} must not be empty.')
        return result

    def _normalize_schedule(self, num_steps):
        schedule = self._as_float_list(self.time_schedule, 'time_schedule')
        deltas = self._as_float_list(self.time_deltas, 'time_deltas')
        if schedule is None:
            dt = -1.0 / num_steps
            schedule = [1.0 + i * dt for i in range(num_steps)]
        else:
            num_steps = len(schedule)
        if deltas is None:
            deltas = [
                float(next_time - current_time)
                for current_time, next_time in zip(schedule,
                                                   schedule[1:] + [0.0])
            ]
        elif len(deltas) != num_steps:
            raise ValueError(
                f'time_deltas length must match num_steps: {len(deltas)} vs '
                f'{num_steps}')
        return num_steps, schedule, deltas

    def _prepare_adarms_cond_from_schedule(self, schedule):
        min_period = 4e-3
        max_period = 4.0
        embedding_dim = 1024
        fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device='cuda')
        period = min_period * (max_period / min_period)**fraction
        time_embs = []
        for value in schedule:
            time_val = torch.tensor(float(value), dtype=torch.float32,
                                    device='cuda')
            sinusoid_input = (
                time_val.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 *
                math.pi)
            emb = torch.cat(
                [torch.sin(sinusoid_input),
                 torch.cos(sinusoid_input)], dim=-1)
            time_embs.append(emb.to(torch.bfloat16))
        return torch.cat(time_embs, dim=0)

    def prepare_triton_inference(self, num_views, max_prompt_len, chunk_size,
                                 num_steps):
        num_steps, schedule, deltas = self._normalize_schedule(num_steps)
        super().prepare_triton_inference(num_views, max_prompt_len, chunk_size,
                                         num_steps)
        self._triton_weights['decoder_time_embeds'] = (
            self._prepare_adarms_cond_from_schedule(schedule))
        self._num_steps = num_steps
        self.num_steps = num_steps
        self._decoder_time_schedule = tuple(schedule)
        self._decoder_time_deltas = tuple(deltas)
        self._cuda_graph = None
        self._cuda_graph_ready = False

    def _run_forward(self):
        pi05_model_time_schedule(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers,
            self._num_steps, self._visual_tokens_per_view,
            self._visual_grid_size, self._visual_token_downscale_factor,
            time_deltas=self._decoder_time_deltas)


@VLAS.register_module()
class PI05FlowMatchingSpeedModulatedTimeScheduleInference(
        PI05FlowMatchingSpeedModulatedInference):
    """Speed-modulated inference with custom denoising time schedules.

    This is for checkpoints trained with PI05FlowMatchingSpeedModulated. It
    keeps the speed MLP / tempo conditioning path intact, while replacing the
    default evenly spaced timestep schedule with a user-provided one.
    """

    def __init__(self, time_schedule=None, time_deltas=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_schedule = time_schedule
        self.time_deltas = time_deltas

    @staticmethod
    def _as_float_list(values, name):
        if values is None:
            return None
        if isinstance(values, str):
            values = [item.strip() for item in values.split(',') if item.strip()]
        result = [float(value) for value in values]
        if not result:
            raise ValueError(f'{name} must not be empty.')
        return result

    def _normalize_schedule(self, num_steps):
        schedule = self._as_float_list(self.time_schedule, 'time_schedule')
        deltas = self._as_float_list(self.time_deltas, 'time_deltas')
        if schedule is None:
            dt = -1.0 / num_steps
            schedule = [1.0 + i * dt for i in range(num_steps)]
        else:
            num_steps = len(schedule)
        if deltas is None:
            deltas = [
                float(next_time - current_time)
                for current_time, next_time in zip(schedule,
                                                   schedule[1:] + [0.0])
            ]
        elif len(deltas) != num_steps:
            raise ValueError(
                f'time_deltas length must match num_steps: {len(deltas)} vs '
                f'{num_steps}')
        return num_steps, schedule, deltas

    def _prepare_adarms_cond_from_schedule(self, schedule):
        min_period = 4e-3
        max_period = 4.0
        embedding_dim = 1024
        fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device='cuda')
        period = min_period * (max_period / min_period)**fraction
        time_embs = []
        for value in schedule:
            time_val = torch.tensor(float(value), dtype=torch.float32,
                                    device='cuda')
            sinusoid_input = (
                time_val.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 *
                math.pi)
            emb = torch.cat(
                [torch.sin(sinusoid_input),
                 torch.cos(sinusoid_input)], dim=-1)
            time_embs.append(emb.to(torch.bfloat16))
        return torch.cat(time_embs, dim=0)

    def prepare_triton_inference(self,
                                 num_views,
                                 max_prompt_len,
                                 chunk_size,
                                 num_steps,
                                 tempo_speed=None):
        num_steps, schedule, deltas = self._normalize_schedule(num_steps)
        super().prepare_triton_inference(
            num_views,
            max_prompt_len,
            chunk_size,
            num_steps,
            tempo_speed=tempo_speed)
        self._triton_weights['decoder_time_embeds'] = (
            self._prepare_adarms_cond_from_schedule(schedule))
        self._num_steps = num_steps
        self.num_steps = num_steps
        self._decoder_time_schedule = tuple(schedule)
        self._decoder_time_deltas = tuple(deltas)
        self._cuda_graph = None
        self._cuda_graph_ready = False

    def _run_forward(self):
        pi05_model_time_schedule(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers,
            self._num_steps, self._visual_tokens_per_view,
            self._visual_grid_size, self._visual_token_downscale_factor,
            time_deltas=self._decoder_time_deltas,
            use_ultra_fusion=self.use_ultra_fusion)
