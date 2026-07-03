"""RTC-like decoupled inference: full encoder + prefix-clamped decoder graph.

Execution pattern:
  1. Full inference (VLM + Encoder + Decoder) returns 10 actions.
     The runner executes actions [0:5] and stores that prefix.
  2. Prefix-clamped decoder-only inference reuses encoder K/V from the last
     full call, clamps the stored prefix during every denoising step, and the
     runner executes the generated suffix [5:10].

Key design decisions:
- n_action_steps=10 unchanged (training distribution match)
- predict_action() returns FULL n_action_steps (no truncation — API preserved)
- Truncation to exec_chunk_size is done by the CALLER (eval runner), not model
- predict_action_decoder_only_prefix() clamps an executed action prefix during
  every decoder denoising step for RTC-like suffix generation
- encoder_K / encoder_V buffers are shared between both graphs
"""

import torch

from fluxvla.engines import VLAS
from fluxvla.ops.atomic_ops import (adarms_norm_style_proj, matmul_attn_v,
                                    matmul_bias_small, matmul_bias_silu,
                                    matmul_qkv_rope, matmul_res_gate)
from fluxvla.ops.triton.attention_triton_ops import (matmul_abT_scale,
                                                     softmax_kernel_prefix_suffix)
from fluxvla.ops.ultra_v2 import matmul_gate_v2
from .pi05_flowmatching_inference import (PI05FlowMatchingInference,
                                          transformer_encoder, vision_encoder)
from .pi05_flowmatching_ultra_v2_inference import transformer_decoder_ultra_v2


def transformer_decoder_ultra_v2_prefix_clamp(weights,
                                              buffers,
                                              encoder_seq_len,
                                              num_decoder_layers=18,
                                              num_steps=10,
                                              prefix_len=5):
    """Decoder-only loop that clamps known prefix actions every denoise step.

    The prefix is clamped to the correct pre-update interpolation for each
    denoising step:
      prefix_at_step = (1-progress) * noise_init + progress * clean_action
    where progress = step / num_steps. This keeps the prefix at the same noise
    level as the suffix for the time embedding used by that step, so the model
    does not see a mix of clean prefix + noisy suffix.

    At step 0: prefix is pure noise, matching time t=1.0.
    At step 9: prefix is 90% denoised, matching time t=0.1.
    After the final Euler update, prefix is clamped exactly to clean action.

    This is intentionally local to the decoupled experiment so the verified
    Ultra-V2 decoder stays unchanged.
    """
    noise_init_prefix = buffers['noise_init_prefix'][:prefix_len]
    prefix_interp_buf = buffers['prefix_interp_buf'][:prefix_len]

    for step in range(num_steps):
        # Pre-update state must match decoder_time_embeds[step].
        progress = step / float(num_steps)  # 0.0, 0.1, ..., 0.9
        prefix_interp_buf.copy_(noise_init_prefix)
        prefix_interp_buf.mul_(1.0 - progress)
        prefix_interp_buf.add_(
            buffers['action_prefix'][:prefix_len], alpha=progress)
        buffers['diffusion_noise'][:prefix_len].copy_(prefix_interp_buf)

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
        buffers['decoder_time_emb'].copy_(buffers['decoder_time_scratch'])

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

            matmul_gate_v2(
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                in_features=1024,
                intermediate_dim=4096)

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

        seq_len = buffers['decoder_x'].shape[0]
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
            buffers['decoder_action_buf'], alpha=-1.0 / num_steps)
        # No post-step clamp here. The next iteration sets the prefix to the
        # exact interpolation level for the next time embedding.
    # Final clamp: ensure prefix is exactly clean action after all steps
    buffers['diffusion_noise'][:prefix_len].copy_(
        buffers['action_prefix'][:prefix_len])


@VLAS.register_module()
class PI05FlowMatchingDecoupledInference(PI05FlowMatchingInference):
    """Decoupled inference with separate full and decoder-only CUDA Graphs.

    This version is for textual TempoVLA/VSTA checkpoints: speed is encoded in
    the prompt, not in a speed_mlp. It still uses the autotuned UltraV2
    ffn_gate kernel in the decoder. Adds a second CUDA graph that only runs the
    decoder, reusing encoder K/V from the last full inference.

    Args:
        exec_chunk_size (int): How many of the n_action_steps to actually
            return/execute. Default: 5.
        *args, **kwargs: Forwarded to PI05FlowMatchingInference.
    """

    def __init__(self, exec_chunk_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_chunk_size = exec_chunk_size
        self._cuda_graph_decoder_prefix = None
        self._cuda_graph_decoder_prefix_ready = False
        self._cuda_graph_decoder_prefix_len = None

    def _run_forward(self):
        vision_encoder(self._triton_weights, self._triton_bufs,
                       self.num_views, self._num_vit_layers)
        transformer_encoder(
            self._triton_weights, self._triton_bufs, self._encoder_seq_len,
            self._num_encoder_layers, self._visual_tokens_per_view,
            self._visual_grid_size, self._visual_token_downscale_factor)
        transformer_decoder_ultra_v2(
            self._triton_weights, self._triton_bufs, self._encoder_seq_len,
            self._num_decoder_layers, self._num_steps)

    def _ensure_action_prefix_buffer(self):
        if 'action_prefix' not in self._triton_bufs:
            self._triton_bufs['action_prefix'] = torch.zeros_like(
                self._triton_bufs['diffusion_noise'])
        if 'noise_init_prefix' not in self._triton_bufs:
            self._triton_bufs['noise_init_prefix'] = torch.zeros_like(
                self._triton_bufs['diffusion_noise'])
        if 'prefix_interp_buf' not in self._triton_bufs:
            self._triton_bufs['prefix_interp_buf'] = torch.zeros_like(
                self._triton_bufs['diffusion_noise'])

    def _run_decoder_only_prefix(self, prefix_len):
        self._ensure_action_prefix_buffer()
        transformer_decoder_ultra_v2_prefix_clamp(
            self._triton_weights, self._triton_bufs,
            self._encoder_seq_len, self._num_decoder_layers,
            self._num_steps, prefix_len=prefix_len)

    def _build_decoder_only_prefix_graph(self, prefix_len):
        """Record a CUDA Graph for prefix-clamped decoder-only inference."""
        print('[Decoupled] Recording decoder-only-prefix CUDA Graph ...')
        self._ensure_action_prefix_buffer()
        for _ in range(3):
            self._run_decoder_only_prefix(prefix_len)
        torch.cuda.synchronize()

        self._cuda_graph_decoder_prefix = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_graph_decoder_prefix.capture_begin()
            self._run_decoder_only_prefix(prefix_len)
            self._cuda_graph_decoder_prefix.capture_end()
        torch.cuda.synchronize()

        self._cuda_graph_decoder_prefix_ready = True
        self._cuda_graph_decoder_prefix_len = prefix_len
        print('[Decoupled] Decoder-only-prefix CUDA Graph recorded successfully!')

    def _ensure_decoder_prefix_graph(self, prefix_len):
        """Lazily build the prefix-clamped decoder-only graph."""
        if (not self._cuda_graph_decoder_prefix_ready
                or self._cuda_graph_decoder_prefix_len != prefix_len):
            if not self._cuda_graph_ready:
                raise RuntimeError(
                    'Must run at least one full predict_action() before '
                    'predict_action_decoder_only_prefix()')
            self._build_decoder_only_prefix_graph(prefix_len)

    def _prepare_decoder_noise(self, noise):
        chunk_size = self.n_action_steps
        device = self._triton_bufs['diffusion_noise'].device
        if noise is None:
            noise_t = torch.randn(
                chunk_size, self.max_action_dim,
                dtype=torch.bfloat16, device=device)
        else:
            noise_t = noise[0].to(dtype=torch.bfloat16, device=device)
        if noise_t.shape[-1] < 32:
            pad = torch.zeros(
                noise_t.shape[0], 32 - noise_t.shape[-1],
                dtype=torch.bfloat16, device=device)
            noise_t = torch.cat([noise_t, pad], dim=-1)
        return noise_t

    def _write_prefix_actions(self, prefix_actions, prefix_len):
        if prefix_actions is None:
            raise ValueError('prefix_actions is required for prefix RTC mode')
        if prefix_len <= 0 or prefix_len > self.n_action_steps:
            raise ValueError(
                f'Invalid prefix_len={prefix_len}; n_action_steps='
                f'{self.n_action_steps}')

        self._ensure_action_prefix_buffer()
        device = self._triton_bufs['diffusion_noise'].device
        prefix_t = prefix_actions.to(dtype=torch.bfloat16, device=device)
        if prefix_t.dim() == 3:
            prefix_t = prefix_t[0]
        if prefix_t.shape[0] < prefix_len:
            raise ValueError(
                f'prefix_actions has {prefix_t.shape[0]} steps, '
                f'but prefix_len={prefix_len}')
        if prefix_t.shape[-1] < 32:
            pad = torch.zeros(
                prefix_t.shape[0], 32 - prefix_t.shape[-1],
                dtype=torch.bfloat16, device=device)
            prefix_t = torch.cat([prefix_t, pad], dim=-1)
        self._triton_bufs['action_prefix'][:prefix_len].copy_(
            prefix_t[:prefix_len, :32])

    def predict_action_decoder_only_prefix(self, prefix_actions, prefix_len,
                                           noise=None, tempo_speed=None):
        """Decoder-only RTC-like suffix generation with a clamped action prefix.

        prefix_actions are normalized model actions from the previous full call.
        The first prefix_len actions are used as clean prefix targets; inside
        the graph they are clamped to the time-aligned interpolation for each
        denoising step, and the final output prefix is clamped exactly to those
        targets. Caller should execute the suffix slice [prefix_len:2*prefix_len].
        """
        self._ensure_decoder_prefix_graph(prefix_len)

        if tempo_speed is not None:
            raise ValueError(
                'PI05FlowMatchingDecoupledInference is textual TempoVLA; '
                'set speed through LiberoPromptFromInputs.speed, not '
                'tempo_speed.')

        noise_t = self._prepare_decoder_noise(noise)
        self._write_prefix_actions(prefix_actions, prefix_len)

        # Save initial noise for the prefix positions so the decoder can
        # interpolate between noise and clean action at each timestep.
        # This must be written BEFORE diffusion_noise (graph reads both).
        self._triton_bufs['noise_init_prefix'][:prefix_len].copy_(
            noise_t[:prefix_len])

        self._triton_bufs['diffusion_noise'].copy_(noise_t)
        self._cuda_graph_decoder_prefix.replay()

        result = (self._triton_bufs['diffusion_noise'][:, :self.max_action_dim]
                  .unsqueeze(0).float())
        return result

    def predict_action(self, images, lang_tokens, states, img_masks=None,
                       lang_masks=None, past_key_values=None, noise=None,
                       tempo_speed=None, *args, **kwargs):
        """Full inference (VLM + Encoder + Decoder).

        Returns FULL n_action_steps (no truncation — API preserved).
        Caller decides how many steps to execute.
        """
        if tempo_speed is not None:
            raise ValueError(
                'PI05FlowMatchingDecoupledInference is textual TempoVLA; '
                'set speed through LiberoPromptFromInputs.speed, not '
                'tempo_speed.')
        result = super().predict_action(
            images=images, lang_tokens=lang_tokens, states=states,
            img_masks=img_masks, lang_masks=lang_masks,
            past_key_values=past_key_values, noise=noise, *args, **kwargs)

        return result
