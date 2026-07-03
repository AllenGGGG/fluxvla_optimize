"""Ultra-v2 inference: only the ffn_gate kernel is replaced with an autotuned
version that is 1.46x faster at M=10 (microbench verified). Everything else
(attn_o, ffn_down, norm, time_mlp, attention) stays on the proven standard path.

This file does NOT modify any existing code. It defines:
- transformer_decoder_ultra_v2: decoder loop with autotuned ffn_gate
- pi05_model_ultra_v2: full model forward
- PI05FlowMatchingUltraV2Inference: registered class for configs
"""

from fluxvla.engines import VLAS
from fluxvla.ops.atomic_ops import (adarms_norm_style_proj, matmul_attn_v,
                                    matmul_bias_small, matmul_bias_silu,
                                    matmul_qkv_rope, matmul_res_gate)
from fluxvla.ops.triton.attention_triton_ops import (matmul_abT_scale,
                                                     softmax_kernel_prefix_suffix)
from fluxvla.ops.ultra_v2 import matmul_gate_v2
from .pi05_flowmatching_inference import transformer_encoder, vision_encoder
from .pi05_flowmatching_inference import PI05FlowMatchingInference
from .pi05_flowmatching_speed_modulated_inference import (
    PI05FlowMatchingSpeedModulatedInference)


def transformer_decoder_ultra_v2(weights,
                                 buffers,
                                 encoder_seq_len,
                                 num_decoder_layers=18,
                                 num_steps=10):
    """Decoder with autotuned ffn_gate kernel only.

    Identical to the standard transformer_decoder EXCEPT:
    - matmul_gate -> matmul_gate_v2 (autotuned, 1.46x faster at M=10)

    attn_o and ffn_down stay on matmul_res_gate (standard path won in microbench).
    """
    for step in range(num_steps):
        # Time MLP
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

            # attn output projection — STANDARD path (won in microbench)
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

            # FFN gate+up — AUTOTUNED v2 (1.46x faster at M=10)
            matmul_gate_v2(
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                in_features=1024,
                intermediate_dim=4096)

            # FFN down — STANDARD path (won in microbench)
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


def pi05_model_ultra_v2(weights,
                        buffers,
                        num_views,
                        encoder_seq_len,
                        num_vit_layers=27,
                        num_encoder_layers=18,
                        num_decoder_layers=18,
                        num_steps=10,
                        visual_tokens_per_view=256,
                        visual_grid_size=16,
                        visual_token_downscale_factor=1):
    """Full model with ultra-v2 decoder (autotuned ffn_gate only)."""
    vision_encoder(weights, buffers, num_views, num_vit_layers)
    transformer_encoder(weights, buffers, encoder_seq_len, num_encoder_layers,
                        visual_tokens_per_view, visual_grid_size,
                        visual_token_downscale_factor)
    transformer_decoder_ultra_v2(weights, buffers, encoder_seq_len,
                                 num_decoder_layers, num_steps)


@VLAS.register_module()
class PI05FlowMatchingUltraV2Inference(PI05FlowMatchingSpeedModulatedInference):
    """Ultra-v2 inference: speed-modulated + autotuned ffn_gate kernel.

    Only overrides _run_forward to route through the v2 decoder. Everything
    else (speed caching, CUDA graph, buffer init, predict_action) is inherited.
    """

    def __init__(self, *args, **kwargs):
        # Force use_ultra_fusion=False since we don't use the old ultra path
        kwargs['use_ultra_fusion'] = False
        super().__init__(*args, **kwargs)

    def _run_forward(self):
        pi05_model_ultra_v2(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers,
            self._num_steps, self._visual_tokens_per_view,
            self._visual_grid_size,
            self._visual_token_downscale_factor)


@VLAS.register_module()
class PI05FlowMatchingPlainUltraV2Inference(PI05FlowMatchingInference):
    """Ultra-v2 inference for non-TempoVLA PI0.5 checkpoints."""

    def _run_forward(self):
        pi05_model_ultra_v2(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers,
            self._num_steps, self._visual_tokens_per_view,
            self._visual_grid_size,
            self._visual_token_downscale_factor)
