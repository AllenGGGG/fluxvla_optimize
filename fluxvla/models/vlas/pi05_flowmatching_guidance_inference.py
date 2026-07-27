"""Triton/CUDA-graph PI0.5 inference with non-VJP guidance RTC.

The deployed guidance configuration uses ``use_vjp=False``.  In that mode the
RTC correction is closed-form pointwise arithmetic after every denoising step,
so it can be captured in the same CUDA Graph without autograd.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from fluxvla.engines import VLAS
from fluxvla.engines.utils.rtc_guidance import (compute_guidance_weight,
                                                compute_prefix_weights)
from fluxvla.ops.atomic_ops import (
    adarms_matmul_k_1024_32_bias_res_rowwise,
    adarms_norm_mod_proj_rowwise,
    matmul_attn_v,
    matmul_bias_small,
    matmul_k_1024_2560_qkv_rope,
    matmul_res_gate,
    matmul_small_gate,
)
from fluxvla.ops.triton.attention_triton_ops import (matmul_abT_scale,
                                                     softmax_kernel_prefix_suffix)

from .pi05_flowmatching_inference_rtc import (
    PI05FlowMatchingRTCInference,
    transformer_encoder,
    vision_encoder,
)


@triton.jit
def _guidance_update_kernel(
    x_new_ptr,
    x_old_ptr,
    prev_ptr,
    prefix_weights_ptr,
    step_scales_ptr,
    numel: tl.constexpr,
    ACTION_DIM: tl.constexpr,
    NUM_STEPS: tl.constexpr,
    STEP: tl.constexpr,
    TIME: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    positions = offsets // ACTION_DIM

    x_new = tl.load(x_new_ptr + offsets, mask=mask).to(tl.float32)
    x_old = tl.load(x_old_ptr + offsets, mask=mask).to(tl.float32)
    prev = tl.load(prev_ptr + offsets, mask=mask).to(tl.float32)
    weight = tl.load(prefix_weights_ptr + positions, mask=mask).to(tl.float32)
    step_scale = tl.load(step_scales_ptr + STEP).to(tl.float32)

    # The graph has already applied x_new = x_old - denoise / NUM_STEPS.
    # Eager guidance predicts the clean action as
    # x_1 = x_old - denoise * TIME, which can be reconstructed without
    # retaining a separate velocity tensor.
    x_1 = x_old + (x_new - x_old) * (NUM_STEPS * TIME)
    corrected = x_new + step_scale * weight * (prev - x_1)
    tl.store(x_new_ptr + offsets, corrected, mask=mask)


def _apply_guidance_step(buffers: dict, step: int, num_steps: int) -> None:
    x_new = buffers['diffusion_noise']
    numel = x_new.numel()
    time_value = 1.0 - float(step) / float(num_steps)
    _guidance_update_kernel[(triton.cdiv(numel, 256), )](
        x_new,
        buffers['guidance_x_old'],
        buffers['guidance_prev_actions'],
        buffers['guidance_prefix_weights'],
        buffers['guidance_step_scales'],
        numel=numel,
        ACTION_DIM=x_new.shape[-1],
        NUM_STEPS=num_steps,
        STEP=step,
        TIME=time_value,
        BLOCK_SIZE=256,
    )


def transformer_decoder_guidance(
    weights,
    buffers,
    encoder_seq_len,
    num_decoder_layers=18,
    num_steps=10,
):
    """RTC decoder with one fused non-VJP guidance update per flow step."""
    for step in range(num_steps):
        buffers['guidance_x_old'].copy_(buffers['diffusion_noise'])
        matmul_bias_small(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x'],
            in_features=32,
            out_features=1024,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_K=32,
        )
        seq_len = buffers['decoder_x'].shape[0]
        for layer in range(num_decoder_layers):
            adarms_norm_mod_proj_rowwise(
                buffers['decoder_x'],
                buffers['decoder_adarms_mod_attn'][step, layer],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
            )
            matmul_k_1024_2560_qkv_rope(
                buffers['x_normed_buf'],
                weights['decoder_attn_qkv_w'][layer],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][
                    layer, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['encoder_V'][
                    layer, encoder_seq_len:encoder_seq_len + seq_len],
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            total_keys = encoder_seq_len + seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) *
                              ((total_keys + 31) // 32), )](
                buffers['decoder_q_buf'],
                buffers['encoder_K'][layer, :total_keys],
                buffers['decoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                256**-0.5,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            softmax_kernel_prefix_suffix[((total_queries + 3) // 4, )](
                buffers['decoder_logits_buf'],
                total_queries,
                encoder_seq_len,
                seq_len,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_attn_v(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][layer, :total_keys],
                buffers['decoder_q_buf'],
                head_dim=256,
            )
            matmul_res_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][layer],
                buffers['decoder_x'],
                buffers['gate_buf'],
                in_features=2048,
                out_features=1024,
            )
            adarms_norm_mod_proj_rowwise(
                buffers['decoder_x'],
                buffers['decoder_adarms_mod_ffn'][step, layer],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
            )
            matmul_small_gate[((seq_len + 127) // 128, (4096 + 63) // 64)](
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][layer],
                weights['decoder_ffn_up_w'][layer],
                buffers['decoder_hidden'],
                seq_len,
                1024,
                4096,
            )
            matmul_res_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][layer],
                buffers['decoder_x'],
                buffers['gate_buf'],
                in_features=4096,
                out_features=1024,
                BLOCK_SIZE_N=16,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_K=256,
            )

        adarms_matmul_k_1024_32_bias_res_rowwise(
            buffers['decoder_x'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_adarms_mod_final'][step],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['diffusion_noise'],
        )
        _apply_guidance_step(buffers, step, num_steps)


def pi05_guidance_model(
    weights,
    buffers,
    num_views,
    encoder_seq_len,
    num_vit_layers=27,
    num_encoder_layers=18,
    num_decoder_layers=18,
    num_steps=10,
):
    vision_encoder(weights, buffers, num_views, num_vit_layers)
    transformer_encoder(weights, buffers, encoder_seq_len, num_encoder_layers)
    transformer_decoder_guidance(
        weights, buffers, encoder_seq_len, num_decoder_layers, num_steps)


@VLAS.register_module()
class PI05FlowMatchingGuidanceInference(PI05FlowMatchingRTCInference):
    """Fused Triton/CUDA-graph inference with non-VJP guidance RTC."""

    def _init_buffers(self):
        super()._init_buffers()
        dec = self._decoder_seq_len
        action_dim = self._action_dim
        self._triton_bufs.update({
            # fp32: guidance_x_old is a snapshot of diffusion_noise (now
            # fp32) taken before each step's velocity update, and
            # guidance_prev_actions is the correction target. Keeping either
            # bf16 would reintroduce the quantization the fp32
            # diffusion_noise buffer is meant to avoid, since
            # _guidance_update_kernel reconstructs x_1 from x_old and
            # computes (prev - x_1).
            'guidance_x_old':
            torch.zeros(dec, action_dim, dtype=torch.float32, device='cuda'),
            'guidance_prev_actions':
            torch.zeros(dec, action_dim, dtype=torch.float32, device='cuda'),
            'guidance_prefix_weights':
            torch.zeros(dec, dtype=torch.float32, device='cuda'),
            'guidance_step_scales':
            torch.zeros(self._num_steps, dtype=torch.float32, device='cuda'),
        })

    def _run_forward(self):
        pi05_guidance_model(
            self._triton_weights,
            self._triton_bufs,
            self.num_views,
            self._encoder_seq_len,
            self._num_vit_layers,
            self._num_encoder_layers,
            self._num_decoder_layers,
            self._num_steps,
        )

    def _build_cuda_graph(self):
        """Warm up/capture without consuming the caller's initial noise."""
        print('[Triton Guidance Inference] Recording CUDA Graph ...')
        initial_noise = self._triton_bufs['diffusion_noise'].clone()
        initial_encoder_x = self._triton_bufs['encoder_x'].clone()
        for _ in range(3):
            self._triton_bufs['diffusion_noise'].copy_(initial_noise)
            self._triton_bufs['encoder_x'].copy_(initial_encoder_x)
            self._run_forward()
        torch.cuda.synchronize()

        self._triton_bufs['diffusion_noise'].copy_(initial_noise)
        self._triton_bufs['encoder_x'].copy_(initial_encoder_x)
        torch.cuda.synchronize()
        self._cuda_graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self._cuda_graph.capture_begin()
            self._run_forward()
            self._cuda_graph.capture_end()
        torch.cuda.synchronize()

        # _triton_forward replays once after this method returns. Restore the
        # real request input so that first replay is identical to later calls.
        self._triton_bufs['diffusion_noise'].copy_(initial_noise)
        self._triton_bufs['encoder_x'].copy_(initial_encoder_x)
        self._cuda_graph_ready = True
        print('[Triton Guidance Inference] CUDA Graph recorded successfully!')

    def _prepare_guidance(self, prev_actions, prefix_len, rtc_config):
        prev_buffer = self._triton_bufs['guidance_prev_actions']
        weight_buffer = self._triton_bufs['guidance_prefix_weights']
        scale_buffer = self._triton_bufs['guidance_step_scales']
        prev_buffer.zero_()
        weight_buffer.zero_()

        enabled = (
            prev_actions is not None
            and int(prefix_len or 0) > 0
            and rtc_config is not None
            and rtc_config.get('enabled', True)
        )
        if not enabled:
            scale_buffer.zero_()
            return
        method = rtc_config.get('method', 'guidance')
        if method != 'guidance':
            raise NotImplementedError(
                'PI05FlowMatchingGuidanceInference only supports guidance RTC.')
        if rtc_config.get('use_vjp', False):
            raise NotImplementedError(
                'The fused guidance path supports use_vjp=False only.')

        prev = torch.as_tensor(
            prev_actions, dtype=torch.float32, device='cuda')
        if prev.ndim == 3:
            prev = prev[0]
        if prev.shape[-1] < self._action_dim:
            prev = torch.nn.functional.pad(
                prev, (0, self._action_dim - prev.shape[-1]))
        copy_len = min(len(prev), self._decoder_seq_len)
        prev_buffer[:copy_len].copy_(prev[:copy_len])

        # Clamp both prefix_len and decay_end to copy_len (not just
        # decoder_seq_len): compute_prefix_weights sets weights[:prefix_len]
        # = 1.0 unconditionally, and _guidance_update_kernel applies
        # weight * (prev - x_1) at every position with nonzero weight. But
        # prev_buffer only holds real data up to copy_len -- beyond that
        # it's the zero_() fill above, not an actual previous action.
        # Clamping decay_end alone is not enough: compute_prefix_weights
        # still forces the locked region weights[:prefix_len] = 1.0 even
        # when prefix_len > copy_len, which would pull x_1 toward that fake
        # zero-filled prev data. Eager's compute_full_error
        # (rtc_guidance.py) gets this for free because it slices
        # error_full[:, :L_prev, :] = ...; this mirrors that guard so a
        # caller-supplied prefix_len/decay_end > copy_len can't silently
        # pull the correction toward zero instead of leaving it unguided.
        prefix_len = min(int(prefix_len), self._decoder_seq_len, copy_len)
        decay_end = min(
            int(rtc_config.get('decay_end', prefix_len * 2)),
            self._decoder_seq_len,
            copy_len,
        )
        weights = compute_prefix_weights(
            self._decoder_seq_len,
            prefix_len,
            decay_end,
            rtc_config.get('schedule', 'exp'),
            device='cpu',
        )
        weight_buffer.copy_(weights)

        max_weight = float(rtc_config.get('max_guidance_weight', 5.0))
        scales = []
        for step in range(self._num_steps):
            progress = float(step) / float(self._num_steps)
            scales.append(
                compute_guidance_weight(progress, max_weight) /
                float(self._num_steps))
        scale_buffer.copy_(torch.tensor(scales, dtype=torch.float32))

    def _triton_forward(
        self,
        images_nhwc,
        prompt_embeds,
        prompt_len,
        diffusion_noise,
        prev_actions=None,
        prefix_len=0,
        rtc_config=None,
    ):
        buffers = self._triton_bufs
        buffers['observation_images_normalized'].copy_(images_nhwc)
        start = self.num_views * 256
        buffers['encoder_x'][start:start + prompt_len].copy_(prompt_embeds)
        buffers['valid_encoder_len'].fill_(start + prompt_len)
        buffers['decoder_rope_weights'].copy_(
            self._get_decoder_rope_weights(prompt_len))
        buffers['diffusion_noise'].copy_(diffusion_noise)
        self._update_runtime_adarms_mods(prefill_len=0)
        self._prepare_guidance(prev_actions, prefix_len, rtc_config)

        if not self._cuda_graph_ready:
            self._build_cuda_graph()
        self._cuda_graph.replay()
        return buffers['diffusion_noise']

    def predict_action(
        self,
        images,
        lang_tokens,
        states,
        img_masks=None,
        lang_masks=None,
        past_key_values=None,
        noise=None,
        prev_actions=None,
        prefix_len=0,
        rtc_config=None,
        *args,
        **kwargs,
    ):
        if not self._triton_ready:
            self.materialize_inference_weights(
                num_views=self.num_views,
                max_prompt_len=self.triton_max_prompt_len,
                chunk_size=self.n_action_steps,
                num_steps=self.num_steps,
            )
            self._triton_ready = True

        pixel_values = images.unflatten(1, (-1, 3))[0]
        images_nhwc = pixel_values.permute(0, 2, 3, 1).contiguous().bfloat16()
        prompt_len = (
            int(lang_masks[0].sum().item())
            if lang_masks is not None else lang_tokens.shape[1])
        embed_device = self.llm_backbone.embed_tokens.weight.device
        prompt_tokens = lang_tokens[0, :prompt_len].to(embed_device)
        lang_emb = self.llm_backbone.embed_tokens(prompt_tokens)
        lang_emb = (lang_emb * math.sqrt(lang_emb.shape[-1])).to(
            device='cuda', dtype=torch.bfloat16)

        # fp32 to match diffusion_noise's fp32 integration state.
        if noise is None:
            noise_t = torch.randn(
                self.n_action_steps,
                self.max_action_dim,
                dtype=torch.float32,
                device=states.device,
            )
        else:
            noise_t = noise[0].to(dtype=torch.float32)
        if noise_t.shape[-1] < self._action_dim:
            noise_t = torch.nn.functional.pad(
                noise_t, (0, self._action_dim - noise_t.shape[-1]))

        denoised = self._triton_forward(
            images_nhwc,
            lang_emb,
            prompt_len,
            noise_t,
            prev_actions=prev_actions,
            prefix_len=prefix_len,
            rtc_config=rtc_config,
        )
        # clone(): denoised is now fp32, so .float() below is a dtype-only
        # no-op (same storage) instead of an implicit copy. denoised aliases
        # the CUDA graph's static diffusion_noise buffer, which the next
        # predict_action call overwrites in place on graph replay. Without
        # an explicit copy, a caller holding this return value would see it
        # silently change after the next inference call.
        return denoised[:, :self.max_action_dim].unsqueeze(0).float().clone()
