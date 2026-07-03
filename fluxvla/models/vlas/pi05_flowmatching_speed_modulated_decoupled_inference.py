# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Speed-modulated RTC-like decoupled inference."""

import torch

from fluxvla.engines import VLAS
from .pi05_flowmatching_decoupled_inference import (
    transformer_decoder_ultra_v2_prefix_clamp,
)
from .pi05_flowmatching_speed_modulated_inference import (
    PI05FlowMatchingSpeedModulatedInference,
)


@VLAS.register_module()
class PI05FlowMatchingSpeedModulatedDecoupledInference(
        PI05FlowMatchingSpeedModulatedInference):
    """Full/decoder-only alternating inference for speed-modulated TempoVLA.

    A full call refreshes VLM/encoder features and returns all n_action_steps.
    The eval runner executes the prefix. The next decoder-only call reuses the
    last full call's encoder K/V, clamps the executed prefix during denoising,
    and the runner executes the generated suffix.
    """

    def __init__(self, exec_chunk_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_chunk_size = exec_chunk_size
        self._cuda_graph_decoder_prefix = None
        self._cuda_graph_decoder_prefix_ready = False
        self._cuda_graph_decoder_prefix_len = None

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
            self._triton_weights,
            self._triton_bufs,
            self._encoder_seq_len,
            self._num_decoder_layers,
            self._num_steps,
            prefix_len=prefix_len)

    def _build_decoder_only_prefix_graph(self, prefix_len):
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

    def _ensure_decoder_prefix_graph(self, prefix_len):
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
                chunk_size,
                self.max_action_dim,
                dtype=torch.bfloat16,
                device=device)
        else:
            noise_t = noise[0].to(dtype=torch.bfloat16, device=device)
        if noise_t.shape[-1] < 32:
            pad = torch.zeros(
                noise_t.shape[0],
                32 - noise_t.shape[-1],
                dtype=torch.bfloat16,
                device=device)
            noise_t = torch.cat([noise_t, pad], dim=-1)
        return noise_t

    def _write_prefix_actions(self, prefix_actions, prefix_len):
        if prefix_actions is None:
            raise ValueError('prefix_actions is required for decoupled mode')
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
                prefix_t.shape[0],
                32 - prefix_t.shape[-1],
                dtype=torch.bfloat16,
                device=device)
            prefix_t = torch.cat([prefix_t, pad], dim=-1)
        self._triton_bufs['action_prefix'][:prefix_len].copy_(
            prefix_t[:prefix_len, :32])

    def predict_action_decoder_only_prefix(self,
                                           prefix_actions,
                                           prefix_len,
                                           noise=None,
                                           tempo_speed=None):
        """Generate the suffix while reusing the last full call's visual cache."""
        if tempo_speed is not None:
            tempo_speed = self._tempo_speed_to_float(tempo_speed)
            if self._current_tempo_speed != tempo_speed:
                self.set_tempo_speed(tempo_speed)

        self._ensure_decoder_prefix_graph(prefix_len)

        noise_t = self._prepare_decoder_noise(noise)
        self._write_prefix_actions(prefix_actions, prefix_len)
        self._triton_bufs['noise_init_prefix'][:prefix_len].copy_(
            noise_t[:prefix_len])
        self._triton_bufs['diffusion_noise'].copy_(noise_t)
        self._cuda_graph_decoder_prefix.replay()

        return (self._triton_bufs['diffusion_noise']
                [:, :self.max_action_dim].unsqueeze(0).float())
