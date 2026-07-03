# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speed-modulated inference with Triton/CUDA Graph acceleration."""

import torch
import torch.nn as nn

from fluxvla.engines import VLAS
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi05_flowmatching_inference import PI05FlowMatchingInference

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatchingSpeedModulatedInference(PI05FlowMatchingInference):
    """Inference variant of PI05FlowMatchingSpeedModulated with Triton acceleration.

    Extends PI05FlowMatchingInference to support TempoVLA speed conditioning
    with RealTimeVLA-inspired optimizations:

    1. Speed embedding pre-computation and caching (eliminates speed_mlp overhead)
    2. CUDA Graph reuse across different speeds (no graph rebuild)
    3. Optional ultra-optimized fusion kernels (aggressive operator fusion)

    The speed MLP is evaluated before CUDA graph replay and added to the
    decoder timestep conditioning after the time MLP, matching the training
    implementation in PI05FlowMatchingSpeedModulated.embed_suffix().

    Args:
        speed_mlp_hidden_dim (int): Hidden dimension for speed MLP. Default: 256.
        default_tempo_speed (float): Default speed for inference. Default: 1.0.
        use_ultra_fusion (bool): Use aggressive RealTimeVLA fusion kernels.
            Default: False. NOTE: verified to give NO speedup (slightly slower)
            when CUDA Graph is enabled, because CUDA Graph already eliminates
            kernel-launch overhead — the very thing fusion targets. Kept for
            A/B comparison only. Numerically identical to the standard path.
        *args, **kwargs: Forwarded to PI05FlowMatchingInference.
    """

    def __init__(
        self,
        speed_mlp_hidden_dim: int = 256,
        default_tempo_speed: float = 1.0,
        use_ultra_fusion: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_ultra_fusion = use_ultra_fusion

        # Speed modulation MLP (same as training version)
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(speed_mlp_hidden_dim, self.proj_width)
        )

        self.default_tempo_speed = default_tempo_speed
        self._current_tempo_speed = None
        self._speed_emb_buffer = None

        # Pre-computed speed embedding cache (initialized during prepare_triton_inference)
        self._speed_embedding_cache = None

        overwatch.info(
            f"Initialized speed-modulated inference MLP: "
            f"1 -> {speed_mlp_hidden_dim} -> {self.proj_width}, "
            f"default_tempo_speed={default_tempo_speed}, "
            f"use_ultra_fusion={use_ultra_fusion}"
        )

    def _tempo_speed_to_float(self, tempo_speed) -> float:
        """Convert supported speed inputs to a scalar float for batch-1 inference."""
        if tempo_speed is None:
            return float(self.default_tempo_speed)
        if torch.is_tensor(tempo_speed):
            if tempo_speed.numel() != 1:
                raise ValueError(
                    'PI05FlowMatchingSpeedModulatedInference supports one '
                    f'tempo_speed value per predict_action call, got shape '
                    f'{tuple(tempo_speed.shape)}')
            return float(tempo_speed.detach().float().reshape(-1)[0].item())
        if isinstance(tempo_speed, (list, tuple)):
            if len(tempo_speed) != 1:
                raise ValueError(
                    'PI05FlowMatchingSpeedModulatedInference supports one '
                    f'tempo_speed value per predict_action call, got '
                    f'{len(tempo_speed)} values')
            return self._tempo_speed_to_float(tempo_speed[0])
        return float(tempo_speed)

    def _compute_speed_embedding(self, tempo_speed: float) -> torch.Tensor:
        """Compute speed embedding for a given speed.

        Uses cache if available to avoid recomputing speed_mlp forward.

        Args:
            tempo_speed (float): Speed value (e.g., 0.5, 1.0, 2.0).

        Returns:
            torch.Tensor: Speed embedding of shape (1, proj_width).
        """
        tempo_speed = self._tempo_speed_to_float(tempo_speed)

        # Use cached embedding if available
        if self._speed_embedding_cache is not None and tempo_speed in self._speed_embedding_cache:
            return self._speed_embedding_cache[tempo_speed]

        # Fallback: compute on-the-fly (used before prepare_triton_inference)
        speed_param = next(self.speed_mlp.parameters())
        speed_tensor = torch.tensor(
            [[tempo_speed]],
            dtype=speed_param.dtype,
            device=speed_param.device,
        )
        with torch.no_grad():
            speed_emb = self.speed_mlp(speed_tensor).to(torch.bfloat16)
        return speed_emb

    def _prepare_adarms_cond(self, num_steps, tempo_speed=None):
        """Pre-compute base time embeddings and cache speed modulation.

        Args:
            num_steps (int): Number of denoising steps.
            tempo_speed (float, optional): Speed for conditioning.
                If None, uses self.default_tempo_speed.

        Returns:
            torch.Tensor: Base sinusoidal time embeddings,
                shape (num_steps, proj_width).
        """
        # Parent returns raw sinusoidal embeddings. The accelerated decoder
        # applies time_mlp_in/out inside the graph, so speed must be added
        # after that MLP to match the training path.
        time_embs = super()._prepare_adarms_cond(num_steps)

        tempo_speed = self._tempo_speed_to_float(tempo_speed)
        speed_emb = self._compute_speed_embedding(tempo_speed)

        # Cache for later use
        self._current_tempo_speed = tempo_speed
        self._speed_emb_buffer = speed_emb

        return time_embs

    def prepare_triton_inference(
        self,
        num_views,
        max_prompt_len,
        chunk_size,
        num_steps,
        tempo_speed=None
    ):
        """Collect weights and build the Triton pipeline with speed conditioning.

        Args:
            num_views (int): The number of views.
            max_prompt_len (int): The maximum prompt length.
            chunk_size (int): The chunk size.
            num_steps (int): Denoising steps.
            tempo_speed (float, optional): Speed for conditioning.
                If None, uses self.default_tempo_speed.
        """
        # Store tempo_speed for use in parent's prepare
        tempo_speed = self._tempo_speed_to_float(tempo_speed)

        # Pre-compute ALL speed embeddings for common speeds (OpenVLA-style caching)
        common_speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        self._speed_embedding_cache = {}
        speed_param = next(self.speed_mlp.parameters())

        overwatch.info(f"Pre-computing speed embeddings for speeds: {common_speeds}")
        with torch.no_grad():
            for speed in common_speeds:
                speed_tensor = torch.tensor(
                    [[speed]],
                    dtype=speed_param.dtype,
                    device=speed_param.device,
                )
                speed_emb = self.speed_mlp(speed_tensor).to(torch.bfloat16)
                self._speed_embedding_cache[speed] = speed_emb

        overwatch.info(f"Speed embedding cache ready with {len(self._speed_embedding_cache)} entries")

        # Collect all weights
        self._triton_weights = {}
        self._triton_weights.update(self.vision_backbone.prepare_triton())
        self._triton_weights.update(
            self.llm_backbone.prepare_triton(role='llm'))
        self._triton_weights.update(
            self.llm_expert.prepare_triton(role='expert'))
        self._triton_weights.update(
            self.projector.prepare_triton(
                prefix='encoder_multi_modal_projector'))
        self._triton_weights.update(self._prepare_action_time_triton())

        decoder_time_embeds = self._prepare_adarms_cond(num_steps, tempo_speed)

        # Allocate a STABLE speed-embedding buffer that the CUDA graph will
        # capture by pointer. set_tempo_speed() writes into this buffer in
        # place (copy_), so graph replay always reads the current speed without
        # a rebuild. Rebinding the dict entry (the previous approach) does NOT
        # work: a recorded graph captures the device pointer, not the dict.
        self._speed_emb_graph_buf = torch.zeros_like(self._speed_emb_buffer)
        self._speed_emb_graph_buf.copy_(self._speed_emb_buffer)
        self._triton_weights.update({
            'decoder_time_embeds': decoder_time_embeds,
            'decoder_speed_emb': self._speed_emb_graph_buf,
        })

        self._max_prompt_len = max_prompt_len
        self._num_steps = num_steps

        # Copy dimension setup from parent
        # Vision dimensions
        vit_cfg = self.vision_backbone.vision.vision_model.config
        self._vit_image_size = vit_cfg.image_size
        self._vit_num_patches = (vit_cfg.image_size // vit_cfg.patch_size)**2
        self._visual_grid_size = vit_cfg.image_size // vit_cfg.patch_size
        self._visual_token_downscale_factor = int(
            getattr(self.projector, 'downscale_factor', 1))
        if self._visual_grid_size % self._visual_token_downscale_factor != 0:
            raise ValueError(
                'visual grid size must be divisible by '
                'visual_token_downscale_factor')
        self._visual_tokens_per_view = (
            self._visual_grid_size //
            self._visual_token_downscale_factor)**2
        self._vit_hidden = vit_cfg.hidden_size
        self._vit_intermediate = vit_cfg.intermediate_size
        self._num_vit_layers = vit_cfg.num_hidden_layers

        # Encoder dimensions
        enc_cfg = self.llm_backbone.config
        self._enc_hidden = enc_cfg.hidden_size
        self._enc_intermediate = enc_cfg.intermediate_size
        self._num_encoder_layers = len(self.llm_backbone.layers)

        # Decoder dimensions
        dec_cfg = self.llm_expert.config
        self._dec_hidden = dec_cfg.hidden_size
        self._dec_intermediate = dec_cfg.intermediate_size
        self._num_decoder_layers = len(self.llm_expert.layers)
        self._head_dim = enc_cfg.head_dim
        self._num_kv_heads = enc_cfg.num_attention_heads
        self._dec_style_dim = 3 * self._dec_hidden

        # Sequence lengths
        self._encoder_seq_len = (
            num_views * self._visual_tokens_per_view + max_prompt_len)
        self._decoder_seq_len = chunk_size
        self._action_dim = self.max_action_dim

        # Allocate buffers
        self._init_buffers()

        self._init_rope_table()

        self._triton_ready = True
        self._cuda_graph = None
        self._cuda_graph_ready = False

        overwatch.info(
            f'Triton inference ready: tempo_speed={tempo_speed}, '
            f'num_steps={num_steps}, '
            f'encoder_seq_len={self._encoder_seq_len}, '
            f'decoder_seq_len={self._decoder_seq_len}'
        )

    def set_tempo_speed(self, tempo_speed: float):
        """Update tempo_speed after Triton preparation.

        This allows dynamic speed changes without re-preparing the entire graph.
        Speed embeddings are looked up from cache, avoiding speed_mlp forward.
        CUDA graph is NOT invalidated if the speed is in cache.

        Args:
            tempo_speed (float): New speed value.
        """
        if not self._triton_ready:
            raise RuntimeError(
                "Must call prepare_triton_inference() before set_tempo_speed()"
            )
        tempo_speed = self._tempo_speed_to_float(tempo_speed)

        # Update speed embedding (from cache if available)
        speed_emb = self._compute_speed_embedding(tempo_speed)
        self._current_tempo_speed = tempo_speed
        self._speed_emb_buffer = speed_emb

        # Write into the graph-captured buffer IN PLACE. The recorded CUDA
        # graph reads decoder_speed_emb by pointer, so copy_ (not rebind) is
        # what makes the new speed take effect on the next replay.
        self._speed_emb_graph_buf.copy_(speed_emb)

        # NOTE: decoder_time_embeds is speed-INDEPENDENT (it is the raw
        # sinusoidal timestep embedding; speed is added separately via
        # decoder_speed_emb inside the decoder). It must NOT be recomputed or
        # rebound here — doing so would swap in a fresh tensor the graph never
        # captured, silently breaking time conditioning after the first speed
        # change.

        # CUDA graph is preserved: it reads decoder_speed_emb in place.

        overwatch.info(f'Updated tempo_speed to {tempo_speed} (from cache, graph preserved)')

    def _run_forward(self):
        """Override to pass use_ultra_fusion flag to decoder."""
        from .pi05_flowmatching_inference import pi05_model
        pi05_model(
            self._triton_weights, self._triton_bufs, self.num_views,
            self._encoder_seq_len, self._num_vit_layers,
            self._num_encoder_layers, self._num_decoder_layers,
            self._num_steps, self._visual_tokens_per_view,
            self._visual_grid_size,
            self._visual_token_downscale_factor,
            use_ultra_fusion=self.use_ultra_fusion
        )

    def predict_action(
        self,
        images,
        lang_tokens,
        states,
        img_masks=None,
        lang_masks=None,
        past_key_values=None,
        noise=None,
        tempo_speed=None,
        *args,
        **kwargs
    ):
        """Predict action with optional tempo_speed override.

        Args:
            tempo_speed (float, optional): Override the prepared tempo_speed.
                If provided and different from current, will call set_tempo_speed().
            Other args: Same as parent predict_action.

        Returns:
            torch.Tensor: Predicted actions.
        """
        # Handle tempo_speed override
        if tempo_speed is not None:
            tempo_speed = self._tempo_speed_to_float(tempo_speed)
            if self._current_tempo_speed != tempo_speed:
                if not self._triton_ready:
                    # First call: prepare with this speed
                    self.prepare_triton_inference(
                        num_views=self.num_views,
                        max_prompt_len=self.triton_max_prompt_len,
                        chunk_size=self.n_action_steps,
                        num_steps=self.num_steps,
                        tempo_speed=tempo_speed
                    )
                    self._triton_ready = True
                else:
                    # Already prepared: update speed
                    self.set_tempo_speed(tempo_speed)

        # Call parent predict_action (which uses the prepared time embeddings)
        return super().predict_action(
            images=images,
            lang_tokens=lang_tokens,
            states=states,
            img_masks=img_masks,
            lang_masks=lang_masks,
            past_key_values=past_key_values,
            noise=noise,
            *args,
            **kwargs
        )
