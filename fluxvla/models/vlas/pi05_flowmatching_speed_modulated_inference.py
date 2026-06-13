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

    Extends PI05FlowMatchingInference to support TempoVLA speed conditioning.
    The speed MLP is evaluated before CUDA graph replay and folded into the
    precomputed decoder timestep embeddings used by the Triton pipeline.

    Args:
        speed_mlp_hidden_dim (int): Hidden dimension for speed MLP. Default: 256.
        default_tempo_speed (float): Default speed for inference. Default: 1.0.
        *args, **kwargs: Forwarded to PI05FlowMatchingInference.
    """

    def __init__(
        self,
        speed_mlp_hidden_dim: int = 256,
        default_tempo_speed: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Speed modulation MLP (same as training version)
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(speed_mlp_hidden_dim, self.proj_width)
        )

        self.default_tempo_speed = default_tempo_speed
        self._current_tempo_speed = None
        self._speed_emb_buffer = None

        overwatch.info(
            f"Initialized speed-modulated inference MLP: "
            f"1 -> {speed_mlp_hidden_dim} -> {self.proj_width}, "
            f"default_tempo_speed={default_tempo_speed}"
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

        Args:
            tempo_speed (float): Speed value (e.g., 0.5, 1.0, 2.0).

        Returns:
            torch.Tensor: Speed embedding of shape (1, proj_width).
        """
        tempo_speed = self._tempo_speed_to_float(tempo_speed)
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
        """Pre-compute time embeddings with speed modulation.

        Args:
            num_steps (int): Number of denoising steps.
            tempo_speed (float, optional): Speed for conditioning.
                If None, uses self.default_tempo_speed.

        Returns:
            torch.Tensor: Time embeddings with speed modulation,
                shape (num_steps, proj_width).
        """
        # Get base time embeddings
        time_embs = super()._prepare_adarms_cond(num_steps)

        # Add speed modulation
        tempo_speed = self._tempo_speed_to_float(tempo_speed)

        speed_emb = self._compute_speed_embedding(tempo_speed)

        # Broadcast and add speed embedding to all time steps
        # time_embs: (num_steps, proj_width)
        # speed_emb: (1, proj_width)
        time_embs = time_embs + speed_emb

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

        # Prepare time embeddings with speed modulation
        self._triton_weights.update({
            'decoder_time_embeds': self._prepare_adarms_cond(num_steps, tempo_speed)
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
        Note: This requires re-computing time embeddings, which is relatively cheap.

        Args:
            tempo_speed (float): New speed value.
        """
        if not self._triton_ready:
            raise RuntimeError(
                "Must call prepare_triton_inference() before set_tempo_speed()"
            )
        tempo_speed = self._tempo_speed_to_float(tempo_speed)

        # Re-compute time embeddings with new speed
        self._triton_weights['decoder_time_embeds'] = self._prepare_adarms_cond(
            self._num_steps, tempo_speed
        )

        # Invalidate CUDA graph (needs rebuild with new embeddings)
        self._cuda_graph_ready = False
        self._cuda_graph = None

        overwatch.info(f'Updated tempo_speed to {tempo_speed}')

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
