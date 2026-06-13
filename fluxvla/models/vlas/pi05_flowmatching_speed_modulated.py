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

import torch
import torch.nn as nn

from fluxvla.engines import VLAS
from fluxvla.engines.utils.overwatch import initialize_overwatch
from .pi05_flowmatching import PI05FlowMatching

overwatch = initialize_overwatch(__name__)


@VLAS.register_module()
class PI05FlowMatchingSpeedModulated(PI05FlowMatching):
    """PI05 Flow Matching with Speed-Modulated RMSNorm (TempoVLA Method 2).

    Implements speed conditioning via a small MLP that embeds the speed scalar
    and adds it to the flow-matching timestep embedding. This modulates the
    action expert's RMSNorm conditioning.

    Reference: TempoVLA paper Section 3.3, Method 2.

    Args:
        speed_mlp_hidden_dim (int): Hidden dimension for speed MLP. Default: 256.
        **kwargs: Arguments passed to PI05FlowMatching.
    """

    def __init__(self, speed_mlp_hidden_dim: int = 256, **kwargs):
        super().__init__(**kwargs)

        # Speed modulation MLP: scalar speed -> d_mod embedding
        # d_mod should match proj_width (same as time_emb dimension)
        self.speed_mlp = nn.Sequential(
            nn.Linear(1, speed_mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(speed_mlp_hidden_dim, self.proj_width)
        )

        overwatch.info(
            f"Initialized speed modulation MLP: 1 -> {speed_mlp_hidden_dim} -> {self.proj_width}"
        )

    def embed_suffix(self, states, noisy_actions, timestep):
        """Embed the suffix tokens with speed modulation.

        Speed is read from self._current_speed set by forward().

        Args:
            states (torch.Tensor): State tensor of shape (bsize, state_dim).
            noisy_actions (torch.Tensor): Noisy actions of shape
                (bsize, n_action_steps, action_dim).
            timestep (torch.Tensor): Timestep of shape (bsize,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (embs, pad_masks, att_masks, adarms_cond)
        """
        # Call parent to get base embeddings
        embs, pad_masks, att_masks, time_emb = super().embed_suffix(
            states, noisy_actions, timestep
        )

        # Add speed modulation to time_emb if speed is available
        speed = getattr(self, '_current_speed', None)
        if speed is not None:
            device = time_emb.device
            dtype = time_emb.dtype

            # Ensure speed is on correct device
            if speed.device != device:
                speed = speed.to(device)

            # Ensure speed has correct shape: (bsize, 1)
            if speed.dim() == 1:
                speed = speed.unsqueeze(-1)

            # Embed speed and add to time_emb
            speed_emb = self.speed_mlp(speed.type(dtype))  # (bsize, proj_width)

            # Handle different time_emb shapes
            if time_emb.dim() == 2:
                # Scalar timestep: (bsize, proj_width)
                time_emb = time_emb + speed_emb
            elif time_emb.dim() == 3:
                # Per-position timestep: (bsize, n_action_steps, proj_width)
                time_emb = time_emb + speed_emb.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected time_emb shape: {time_emb.shape}")

        return embs, pad_masks, att_masks, time_emb

    def forward(self, *args, tempo_speed=None, **kwargs):
        """Forward pass with speed conditioning.

        Args:
            tempo_speed: Speed values of shape (bsize,) or scalar. If None, defaults to 1.0.
            *args, **kwargs: Passed to parent forward.

        Returns:
            Model outputs.
        """
        # Extract batch size from images or lang_tokens
        if 'images' in kwargs and kwargs['images'] is not None:
            bsize = kwargs['images'].shape[0]
        elif len(args) > 0:
            bsize = args[0].shape[0]
        else:
            raise ValueError("Cannot determine batch size")

        # Get device from speed_mlp parameters
        device = next(self.speed_mlp.parameters()).device

        # Handle tempo_speed: convert to tensor if needed
        if tempo_speed is None:
            speed = torch.ones(bsize, device=device)
        elif isinstance(tempo_speed, (int, float)):
            speed = torch.full((bsize,), tempo_speed, device=device)
        elif isinstance(tempo_speed, (list, tuple)):
            # Handle list/tuple from collator (shouldn't happen with keys=['tempo_speed'])
            speed = torch.tensor(tempo_speed, device=device, dtype=torch.float32)
        elif torch.is_tensor(tempo_speed):
            speed = tempo_speed.to(device) if tempo_speed.device != device else tempo_speed
        else:
            raise TypeError(f"tempo_speed has unsupported type: {type(tempo_speed)}")

        # Store speed in instance variable for embed_suffix to access
        self._current_speed = speed

        # Call parent forward
        return super().forward(*args, **kwargs)

    def predict_action(self, *args, tempo_speed=None, **kwargs):
        """Predict action with speed conditioning.

        Args:
            tempo_speed: Speed value (scalar or tensor). If None, defaults to 1.0.
            *args, **kwargs: Passed to parent predict_action.

        Returns:
            Predicted actions.
        """
        # Extract batch size
        if 'images' in kwargs and kwargs['images'] is not None:
            bsize = kwargs['images'].shape[0]
        elif len(args) > 0:
            bsize = args[0].shape[0]
        else:
            bsize = 1

        # Get device from speed_mlp parameters
        device = next(self.speed_mlp.parameters()).device

        # Handle tempo_speed
        if tempo_speed is None:
            speed = torch.ones(bsize, device=device)
        elif isinstance(tempo_speed, (int, float)):
            speed = torch.full((bsize,), tempo_speed, device=device)
        elif isinstance(tempo_speed, (list, tuple)):
            speed = torch.tensor(tempo_speed, device=device, dtype=torch.float32)
        elif torch.is_tensor(tempo_speed):
            speed = tempo_speed.to(device) if tempo_speed.device != device else tempo_speed
        else:
            raise TypeError(f"tempo_speed has unsupported type: {type(tempo_speed)}")

        # Store speed for use in embed_suffix
        self._current_speed = speed

        # Call parent predict_action
        return super().predict_action(*args, **kwargs)
