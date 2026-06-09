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
import torch.nn.functional as F

from fluxvla.engines import PROJECTORS


@PROJECTORS.register_module()
class PixelUnshuffleProjector(nn.Module):
    """Spatially reduce ViT patch tokens before projecting to LLM width.

    The default ``mean`` reduction keeps the linear projector shape identical
    to ``LinearProjector``. This lets the module initialize from existing
    PI0.5 projector weights while reducing visual prefix length.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 grid_size: int = 16,
                 downscale_factor: int = 2,
                 reduction: str = 'mean') -> None:
        super().__init__()
        if downscale_factor < 1:
            raise ValueError('downscale_factor must be >= 1')
        if grid_size % downscale_factor != 0:
            raise ValueError(
                'grid_size must be divisible by downscale_factor')
        if reduction not in ('mean', 'concat'):
            raise ValueError("reduction must be one of ['mean', 'concat']")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.downscale_factor = downscale_factor
        self.reduction = reduction

        projector_in_dim = (
            in_dim if reduction == 'mean'
            else in_dim * downscale_factor * downscale_factor)
        self.projector = nn.Linear(projector_in_dim, out_dim, bias=True)

    @property
    def tokens_per_view(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def output_tokens_per_view(self) -> int:
        out_grid = self.grid_size // self.downscale_factor
        return out_grid * out_grid

    def _reduce_tokens(self, input_features: torch.Tensor) -> torch.Tensor:
        bsize, total_tokens, channels = input_features.shape
        if channels != self.in_dim:
            raise ValueError(
                f'Expected feature dim {self.in_dim}, got {channels}')
        if total_tokens % self.tokens_per_view != 0:
            raise ValueError(
                f'Expected token count to be divisible by '
                f'{self.tokens_per_view}, got {total_tokens}')

        num_views = total_tokens // self.tokens_per_view
        factor = self.downscale_factor
        if factor == 1:
            return input_features

        x = input_features.view(bsize, num_views, self.grid_size,
                                self.grid_size, channels)
        x = x.permute(0, 1, 4, 2, 3).reshape(bsize * num_views, channels,
                                             self.grid_size, self.grid_size)
        x = F.pixel_unshuffle(x, factor)

        out_grid = self.grid_size // factor
        if self.reduction == 'mean':
            x = x.view(bsize, num_views, channels, factor * factor, out_grid,
                       out_grid)
            x = x.mean(dim=3)
            x = x.permute(0, 1, 3, 4, 2).reshape(
                bsize, num_views * out_grid * out_grid, channels)
        else:
            x = x.view(bsize, num_views, channels * factor * factor,
                       out_grid, out_grid)
            x = x.permute(0, 1, 3, 4, 2).reshape(
                bsize, num_views * out_grid * out_grid,
                channels * factor * factor)

        return x

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.projector(self._reduce_tokens(input_features))


@PROJECTORS.register_module()
class PixelUnshuffleProjectorInference(PixelUnshuffleProjector):

    def prepare_triton(self, prefix='') -> dict:
        return {
            f'{prefix}_w':
            self.projector.weight.data.T.contiguous().bfloat16().cuda(),
            f'{prefix}_b':
            self.projector.bias.data.bfloat16().cuda(),
        }


@PROJECTORS.register_module()
class PixelUnshuffleMLPProjector(PixelUnshuffleProjector):
    """Pixel-unshuffle concat, channel MLP, then the original projector."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 grid_size: int = 16,
                 downscale_factor: int = 2,
                 reduction: str = 'concat',
                 hidden_dim: int = None) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            grid_size=grid_size,
            downscale_factor=downscale_factor,
            reduction=reduction)
        if reduction != 'concat':
            raise ValueError(
                'PixelUnshuffleMLPProjector expects reduction="concat"')

        hidden_dim = hidden_dim or in_dim
        concat_dim = in_dim * downscale_factor * downscale_factor
        self.channel_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim, bias=True),
        )
        self.projector = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        x = self._reduce_tokens(input_features)
        x = self.channel_mlp(x)
        return self.projector(x)


@PROJECTORS.register_module()
class PixelUnshuffleMLPProjectorInference(PixelUnshuffleMLPProjector):

    def prepare_triton(self, prefix='') -> dict:
        return {
            f'{prefix}_channel_mlp_up_w':
            self.channel_mlp[0].weight.data.T.contiguous().bfloat16().cuda(),
            f'{prefix}_channel_mlp_up_b':
            self.channel_mlp[0].bias.data.bfloat16().cuda(),
            f'{prefix}_channel_mlp_down_w':
            self.channel_mlp[2].weight.data.T.contiguous().bfloat16().cuda(),
            f'{prefix}_channel_mlp_down_b':
            self.channel_mlp[2].bias.data.bfloat16().cuda(),
            f'{prefix}_w':
            self.projector.weight.data.T.contiguous().bfloat16().cuda(),
            f'{prefix}_b':
            self.projector.bias.data.bfloat16().cuda(),
        }
