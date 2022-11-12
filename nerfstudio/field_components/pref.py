# PREF as an encoder https://arxiv.org/pdf/2205.13524.pdf
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.field_components.encodings import Encoding


class SpatialEncoding(Encoding):
    """
    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """
    grid: TensorType[1, "num_components", "resolution", "resolution", "resolution"]

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.grid = nn.Parameter(init_scale * torch.randn((1, num_components, resolution, resolution, resolution), dtype=torch.float32))

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        # Take all the dimensions of the input tensor, except the last one
        # (which is the input dimension)
        shape = in_tensor.shape[:-1]
        # Add the output dimension as the last dimension
        out_shape = shape + (self.get_out_dim(),)
        in_tensor = in_tensor.reshape(-1, 3)
        out_tensor = torch.nn.functional.grid_sample(self.grid, in_tensor[None, None, None, :, :], align_corners=True)
        return out_tensor.view(out_shape)  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """
        grid = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution, resolution), mode="bilinear", align_corners=True
        )

        self.grid = torch.nn.Parameter(grid)
        self.resolution = resolution


class FrequencyEncoding(Encoding):
    pass

class PrefEncoding(Encoding):
    pass