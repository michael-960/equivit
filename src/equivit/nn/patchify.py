from ..geometry import Honeycomb, Triangle, decompose_set_action
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from ..geometry import Group, Lattice, GroupAction

from .lattice_irrep_handler import GroupActionIrrepProjectionCalculator

# We should provide flexibility in whether the main model contains 



class Patchify(nn.Module):
    patch_inds: torch.Tensor
    def __init__(self):
        super().__init__()
        self._setup_patch_indices()

    def _setup_patch_indices(self):
        raise NotImplementedError("This method should be implemented by subclasses to set up the patch indices for the specific lattice structure of the patches.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should have shape (*, L, C)

        # assert H % self.patch_size == 0 and W % self.patch_size == 0, f"Image size ({H}x{W}) must be divisible by patch size ({self.patch_size})"
        # x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # (B, C, H//P, W//P, P, P)
        # x = x.contiguous().view(B, C, -1, self.patch_size*self.patch_size) # (B, C, num_patches, patch_size*patch_size)

        x_unfolded = x[...,self.patch_inds,:] # (*, num_patches, Lpatch, C)
        return x_unfolded


class SquarePatchify(Patchify):
    def __init__(self, img_size: int, patch_size: int):
        self.img_size = img_size
        self.patch_size = patch_size
        super().__init__()

    def _setup_patch_indices(self):
        L = self.img_size**2
        P = self.patch_size

        patch_inds = []

        for i in range(0, self.img_size, P):
            for j in range(0, self.img_size, P):
                _inds = []
                for di in range(P):
                    for dj in range(P):
                        _inds.append((i+di)*self.img_size + (j+dj))
                patch_inds.append(_inds)
        self.patch_inds = torch.tensor(patch_inds, dtype=torch.long)


