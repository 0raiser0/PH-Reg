# ----------------------------------------------------------------------
# All the denosing will have a similar structure
# 
# Just barly change the model 
# ----------------------------------------------------------------------

import sys, os
import torch
import torch.nn.functional as F
import numpy as np


def prepare_patch_idxs(num_patches):
    patch_idxs = torch.stack(
        torch.meshgrid(
            torch.arange(num_patches, dtype=torch.int32),
            torch.arange(num_patches, dtype=torch.int32),
            indexing='ij'
        ),
        dim=0
    ) # shape: (2, num_patches, num_patches)
    # We'll store as (B, 2, num_patches, num_patches) after broadcasting
    # patch_idxs_unsqueezed = patch_idxs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    return patch_idxs
    

def prepare_flip(counts, seed):
    rng = np.random.default_rng(seed=seed)
    flip = rng.integers(0, 2, counts)
    flip[0] = 0
    
    return flip

def prepare_all_offsets(num_patches, shift_frac=0.15, counts=10, seed=42) -> torch.Tensor: 
    # num_patches = int(input_resolution // patch_size)
    # assert input_resolution == int(num_patches * patch_size) 
    # how many patches we can shift
    shift_patch = int(round(shift_frac * num_patches))
    rng = np.random.default_rng(seed=seed)
    
    # We'll draw a large pool of offsets (x,y) from [-shift_patch, shift_patch]
    rand_x = rng.integers(-shift_patch, shift_patch, size=counts*10, endpoint=True)
    rand_y = rng.integers(-shift_patch, shift_patch, size=counts*10, endpoint=True)
    all_offsets = np.stack([rand_x, rand_y], axis=1)[None] # [None] to expand 1 dim
    # shape: (1, counts*10, 2)
    
    all_offsets_torch = torch.from_numpy(all_offsets).float()
    
    return all_offsets_torch


def farthest_point_sample(all_offsets, num_sample, seed) -> torch.Tensor:
    """
    Given a set of XY points [1, N, 2] (all_offsets), selects 'num_sample' points using farthest point sampling.
    This is basically the same approach from your script.
    """
    device = all_offsets.device
    B, N, C = all_offsets.shape
    
    # center points, used to compute the Euclidean distance from reference offset points.
    centroids = torch.zeros(size=(B, num_sample), dtype=torch.long, device=device)
    # represents the Euclidean distance between centroids and offset points
    distance = torch.full(size=(B, N), fill_value=float('inf'), device=device)
    # the farthest point index, just one value
    torch.manual_seed(seed)
    farthest = torch.randint(low=0, high=N, size=(B,), dtype=torch.long, device=device)
    #  tensor([0])
    batch_indices = torch.arange(end=B, dtype=torch.long, device=device)
    
    for i in range(num_sample):
        # give the farthest point index to 'centroids'
        centroids[:, i] = farthest
        # select a center point
        center = all_offsets[batch_indices, farthest, :].view(B, 1, 2)
        # compute distance between center point and each offset point in the pair
        dist = torch.sum((all_offsets - center) ** 2, dim=-1)
        """
        Please ask me. I will give the explaination from deepseek R1.
        It is difficult to describe it here.
        """
        mask = dist < distance
        distance[mask] = dist[mask]
        # return the index of the farthest point, using [1] to select index
        farthest = torch.max(distance, dim=-1)[1]
        
    return centroids


def shuffle_shift(input_tensor, offset_height, offset_width, infill=-100.0):
    """
    Shifts the tensor spatially by (offset_height, offset_width).
    Any area that goes 'out of bounds' is filled by 'infill'.
    input_tensor has shape (B, C, H, W).
    """
    C, H, W = input_tensor.shape
    
    """
    We want the region that remains after shifting:
    If 'offset_height' > 0, we are shifting 'upwards', so the bottom rows become infill.
    Construct slices carefully in PyTorch.
    We'll do it by indexing and then padding.
    For positive 'offset_height', we shift forward in the H dimension. Negative means shift backward, etc.
    """
    """
    If 'offset_height' < 0, we cut the image from 0 to H + offset_height.
    Similarly, when 'offset_width' < 0
    Then the image are shifted downwards.
    """
    # Original Height: 0 -> H from top to down
    # Similar to Original Width
    low_height = max(0, offset_height)
    up_height = min(H, H + offset_height)
    low_width = max(0, offset_width)
    up_width = min(W, W + offset_width)
    
    # Slicing region that remains after shifting
    src_height = slice(low_height, up_height)
    src_width = slice(low_width, up_width)
    
    dst_height = slice(low_height - offset_height, up_height - offset_height)
    dst_width = slice(low_width - offset_width, up_width - offset_width)
    
    # Create an output tensor of the same size, filled with infill
    out = torch.full_like(input_tensor, fill_value=infill)
    
    # Copy the valid region over
    out[:, dst_height, dst_width] = input_tensor[:, src_height, src_width]
    "Maybe we can update infill value here."
    return out
    
# if __name__ == "__main__":
#     # batch_indices = torch.arange(1, dtype=torch.long)
#     # print(batch_indices)
#     all_offsets = prepare_all_offsets()
#     centroids = farthest_point_sample(all_offsets, num_sample=10)