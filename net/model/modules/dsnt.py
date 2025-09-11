import torch 
import torch.nn.functional as F

def to_probability_distributions(volume):
    B, C, D, H, W = volume.shape
    # softmax over voxel domain: do reshape softmax for stability & correctness
    voxels = volume.view(B, C, -1)
    probs = F.softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs

def dsnt_3d_separable(volume):
    # spacing: (dz, dy, dx) in real physical units
    probs = to_probability_distributions(volume)
    _, _, D, H, W = probs.shape
    # marginals
    p_z = probs.sum(dim=(3, 4))  # [B, C, D]
    p_y = probs.sum(dim=(2, 4))  # [B, C, H]
    p_x = probs.sum(dim=(2, 3))  # [B, C, W]

    z_coords = torch.arange(D, device=volume.device, dtype=volume.dtype)
    y_coords = torch.arange(H, device=volume.device, dtype=volume.dtype)
    x_coords = torch.arange(W, device=volume.device, dtype=volume.dtype)

    x = (p_x * x_coords.view(1, 1, W)).sum(dim=-1)
    y = (p_y * y_coords.view(1, 1, H)).sum(dim=-1)
    z = (p_z * z_coords.view(1, 1, D)).sum(dim=-1)

    coords = torch.stack([x, y, z], dim=-1)
    return coords
