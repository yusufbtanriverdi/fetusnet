import torch 

def dsnt_3d_separable(probs, device, dtype):
    # spacing: (dz, dy, dx) in real physical units
    _, _, D, H, W = probs.shape
    # marginals
    p_z = probs.sum(dim=(3, 4))  # [B, C, D]
    p_y = probs.sum(dim=(2, 4))  # [B, C, H]
    p_x = probs.sum(dim=(2, 3))  # [B, C, W]

    z_coords = torch.arange(D, device=device, dtype=dtype)
    y_coords = torch.arange(H, device=device, dtype=dtype)
    x_coords = torch.arange(W, device=device, dtype=dtype)

    x = (p_x * x_coords.view(1, 1, W)).sum(dim=-1)
    y = (p_y * y_coords.view(1, 1, H)).sum(dim=-1)
    z = (p_z * z_coords.view(1, 1, D)).sum(dim=-1)

    coords = torch.stack([x, y, z], dim=-1)
    return coords
