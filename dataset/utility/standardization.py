# Reference
# Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network
# https://arxiv.org/abs/1806.07486
#
# ==============================================================================

import numpy as np
import torch
from dataset.preprocessing import affine3Dmatrix, affine_transform, grid_transform_3d

from dataset.utility.gtpp.utils import utils
from dataset.utility.gtpp.utils.pytorch3d import (
    euler_angles_to_matrix,
    quaternion_to_matrix,
)

# GLOBAL VARIABLES
torch.cuda.empty_cache()
torch.pi = torch.acos(torch.zeros(1)).item() * 2
trans_frac = 0.0/4                            # Percentage of middle volume to sample plane centre from. (0-1)
max_euler = [(0.0)*torch.pi,              # Maximum range to sample the three Euler angles in radians for plane orientation.
                (0.0)*torch.pi,
                (0.0)*torch.pi]

def standardize_with_model(params, model, device, V, L_in_pix, ct):
    V = V.unsqueeze(0)
    params = params.preprocessing.gtpp
    # INITIALIZE MODEL CONFIG SET
    if params.model.seed_val is not None:
        gen = torch.torch.Generator()
        gen.manual_seed(params.model.seed_val + ct)
    else:
        gen = None
    if params.model.int_rad:
        # Random translation
        factor_a = np.array(params.model.desired_size).astype(np.float32)/4 
        tran = ((torch.rand(1, 3, generator=gen) * (factor_a * trans_frac) + factor_a * (1 - trans_frac) / 2.0) - ((factor_a - 1) / 2.0))/factor_a
        # Random uniform sampling of Euler angles with restricted range.
        euler_angles = utils.sample_euler_angles_fix_range(1, 
                                                            max_euler[0], 
                                                            max_euler[1], 
                                                            max_euler[2], 
                                                            params.model.seed_val + ct)
    else:
        euler_angles = torch.zeros(1, 3)
        tran = torch.zeros(1, 3)
        
    rot =  euler_angles_to_matrix(euler_angles, "XYZ")
    slices_in, _ = utils.get_slices(params, 
                                    V, 
                                    rot, 
                                    tran, 
                                    factor=1)
    ##### Cumulative transformation ##### 
    R = rot.to(device)
    T = tran.to(device)
    with torch.no_grad():
        for it in range(params.model.test_it):
            ytr_es, yrr_es, _, _ = model(slices_in.to(device))
            # Estimate output slices / update slices that will be the next input.
            rot = quaternion_to_matrix(yrr_es)  
            # Rotations accumulated in one go.    
            t_current = torch.bmm(R, ytr_es.unsqueeze(2))
            T = T + t_current[:,:,0]
            R = torch.bmm(R, rot)
            slices_in, _ = utils.get_slices(params, V, R, T, factor=1)
    _, Vhat = utils.get_slices(params, V, R, T)
    ct += 1
    # ------------------ For landmarks -------------------------
    R_l = np.linalg.inv(R[0].cpu().numpy())
    T_l = -1 * T.cpu().numpy()
    img_size = np.array(Vhat[0].shape).astype(np.float32)
    L_in_pix_norm = (L_in_pix - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
    L_in_pix_norm_translated = L_in_pix_norm + T_l
    Lhat_in_pix_norm = np.dot(R_l, L_in_pix_norm_translated.T).T
    Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0
    # ------------------ For landmarks -------------------------
    return Vhat.squeeze(0), Lhat_in_pix, ct

def standardize_with_gt(plane_metadata, name, V, p, L_in_pix, ct, logger):
    img_size = np.array(V.shape).astype(np.float32)
    # Retrieve ground truth plane data from dictionary for current image
    plane = plane_metadata.get(name)
    if plane is None:
        logger.warning(f"Ground truth for {name} not found!")
        return None, None, ct
    # Assumption: Centers are in RAS coordinate system
    center_gt_in_pix = np.array([-1 * plane["center"][0], -1 * plane["center"][1], plane["center"][2]]) / p
    # Compute translation vector for affine transform
    translation_vector = (center_gt_in_pix - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
    # Compute rotation matrix from plane parameters
    rotation_3x3 = affine3Dmatrix(plane)
    # Construct full 3x4 affine transformation matrix
    transform_matrix = np.zeros((3, 4))
    transform_matrix[:3, :3] = rotation_3x3
    transform_matrix[:3, 3] = translation_vector
    # Apply affine transformation to the image volume
    Vhat = grid_transform_3d(V.astype(np.float32), transform_matrix.astype(np.float32))
    V = Vhat.copy()
    # ------------------ For landmarks -------------------------
    L_in_pix_norm = (L_in_pix - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
    inv_translation_vector = -1 * translation_vector
    Lhat_in_pix_norm = affine_transform(L_in_pix_norm, np.linalg.inv(rotation_3x3), inv_translation_vector)
    Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0
    L_in_pix = Lhat_in_pix
    ct += 1
    # ------------------ For landmarks -------------------------
    return V, L_in_pix, ct