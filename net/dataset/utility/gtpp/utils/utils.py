import torch
import torch.nn.functional as F
import numpy as np
import nrrd
import pandas as pd
from net.dataset.utility.gtpp.utils.pytorch3d import matrix_to_quaternion
import gc

LMK_LIST=["exR", "enR", "n", "enL", "exL", "acR", "aR", "prn", "aL", "acL", "sn",
          "chR", "cphR", "ls", "cphL", "chL", "li", "sl", "pg"]

def extract_image(filename):
    """Extract the image into a 3D numpy array [x, y, z]. As it was saved in RAS

    Args:
      filename: Path and name of nifti file.

    Returns:
      data: A 3D numpy array [x, y, z]
      pix_dim: pixel spacings

    """
    data,header=nrrd.read(filename)

    if len(data.shape)==4:
        data=data[:,:,:,0]
    return data, header
    
def affine_grid_generator_3D(theta, size):
    N, C, D, H, W = size

    base_grid = torch.zeros((N, D, H, W, 4), dtype=torch.float32)

    w_points = (torch.linspace(-1, 1, W) if D > 1 else torch.Tensor([-1]))
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if W > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points
    base_grid[:, :, :, :, 3] = 1

    grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
    grid = grid.view(N, D, H, W, 3)

    return grid
    
def affine_transform(moving_image, theta):
    
    N, D, H, W = moving_image.shape

    # Adding channel dimension
    moving_image = moving_image.unsqueeze(1)

#    # Extending theta to include batches
#    predicted_theta = torch.empty(N, theta.shape[1], theta.shape[2])
#    predicted_theta[:] = theta

    affine_grid = affine_grid_generator_3D(theta, (N, 1, D, H, W))
    predicted_image = F.grid_sample(moving_image, affine_grid)

    return predicted_image

def translation(gt,p,size_desired,size_im):
    center_planes = abs(np.array([gt["center"][0],gt["center"][1],gt["center"][2]]))
    c_planes_pixels = center_planes/(p)
    # Center normalized -1 to 1
    c_norm = ((c_planes_pixels.astype(np.float32) - (size_im.astype(np.float32)/2.0))/(size_desired.astype(np.float32)/2))
    return c_norm, c_planes_pixels

def affine3Dmatrix(gt, point=[0,0,0]):
    normalsagittal=gt['sagittal']
    normalsagittal=np.array([normalsagittal[0],normalsagittal[1],-normalsagittal[2]])

    normalcoronal=gt['coronal']
    normalcoronal=np.array([normalcoronal[0],normalcoronal[1],-normalcoronal[2]])

    normalaxial=gt['axial']
    normalaxial=np.array([normalaxial[0],normalaxial[1],-normalaxial[2]])

    V0=[[1,0,0],[0,1,0],[0,0,-1]]
    V1=[normalsagittal,normalaxial,normalcoronal]
    M=np.transpose(np.linalg.solve(V0,V1))

    mat = np.zeros((3,4), np.float32)
    mat[:3, :3] = M
    mat[:3, 3] = point
    
    # Quaternions vector
    quat = matrix_to_quaternion(torch.from_numpy(M).unsqueeze(0))
    
    return quat.numpy(), mat

    
def save_transformed_landmarks(Lhat, output_csv_path, output_fscv_path):
    """
    Save transformed landmarks to CSV and FCSV files with predefined header lines.

    Parameters:
    - Lhat (np.ndarray): Array containing transformed landmark coordinates.
    - lmk (pd.DataFrame): DataFrame to copy structure from.
    - output_path (str): Directory path to save the files.
    - name (str): Base name for the output files.
    """
    # Copy structure of landmarks and update with transformed coordinates
    Lhat_df = pd.DataFrame(columns=["","x","y","z","ow","ox","oy","oz","vis","sel","lock","label","desc","associatedNodeID"])
    for idx in range(19):
        Lhat_df.loc[idx, ['x', 'y', 'z']] = Lhat[idx, :]
        Lhat_df.loc[idx, ['vis', 'lock', 'label']] = 1, 1, LMK_LIST[idx]
    # Save CSV file
    Lhat_df.to_csv(output_csv_path, index=False)

    # Define the predefined lines for FCSV format
    predefined_lines = [
        "# Markups fiducial file version = 5.2\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    ]

    # Save FCSV file with predefined lines
    fcsv_path = output_fscv_path
    with open(fcsv_path, 'w') as f:
        # Write predefined lines
        f.writelines(predefined_lines)
        
        # Write each row as a comma-separated string
        for index, row in Lhat_df.iterrows():
            f.write(','.join(map(str, row.values)) + '\n')


def sample_euler_angles_fix_range(num, max_angle1=torch.pi, max_angle2=torch.pi/2.0, max_angle3=torch.pi,seed=None):
    """Uniform random sampling of Euler angles with restricted range. Sample angles between [-max_angle1, max_angle1]

    Args:
    num: number of random samples
    max_angle1, max_angle2, max_angle3: maximum positive angle to sample from. Possible values are [0, pi], [0, pi/2] and [0, pi]
    Using max_angle1=pi, max_angle2=pi/2, max_angle3=pi cover the whole rotation sphere once.

    Returns:
    angles: Euler angles (Roll-Pitch-Yaw) [num, 3]

    """
    
    if seed is not None:
        gen = torch.torch.Generator()
        gen.manual_seed(seed)
    else:
        gen= None
    angles = torch.zeros(num,3)   
    angle1 = 2 * max_angle1 * torch.rand(num, generator=gen) - max_angle1
    a = torch.cos(torch.tensor(torch.pi/2.0 - max_angle2))
    angle2 = torch.arccos((1-2*torch.rand(num, generator=gen)) * a) - torch.pi/2.0
    angle3 = 2 * max_angle3 * torch.rand(num, generator=gen) - max_angle3
    angles[:,0] = angle1
    angles[:,1] = angle2
    angles[:,2] = angle3
    
    return angles

def get_slices(config, image, rot, trans, factor=1):
    desi_s = np.array(config.params.desired_size)/factor
    batch_size = rot.shape[0]
    
    mat_rot = torch.zeros(batch_size, 3, 4)
    mat_rot[:,:, :3] = rot
    mat_rot[:,:,3] = trans
    # Translate image 
    image_i = affine_transform(image, mat_rot)
    
    # Center image is the standard plane 
    c_idx = (np.array(desi_s)/2)+1
    
    slices = torch.zeros(batch_size,3,int(desi_s[0]),int(desi_s[1]))
    
    # Input slices to the network
    slices[:,0,:,:] = image_i[:,0,:,:,int(c_idx[2])]
    slices[:,1,:,:] = image_i[:,0,:,int(c_idx[1]),:]
    slices[:,2,:,:] = image_i[:,0,int(c_idx[0]),:,:]
    
    del image, mat_rot
    gc.collect() 
    
    return slices, image_i[:,0,:,:,:] 