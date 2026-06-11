# Author: Antonia Alomar (12 Jan 2024)
#
# Standard scan plane detection in 3D ultrasound images of fetal head
# Network training --> Adaptation to torch spatial transform
#
# In this script, we use quaternions to represent rotation.
#
# Reference
# Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network
# https://arxiv.org/abs/1806.07486
#
# ==============================================================================


import torch 
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import nrrd
import gc

from net.dataset.utility.gtpp.dataloaders.torch_dataloader import Create_Ultrasound_loader, US_3D_dataset_fast
from net.dataset.utility.gtpp.utils.SonoNet import SonoNet
from net.dataset.utility.gtpp.utils import slicer_to_torch_transform
from net.dataset.utility.gtpp.utils.pytorch3d import euler_angles_to_matrix, quaternion_to_matrix
import json
from net.dataset.utility.rotation import affine_transform, swap_xz_coordinates


torch.cuda.empty_cache()
torch.pi = torch.acos(torch.zeros(1)).item() * 2
trans_frac = 0.0/4                            # Percentage of middle volume to sample plane centre from. (0-1)
max_euler = [(0.0)*torch.pi,              # Maximum range to sample the three Euler angles in radians for plane orientation.
                (0.0)*torch.pi,
                (0.0)*torch.pi]

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
    image_i = slicer_to_torch_transform.affine_transform(image, mat_rot)
    
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
  
def get_coords(config, coord, rot, trans, factor=1):
    coord, rot, trans = coord[0].cpu().detach().numpy(), rot[0].cpu().detach().numpy(), trans.cpu().detach().numpy()
    coord = swap_xz_coordinates(coord)
    img_size = np.array(config.params.desired_size)/factor
    L_in_pix_norm = (coord - img_size / 2.0) * (2.0 / img_size).astype(np.float32)
    print(rot, trans)
    inv_trans = -1 * trans
    Lhat_in_pix_norm = affine_transform(L_in_pix_norm, np.linalg.inv(rot), inv_trans)
    Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0
    return Lhat_in_pix

def gtpp(dataframe, config):
    # DATALOADING AND GROUND TRUTH PLANE DEFINITION    
    prefix= 'test'
    Create_Ultrasound_loader(dataframe,
                             config.file_paths.list_files.test, 
                             config.file_paths.data_dir, 
                             config.file_paths.label_dir, 
                             config.file_paths.out_dir_pre_pross, 
                             config.params.desired_size, prefix, 
                             nSamples=config.params.num_samples, 
                             transform=True, 
                             downsample= config.params.downsampling_factor)
    testset = US_3D_dataset_fast(prefix, 
                                 config.file_paths.out_dir_pre_pross, 
                                 scan_transform=None, 
                                 nSamples=config.params.num_samples,)                                  
                                   
    testLoader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

    print("TEST SET:\n")
    print("\tTotal num of batches : {}".format(len(testLoader)))
    print("\tTotal num of examples in data set : {}".format(len(testLoader.dataset)))
    # Define model
    model = SonoNet(config.params, 
                    config.params.input_plane, 
                    config.params.num_output_tr,
                    config.params.num_output_rr,
                    dropout_p = 0.25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model.to(device)
    #print((config.config.batch_size ,config.input_plane, config.box_size[0], config.box_size[1]))
    #summary(model, (config.batch_size ,config.input_plane, config.desired_size[0], config.desired_size[1]))
    names_planes = ["sagittal","coronal","axial"]
    print(prefix)
    name_out_dir = prefix + "_it_"+ str(config.params.test_it) + '_out/' 
    if config.params.int_rad:
        name_out_dir = 'random_ini_' + name_out_dir
    if not os.path.isdir(config.file_paths.ddir + name_out_dir):
        os.makedirs(config.file_paths.ddir + name_out_dir)
        os.makedirs(config.file_paths.ddir + name_out_dir+ names_planes[0]+'/')
        os.makedirs(config.file_paths.ddir + name_out_dir+ names_planes[1]+'/')
        os.makedirs(config.file_paths.ddir + name_out_dir + names_planes[2]+'/')
        print('dir created')    
    if config.params.resume:
        # Resume previous training
        checkpoint = torch.load(os.path.join(config.model_dir, 
                                             config.checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_number = checkpoint['epoch']
        # loss = checkpoint['loss']
    
    print(' LOADED EPOCH {}:'.format(epoch_number))

    count = 0 
    print(len(testLoader))
  
    filenames_list = []
    model.eval()
    
    allplanes={}
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
          for filenames,images_res, coords, pix_dim, slices_gt, trans_gt, rots_gt, mat_gt in testLoader:
                name = str(filenames)
                filenames_new = name.replace("'", "")
                filenames_new = filenames_new.replace("(", "")
                filenames_new = filenames_new.replace(")", "")
                filenames_new = filenames_new.replace(",", "")   
                if "full_resolution" in prefix:
                    factor_d=2
                    images = images_res[0:images_res.shape[0], 0:images_res.shape[1]:factor_d, 0:images_res.shape[2]:factor_d, 0:images_res.shape[3]:factor_d] 

                else:
                    images = images_res
                    factor_d = 1
                
                filenames_list.append(filenames)
                img_val_saggital = []
                img_val_coronal = []
                img_val_axial = [] 
                # Initialize plane from which we start the search
                batch_size = images.shape[0]
                if config.params.seed_val is not None:
                    gen = torch.torch.Generator()
                    gen.manual_seed(config.params.seed_val + count)
                else:
                    gen= None
                
                if config.params.int_rad:
                    # Random translation
                    factor_a = np.array(config.params.desired_size).astype(np.float32)/4 
                    tran = ((torch.rand(batch_size, 3,generator=gen) * (factor_a * trans_frac) + factor_a * (1-trans_frac) / 2.0) - ((factor_a-1) / 2.0))/factor_a
                    
                    # Random uniform sampling of Euler angles with restricted range
                    euler_angles = sample_euler_angles_fix_range(batch_size, max_euler[0], max_euler[1], max_euler[2], config.params.seed_val + count)

                else:
                    euler_angles =torch.zeros(batch_size, 3)
                    tran =torch.zeros(batch_size, 3)
                    
                rot =  euler_angles_to_matrix(euler_angles,"XYZ")
                # quaternions = matrix_to_quaternion(rot)
                slices_in, _ = get_slices(config, images,rot,tran, factor=factor_d)
                
                rot_in = rot
                tran_in = tran
                
                ##### Cumulative transformation #####
                               
                R = rot.to(device)
                T = tran.to(device)
                for _ in range(0, config.params.test_it):

                    # Make predictions for this batch
                    ytr_es, yrr_es, _, _ = model(slices_in.to(device))
                                       
                    # estimate output slices/ update slices that will be the next input
                    rot = quaternion_to_matrix(yrr_es)  
                    #  Rotations accumulated in one go    
                    t_current = torch.bmm(R, ytr_es.unsqueeze(2))
                    T = T + t_current[:,:,0]
                    R = torch.bmm(R,rot)
                    slices_in,_ = get_slices(config,images,R,T,factor=factor_d)
                    print(images.shape)
                    lmk_rot = get_coords(config, coords.to(device), R, T, factor=factor_d)
                    lmk_rot_in_mm = lmk_rot * pix_dim.cpu().detach().numpy()
                if config.params.save_jpg:   
                    for k in range(config.params.input_plane):
                        # Extract the grayscale tensor (assuming slices_in is already a tensor)
                        image_tensor = slices_in[0, k, :, :]
                        # Ensure the tensor is 2D and add a fake channel dimension for compatibility
                        image_tensor = image_tensor.unsqueeze(0)  # Shape becomes (1, H, W)   
                        # Normalize values to the range [0, 1]
                        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
                        # Rotate the image by 180 degrees
                        rotated_image = TF.rotate(image_tensor, angle=180)
                        save_name =  config.model_dir + name_out_dir +names_planes[k]+'/'+ filenames_new + '_' + names_planes[k]+'.jpg'
                        # Save the grayscale image
                        vutils.save_image(rotated_image, save_name)
                                                         
                slices_in,image_final = get_slices(config,images_res,R,T)
                if config.params.save_nrrd:
                    if config.params.int_rad:
                        filenames_new = filenames_new + '_rand_init_'
                        _ ,images = get_slices(config, images,rot_in,tran_in, factor=factor_d)

                    save_here = config.file_paths.ddir + name_out_dir + 'S' + filenames_new + '.nrrd'  
                    if not os.path.exists(save_here): 
                        nrrd.write(save_here,image_final.detach().numpy(), index_order='C')
                        nrrd.write(config.file_paths.ddir + name_out_dir +'original_' + filenames_new + '.nrrd',images.detach().numpy(), index_order='C')
                
                save_lmk_here = config.file_paths.ddir + name_out_dir + 'S' + filenames_new + '.fcsv'
                save_csv_here = config.file_paths.ddir + name_out_dir + 'S' + filenames_new + '.csv'

                slicer_to_torch_transform.save_transformed_landmarks_gtpp(lmk_rot_in_mm, save_csv_here, save_lmk_here)
                #Dictionary
                info={'name':filenames_new,'R':R.cpu().numpy().tolist(),'t':T.cpu().numpy().tolist()}
                allplanes.update({filenames_new:info})
                count = count + 1
                del img_val_saggital, img_val_axial, img_val_coronal
                gc.collect()
                                  
    with open(''.join((config.file_paths.ddir + name_out_dir + config.dic_name +"_allplanes_inference.txt")), "w") as fp:
        json.dump(allplanes,fp) 
