import numpy as np
import os
import torch
from utils import plane, slicer_to_torch_transform
import nrrd
import json
from ast import literal_eval
from os.path import join, isfile
import scipy.io as sio
import numpy as np
import glob
import matplotlib.pyplot as plt
import copy
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from monai.transforms import (Compose, RandGaussianNoise, RandShiftIntensity, RandAdjustContrast, RandGaussianSmooth, RandGaussianSharpen, RandCoarseDropout, RandCoarseShuffle,)
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

def get_file_list(txt_file):
    """Get a list of filenames.

    Args:
      txt_file: Name of a txt file containing a list of filenames for the images.

    Returns:
      filenames: A list of filenames for the images.

    """
    with open(txt_file) as f:
        filenames = f.read().splitlines()
    return filenames

def extract_image(filename):
    """Extract the image into a 3D numpy array [x, y, z].

    Args:
      filename: Path and name of nifti file.

    Returns:
      data: A 3D numpy array [x, y, z]
      pix_dim: pixel spacings

    """
    data,header=nrrd.read(filename.strip())
    
    if len(data.shape)==4:
        data=data[:,:,:,0]
    return data, header

def Create_Ultrasound_loader(file_list, data_dir, label_dir, out_dir, size_desired, prefix, nSamples=None, transform=None, scan_transform= None, downsample= None):

    """Load the input images and the standard planes.

    Args:
      file_list: txt file containing list of filenames of images
      data_dir: Directory storing images.
      label_dir: Directory storing standard planes.
      out_dir: Directory to save pre-processed data.
      nSamples: Number of samples in the set.
      transform: Transformation that is applied to the original data.
      size_desired: ultrasound fixed size to input the network.
      
    Returns:
      images: list of img_count 4D numpy arrays with dimensions=[width, height, depth]. Eg. [324, 207, 279]
      trans_vecs: 3D centre point of the ground truth plane. [img_count, 3]
      quats: Quaternions that rotate xy-plane to the GT plane. [img_count, 4]
      gt_transform: transformation to obtain standard planes  [3, 4]"""
    
    filenames = get_file_list(file_list)
    if nSamples is not None:
        filenames= filenames[0:nSamples]    

    if not os.path.exists(join(out_dir,prefix)):
        os.makedirs(join(out_dir,prefix))
    
    file_count = len(filenames)
    image= []

    headers = []

    gt=json.loads(get_file_list(label_dir)[0])
    
    for i in range(len(filenames)):

        # Read 3D Ultrasounds
        
        filename =''.join((data_dir,filenames[i].strip()))
        print("Loading image {}/{}: {}".format(i+1, len(filenames), filename))
        name=filename.split('/')[-1][:-5]
        
        if len(name)<9:
            name_gt = '0'+ name
        else:
            name_gt = name

        # load image
        img, header= extract_image(filename)

        if 'spacings' in header:
            p=header['spacings']

        elif 'space directions' in header:
            p=np.array([header['space directions'][0,0],header['space directions'][1,1],header['space directions'][2,2]])
        
        pix_dim = p[0:3]
        
        if downsample is not None:
            pix_dim = pix_dim*downsample
            img = img[0:img.shape[0]:downsample, 0:img.shape[1]:downsample, 0:img.shape[2]:downsample] 
        
        img_siz = np.array(img.shape[0:3])
        try:
            gt_item=gt[name]
        except:
            print('Not in dictionay')
            continue
        # Compute translation and rotation of GT plane wrt reference coordinate system (origin at centre of volume)

        
        trans_gt, sl = slicer_to_torch_transform.translation(gt_item,pix_dim,np.array(size_desired),np.array(img_siz))
        quats, mat = slicer_to_torch_transform.affine3Dmatrix(gt_item, point= trans_gt) # No translation in the totation matrix as first is translated and the rotated 
        
        # Image transform ( permute as the torch grid transforms considering D H W  and we have an image with H W D)
        if transform is not None:
            image=  torch.from_numpy(img.astype(np.float32))
            # Padding image
            diff_size = np.array(size_desired) - img_siz
            diff_size = diff_size/2
            
            
            a= int(diff_size[2]) 
            b= int(diff_size[1]) 
            c= int(diff_size[0]) 
            
            if (diff_size[2]*2)%2 == 0:
                x = 0
            else:
                x=1
                
            if (diff_size[1]*2)%2 == 0:
                y = 0
            else:
                y=1
                
            if (diff_size[0]*2)%2 == 0:
                z = 0
            else:
                z=1                
            
            m= torch.nn.ConstantPad3d([a+x,a,b+y,b,c+z,c], 0)
            image = m(image)
            image=  torch.permute(image, (2, 1, 0))
                    
        # Center image is the standard plane 
        c_idx = (np.array(size_desired)/2)+1        
        
        # Extract image standard planes

        # Rotate image 
        mat_gt = torch.from_numpy(mat)
        slices_t = slicer_to_torch_transform.affine_transform(image.unsqueeze(0), mat_gt.unsqueeze(0)) 
              
        slices_gt = torch.zeros(3,size_desired[0],size_desired[1])
        
        # Ground truth slices
        slices_gt[0,:,:] = slices_t[0,0,:,:,int(c_idx[2])].unsqueeze(0)
        slices_gt[1,:,:] = slices_t[0,0,:,int(c_idx[1]),:].unsqueeze(0)
        slices_gt[2,:,:] = slices_t[0,0,int(c_idx[0]),:,:].unsqueeze(0)  
               
        names_planes = ["sagittal","coronal","axial"]
        for i in range(3):
              if not os.path.exists(join(out_dir,prefix,names_planes[i])):
                  os.makedirs(join(out_dir,prefix,names_planes[i]))
              image_tensor = slices_gt[i,:, :]
              # Ensure the tensor is 2D and add a fake channel dimension for compatibility
              image_tensor = image_tensor.unsqueeze(0)  # Shape becomes (1, H, W)   
              # Normalize values to the range [0, 1]
              image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
              # Rotate the image by 180 degrees
              rotated_image = TF.rotate(image_tensor, angle=180)   
              save_name =  join(out_dir,prefix,names_planes[i], name_gt+'_'+names_planes[i]+'.jpg')
              # Save the grayscale image
              vutils.save_image(rotated_image, save_name)
              
              '''plt.imshow(slices_gt[i,:, :],cmap='gray')
              plt.show()
              plt.savefig(join(out_dir,prefix,names_planes[i],'gt_plane_R&T_'+name_gt+names_planes[i]+'.jpg'))'''          
           


        

