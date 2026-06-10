"""Functions for reading input data (image (nrrd), and standard planes)."""

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

def Create_Ultrasound_loader_nrrd(folder_data, data_dir, out_dir, size_desired, prefix, nSamples=None, transform=None, scan_transform= None, downsample= None):

    """Load the input images and the standard planes.

    Args:
      Folders: Folders of filenames of images
      data_dir: Directory storing images.
      out_dir: Directory to save pre-processed data.
      nSamples: Number of samples in the set.
      transform: Transformation that is applied to the original data.
      size_desired: ultrasound fixed size to input the network.
      
    Returns:
      name_case : Name case
      images: list of img_count 4D numpy arrays with dimensions=[width, height, depth]. Eg. [324, 207, 279]
      header: header nrrd file"""
      
      
    print(data_dir)
    folders_list = os.listdir(folder_data)
    if nSamples is not None:
        folders_list= folders_list[0:nSamples]    

    if not os.path.exists(join(out_dir,prefix)):
        os.makedirs(join(out_dir,prefix))
    
    file_count = len(folders_list)
    image= []

    headers = []
    
    for folder in folders_list:
        list_files=  os.listdir(os.path.join(folder_data,folder))
        for name in list_files:
            if name.endswith('.nrrd'):
                # Read 3D Ultrasounds
                
                filename =os.path.join(folder_data,folder,name.strip())
                name= name[0:len(name)-5]
                
                name_gt = name.rjust(50) 

                print(name) 
                if os.path.isfile(join(out_dir,prefix, ('US_Data_'+ name+ '.pt'))):
                    continue
        
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
        
                   
                data = (name_gt, image, header)
                
                '''mdic = {'filename': name_gt, 'image': image.numpy(), 'slices_gt': slices_gt.numpy(), 'trans': trans_gt, 'quat': quats, 'rot': mat_gt.numpy()}
                sio.savemat('Dataloader_debug.mat',mdic)'''
                torch.save(data, join(out_dir,prefix, ('US_Data_'+ name+ '.pt')))
                

class US_3D_dataset_fast_nrrd(Dataset):
    def __init__(self, prefix, out_dir, scan_transform, nSamples = None):
        self.out_dir = join(out_dir,prefix)
        self.data_files = [x for x in os.listdir(self.out_dir) if x.endswith('.pt')]
        #print(self.data_files)
        if nSamples is not None:
            self.data_files = self.data_files[:nSamples]
           
        self.scan_transform = scan_transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx): 
        
        name_gt, image, header = torch.load(join(self.out_dir, self.data_files[idx]))
        print(name_gt)
        # DATA AUGMENTATION TRANSFORMATION 
        if self.scan_transform:
            print('Appling trasnform \n')
            image = self.scan_transform(image)
        return name_gt, image, header