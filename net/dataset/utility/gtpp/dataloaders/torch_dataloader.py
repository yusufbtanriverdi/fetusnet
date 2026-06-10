"""Functions for reading input data (image (nrrd), and standard planes)."""

import numpy as np
import os
import torch
from net.dataset.utility.gtpp.utils import slicer_to_torch_transform
from net.dataset.utility.rotation import get_matrix_of_lmks, affine_transform
import nrrd
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from os.path import join
import pandas as pd

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

def Create_Ultrasound_loader(dataframe, file_list, data_dir, label_dir, out_dir, size_desired, prefix, nSamples=None, transform=None, scan_transform= None, downsample= None):

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
    
    image= []
    gt=json.loads(get_file_list(label_dir)[0])
    
    for i in range(len(filenames)):
        # Read 3D Ultrasounds
        filename =''.join((data_dir,filenames[i].strip()))
        osid = os.path.splitext(os.path.basename(filename))[0]
        coord_file = join(data_dir, dataframe[dataframe['osid']==osid]['_fcsv'].values[0])
        print("Loading image {}/{}: {}".format(i+1, len(filenames), filename))
        name=filename.split('/')[-1][:-5]
        
        if len(name)<9:
            name_gt = '0'+ name
        else:
            name_gt = name
        
                # Attempt to load landmarks CSV file
        ct_not_found = 0
        try:
            lmk = pd.read_csv(coord_file)
        except Exception as e:
            print(str(e))
            ct_not_found += 1
            print(f"Could not load landmarks from {coord_file} for {osid}! Skipping!! Count: {ct_not_found}")
            dataframe.loc[i, 'csvfound'] = False
            continue

        # Convert landmark points to matrix format in millimeters
        L_in_mm = get_matrix_of_lmks(lmk)
        if os.path.isfile(join(out_dir,prefix, ('US_Data_'+ name_gt + '.pt'))):
            continue

        # load image
        img, header= extract_image(filename)

        if 'spacings' in header:
            p=header['spacings']

        elif 'space directions' in header:
            p=np.array([header['space directions'][0,0],header['space directions'][1,1],header['space directions'][2,2]])
        
        pix_dim = p[0:3]
        coord = L_in_mm / pix_dim
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


        '''
        # Extract image standard planes
        mat_t_gt = torch.zeros(3, 4)
        mat_t_gt[:,:3]= torch.eye(3)
        
        mat_t_gt[:,3] =torch.from_numpy(trans_gt.astype(np.float32))
         
    
        # Translate image 
        slices_t = slicer_to_torch_transform.affine_transform(image.unsqueeze(0), mat_t_gt.unsqueeze(0))
        
        # Center image is the standard plane 
        c_idx = (np.array(size_desired)/2)+1
        
        
        plt.imshow(slices_t[0,0,:,:,int(c_idx[2])],cmap='gray')
        plt.show()
        plt.savefig('gt_plane_translate'+'.jpg')
  
        # Rotate image 
        mat_gt = torch.from_numpy(mat)
        slices_t = slicer_to_torch_transform.affine_transform(slices_t[:,0,:,:,:], mat_gt.unsqueeze(0)) 
        

        
        slices_gt = torch.zeros(3,size_desired[0],size_desired[1])
        
        # Ground truth slices
        slices_gt[0,:,:] = slices_t[0,0,:,:,int(c_idx[2])].unsqueeze(0)
        slices_gt[1,:,:] = slices_t[0,0,:,int(c_idx[1]),:].unsqueeze(0)
        slices_gt[2,:,:] = slices_t[0,0,int(c_idx[0]),:,:].unsqueeze(0)  
               
        names_planes = ["saggital","axial","coronal"]
        for i in range(3):
              if not os.path.exists(join(out_dir,prefix,names_planes[i])):
                  os.makedirs(join(out_dir,prefix,names_planes[i]))
                  
              plt.imshow(slices_gt[i,:, :],cmap='gray')
              plt.show()
              plt.savefig(join(out_dir,prefix,names_planes[i],'gt_plane_'+name_gt+names_planes[i]+'.jpg'))'''
                    
        # Center image is the standard plane 
        c_idx = (np.array(size_desired)/2)+1        
        
        # Extract image standard planes

        # Rotate image 
        mat_gt = torch.from_numpy(mat)
        slices_t = slicer_to_torch_transform.affine_transform(image.unsqueeze(0), mat_gt.unsqueeze(0)) 
      
        slices_gt = torch.zeros(3,size_desired[0],size_desired[1])

        try:         
            # Ground truth slices
            slices_gt[0,:,:] = slices_t[0,0,:,:,int(c_idx[2])].unsqueeze(0)
            slices_gt[1,:,:] = slices_t[0,0,:,int(c_idx[1]),:].unsqueeze(0)
            slices_gt[2,:,:] = slices_t[0,0,int(c_idx[0]),:,:].unsqueeze(0)  
        except Exception as e:
            print(str(e))
            print(image.shape, name_gt)
            continue

        names_planes = ["saggital","axial","coronal"]
        for i in range(3):
              if not os.path.exists(join(out_dir,prefix,names_planes[i])):
                  os.makedirs(join(out_dir,prefix,names_planes[i]))
                  
              plt.imshow(slices_gt[i,:, :],cmap='gray')
              # plt.show()
              plt.savefig(join(out_dir,prefix,names_planes[i],'gt_plane_R&T_'+name_gt+names_planes[i]+'.jpg'))          
              plt.close()
        data = (name_gt, image, coord, pix_dim, slices_gt, torch.from_numpy(np.array(trans_gt).astype(np.float32)), torch.from_numpy(np.array(quats.squeeze(0)).astype(np.float32)), mat_gt )
        
        '''mdic = {'filename': name_gt, 'image': image.numpy(), 'slices_gt': slices_gt.numpy(), 'trans': trans_gt, 'quat': quats, 'rot': mat_gt.numpy()}
        sio.savemat('Dataloader_debug.mat',mdic)'''
        torch.save(data, join(out_dir,prefix, ('US_Data_'+ name_gt + '.pt')))
        

class US_3D_dataset_fast(Dataset):
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
        
        filename, image, coord, pix_dim, slices_gt, trans_vecs, quats, mat_gt = torch.load(join(self.out_dir, self.data_files[idx]))
        print(filename)
        # DATA AUGMENTATION TRANSFORMATION 
        if self.scan_transform:
            print('Appling trasnform \n')
            image = self.scan_transform(image)
            slices_t = slicer_to_torch_transform.affine_transform(image.unsqueeze(0), mat_gt.unsqueeze(0)) 
              
            slices_gt = torch.zeros(3,image.shape[0],image.shape[1])
            c_idx = (np.array(image.shape)/2)+1
            
            # Ground truth slices
            slices_gt[0,:,:] = slices_t[0,0,:,:,int(c_idx[2])].unsqueeze(0)
            slices_gt[1,:,:] = slices_t[0,0,:,int(c_idx[1]),:].unsqueeze(0)
            slices_gt[2,:,:] = slices_t[0,0,int(c_idx[0]),:,:].unsqueeze(0)  
            
            '''names_planes = ["saggital","axial","coronal"]
            for i in range(3):
   
                  plt.imshow(slices_gt[i,:, :],cmap='gray')
                  plt.show()
                  plt.savefig(join('gt_plane_R&T_'+filename+names_planes[i]+'.jpg'))'''    
            
        return filename, image, coord, pix_dim, slices_gt, trans_vecs, quats, mat_gt