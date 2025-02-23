import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from net.dataset.utility.rotation import *
import SimpleITK as sitk


def main(dataframe, params):
    """
    Processes ultrasound data by loading images and applying transformations to produce 
    pre-processed, rotated, and translated ultrasound images along with associated landmark points.

    Args:
        dataframe (pd.DataFrame)
        params : 
    Returns:
        None. Saves processed images and landmarks in specified `out_dir`.
    """

    image_folder = params.root + 'volumes'
    landm_folder = params.root + 'landmarks'
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(landm_folder, exist_ok=True)
    os.makedirs(landm_folder + '/csv', exist_ok=True)
    os.makedirs(landm_folder + '/fcsv', exist_ok=True)

    filenames = list(dataframe['path_to_nrrd'].values)

    # Load ground truth data for standard planes
    dicto = json.loads(get_file_list('datasets/gt.txt')[0])

    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        
        # Prepare file paths and identifiers
        image_path = filename
        name = dataframe.loc[i, 'full_id']
        week, pid = dataframe.loc[i, 'week'], dataframe.loc[i, 'pid']

        if os.path.exists(os.path.join(image_folder, name) + '.nrrd'):
            continue

        print(f"Processing image {i + 1}/{len(filenames)}: {dataframe.loc[i, 'full_id']}")

        # Load ultrasound image and header information
        V, header = extract_image(image_path)
        V = np.array(V)

        # save_3d_image(V, header, V.shape, image_folder, name, prefix='init')
        
        if dataframe.loc[i, 'landmark_antonia_found']:
            lmk_path = dataframe.loc[i, 'path_to_csv_antonia']
            lmk = pd.read_csv(lmk_path)
            print(f"Loaded {len(lmk)} landmarks from {lmk_path}")
        else:
            print(f"Could not found landmarks. skipping...")
            continue
        # Prepare landmark matrix and pixel dimensions
        L_in_mm = get_matrix_of_lmks(lmk)

        # save_transformed_landmarks(L_in_mm, lmk, output_path, name, prefix='init')
        
        # Determine pixel dimensions
        try:
            p = header.get('spacings')[:3]
        except:
            p = np.array([header['space directions'][0, 0], 
                          header['space directions'][1, 1], 
                          header['space directions'][2, 2]])

        img_size = np.array(V.shape)

        # --------- #
        # LP FILTER #
        # --------- #
        V = filter_3d_image(V) # type: ignore
        save_3d_image(V, header, V.shape, image_folder, name, prefix='f')

        # ----------  #
        # INTERPOLATE #
        # ----------  #

        image = sitk.GetImageFromArray(V)
        image.SetSpacing(p)  # Set original spacing

        new_spacing = params.desired_spacings
        original_size = image.GetSize()
        new_size = [int(round(os * ospc / nspc)) for os, ospc, nspc in zip(original_size, p, new_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        resampled_image = resampler.Execute(image)
        V = sitk.GetArrayFromImage(resampled_image)

        header['spacings'] = params.desired_spacings
        save_3d_image(V, header, np.array(V.shape), image_folder, name, prefix=F'inp_f{filter}')
        save_transformed_landmarks(L_in_mm, lmk, landm_folder, name, prefix='mm')
        p = params.desired_spacings
        
        img_size = np.array(V.shape)

        # ----------  #
        # DOWNSAMPLE #
        # ----------  #
        
        if params.drate:
            downsample_rates = np.array([
                                        params.drate, 
                                        params.drate, 
                                        params.drate]
                                        )
        else:
            downsample_rates = np.ceil(img_size / params.desired_size).astype(int)

        p *= downsample_rates
        V = V[::downsample_rates[0], ::downsample_rates[1], ::downsample_rates[2]]
        header['spacings'] = p
        # save_3d_image(V, header, np.array(V.shape), image_folder, name, prefix=F'ds_f{filter}')

        # Retrieve plane data from dictionary
        plane = dicto.get(name)
        if plane is None:
            print(f"Ground truth for {name} not found!")
            continue

        L_in_pix = L_in_mm / p
        
        V = np.transpose(V, (2, 1, 0))
        L_in_pix = swap_xz_coordinates(L_in_pix)
        
        # Assumption 3: Centers are in RAS.
        center_gt_in_pix = np.array([-1*plane["center"][0], -1*plane["center"][1], plane["center"][2]]) / p

        # ------- #
        # PADDING #
        # ------- #
        
        for dim in V.shape:
            if dim > params.desired_size[0]:                 # assuming iso
                # TODO: what should I do?
                print(f"dim exceeded: ", V.shape)
                pass
        V, L_in_pix, center_gt_in_pix = pad_3d_image(V, L_in_pix, center_gt_in_pix, params.desired_size)
        L_in_pix = L_in_pix.astype(np.float32)
        center_gt_in_pix = center_gt_in_pix.astype(np.float32)
        # save_3d_image(V, header, params.desired_size, image_folder, name, prefix='pd')
        # save_transformed_landmarks(L_in_pix * p, lmk, landm_folder, name, prefix='pdmm')

        # ------ -------------- #
        # AFFINE TRANSFORMATION #
        # ------ -------------- #

        # Update image size
        img_size = np.array(V.shape).astype(np.float32)
        # Find translation vector.
        translation_vector = (center_gt_in_pix - img_size/ 2.0) * ( 2.0 / img_size).astype(np.float32)
        rotation_3x3 = affine3Dmatrix(plane)

        transform_matrix = np.zeros((3,4))
        transform_matrix[:3, :3] = rotation_3x3
        transform_matrix[:3, 3] = translation_vector

        # Apply affine transformations
        Vhat = grid_transform_3d(V.astype(np.float32), transform_matrix.astype(np.float32))
        Vhat = np.transpose(Vhat, (2, 1, 0))

        save_3d_image(Vhat, header, img_size, image_folder, name, prefix=f'')

        L_in_pix = swap_xz_coordinates(L_in_pix)
        L_in_pix_norm = (L_in_pix - img_size/ 2.0) * ( 2.0 / img_size).astype(np.float32)
        inv_translation_vector = -1 * translation_vector
        Lhat_in_pix_norm = affine_transform(L_in_pix_norm, np.linalg.inv(rotation_3x3), inv_translation_vector)
        Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0

        save_transformed_landmarks(Lhat_in_pix * p, lmk, landm_folder, name, prefix='')

