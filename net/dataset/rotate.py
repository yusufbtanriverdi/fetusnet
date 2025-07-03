import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from net.dataset.utility.rotation import *
except ModuleNotFoundError:
    from utility.rotation import *
    
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
    # params.root --> where to save
    # params.raw_dir --> access raw files
    # image_folder = params.root + 'volumes'
    # landm_folder = params.root + 'landmarks'

    # os.makedirs(image_folder, exist_ok=True)
    # os.makedirs(landm_folder, exist_ok=True)
    # os.makedirs(landm_folder + '/csv', exist_ok=True)
    # os.makedirs(landm_folder + '/fcsv', exist_ok=True)

    filenames = list(dataframe['path_to_nrrd'].values)

    # Load ground truth data for standard planes
    dicto = json.loads(get_file_list('doc/info//gt.txt')[0])

    ct =0
    ct_not_found = 0
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        
        # Prepare file paths and identifiers
        image_path = os.path.join(dataframe.loc[i, 'path_to_nrrd'])
        name = dataframe.loc[i, 'full_id']
        week, pid = dataframe.loc[i, 'week'], dataframe.loc[i, 'pid']
        save_im_path = os.path.join(params.sys + params.root, dataframe.loc[i, 'processed__vol_path'])
        save_csv_path = os.path.join(params.sys + params.root, dataframe.loc[i, 'processed__csv_path'])
        save_lmk_path = os.path.join(params.sys + params.root, dataframe.loc[i, 'processed__lmk_path'])
        
        if os.path.exists(save_im_path):
            print('Image seems to be processed already!!!')
            print(filename)
            continue
        # print(f"Processing image {i + 1}/{len(filenames)}: {dataframe.loc[i, 'full_id']}")

        # Load ultrasound image and header information
        V, header = extract_image(image_path)
        V = np.array(V)

        # if dataframe.loc[i, 'landmark_antonia_found']:
        try: 
            lmk_path = os.path.join(params.root, dataframe.loc[i, 'path_to_csv']) 
            lmk = pd.read_csv(lmk_path)
            # print(f"Loaded {len(lmk)} landmarks from {lmk_path}")
        except Exception as e:
            print(str(e))
            print(f"Could not load landmarks from {lmk_path}! Skipping!! I used to work on the other version...")
            continue
            # if 'modified' in lmk_path:
            #     # Try to load the modified version
            #     lmk_path = lmk_path.replace('_modified', '')
            #     print(f"Trying to load landmarks from {lmk_path}...")
            #     try:
            #         lmk = pd.read_csv(os.path.join(params.root, lmk_path))
            #         print(f"Loaded {len(lmk)} landmarks from {lmk_path}")
            #     except:
            #         print(f"Could not load landmarks from {lmk_path}...")
            #         continue
            # else:
            #     continue
        # else:
        #     print(f"Could not found landmarks. skipping...")
        #     continue

        if len(lmk) != 19: print('Something wrong!!', filename)
        # print(ct)
        # Prepare landmark matrix and pixel dimensions
        L_in_mm = get_matrix_of_lmks(lmk)

        # save_3d_image(V, header, V.shape, image_folder, name, prefix='init')
        # save_transformed_landmarks(L_in_mm, lmk, landm_folder, name, prefix='init')
        print(header)
        # Determine pixel dimensions
        try:
            p = header.get('spacings')[:3]
            header['spacings'] = params.desired_spacings
        except:
            p = np.array([header['space directions'][0, 0], 
                          header['space directions'][1, 1], 
                          header['space directions'][2, 2]])
            header['space directions'] = np.array([[params.desired_spacings[0], 0, 0],
                                                   [0, params.desired_spacings[1], 0],
                                                   [0, 0, params.desired_spacings[2]]])

        img_size = np.array(V.shape)

        # --------- #
        # STEP 1. LP FILTER #
        # --------- #
        V = filter_3d_image(V) # type: ignore
        # save_3d_image(V, header, V.shape, image_folder, name, prefix='f')
        # save_transformed_landmarks(L_in_mm, lmk, landm_folder, name, prefix='f')

        # ----------  #
        # STEP 2. INTERPOLATE #
        # ----------  #

        image = sitk.GetImageFromArray(V)
        image.SetSpacing(p)  # Set original spacing

        new_spacing = params.desired_spacings
        original_size = image.GetSize()
        # new_size = [int(round(os * ospc / nspc)) for os, ospc, nspc in zip(original_size, p, new_spacing)]
        new_size = params.desired_size

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())

        resampled_image = resampler.Execute(image)
        V = sitk.GetArrayFromImage(resampled_image)
        header['sizes'] = params.desired_size
        
        # save_3d_image(V, header, np.array(V.shape), image_folder, name, prefix=F'inp')
        # save_transformed_landmarks(L_in_mm, lmk, landm_folder, name, prefix='inp')
        
        p = params.desired_spacings
        img_size = np.array(V.shape)

        # ----------  #
        # STEP 3. DOWNSAMPLE #
        # ----------  #
        
        # if params.drate:
        #     downsample_rates = np.array([
        #                                 params.drate, 
        #                                 params.drate, 
        #                                 params.drate]
        #                                 )
        # else:
        #     downsample_rates = np.ceil(img_size / params.desired_size).astype(int)

        # p *= downsample_rates
        # V = V[::downsample_rates[0], ::downsample_rates[1], ::downsample_rates[2]]
        # header['spacings'] = p
        
        # save_3d_image(V, header, np.array(V.shape), image_folder, name, prefix=F'ds{downsample_rates}')
        # save_transformed_landmarks(L_in_mm, lmk, landm_folder, name, prefix=F'ds{downsample_rates}')

        # Retrieve plane data from dictionary
        plane = dicto.get(name)
        if plane is None:
            print(f"Ground truth for {name} not found!")
            ct_not_found += 1
            continue
        ct += 1
        L_in_pix = L_in_mm / p
        
        V = np.transpose(V, (2, 1, 0))
        L_in_pix = swap_xz_coordinates(L_in_pix)
        # ------- #
        # Assumption 3: Centers are in RAS.
        center_gt_in_pix = np.array([-1*plane["center"][0], -1*plane["center"][1], plane["center"][2]]) / p

        # ------- #
        # STEP 4. PADDING #
        # ------- #
        
        # for dim in V.shape:
        #     if dim > params.desired_size[0]:                 # assuming iso
        #         # TODO: what should I do?
        #         print(f"dim exceeded: ", V.shape)
        #         pass
        # V, L_in_pix, center_gt_in_pix = pad_3d_image(V, L_in_pix, center_gt_in_pix, params.desired_size)
        # L_in_pix = L_in_pix.astype(np.float32)
        # center_gt_in_pix = center_gt_in_pix.astype(np.float32)
        # save_3d_image(V, header, params.desired_size, image_folder, name, prefix='pad')
        # save_transformed_landmarks(L_in_pix * p, lmk, landm_folder, name, prefix='pad')

        # ------ -------------- #
        # STEP 5. AFFINE TRANSFORMATION #
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


        L_in_pix = swap_xz_coordinates(L_in_pix)
        L_in_pix_norm = (L_in_pix - img_size/ 2.0) * ( 2.0 / img_size).astype(np.float32)
        inv_translation_vector = -1 * translation_vector
        Lhat_in_pix_norm = affine_transform(L_in_pix_norm, np.linalg.inv(rotation_3x3), inv_translation_vector)
        Lhat_in_pix = (Lhat_in_pix_norm / (2.0 / img_size).astype(np.float32)) + img_size / 2.0

        print(header)
        save_3d_image(Vhat, header, img_size, save_im_path)
        save_transformed_landmarks(Lhat_in_pix * p, lmk, save_csv_path, save_lmk_path)
    print("This number of images not found: ", ct_not_found)

def test_main():
    # Set default arguments
    class Params:
        root = "/media/yusuf/HDD 4TB/Casos Mar"
        save_dir = "/media/yusuf/HDD 4TB/Rotated/Processed"
        desired_spacings = [1.0, 1.0, 1.0]
        desired_size = [128, 128, 128]

    # Create a temporary directory for testing
    test_root = Params.root
    os.makedirs(test_root, exist_ok=True)

    # Mock parameters
    params = Params()

    dataframe = pd.read_csv("/media/yusuf/HDD 4TB/Rotated/Processed/Maternitatsinfo.csv")
    # dataframe = dataframe[dataframe['landmark_antonia_found'] == True].reset_index(drop=True)
    # dataframe = dataframe[:50]  # Limit to 10 rows for testing
    print(len(dataframe))

    # Run the main function
    try:
        main(dataframe, params)
        print("Test passed: main function executed without errors.")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_main()