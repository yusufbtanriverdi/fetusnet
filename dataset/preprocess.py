import json
import logging
import os

import nrrd
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage
from tqdm import tqdm

from dataset.utility.gtpp.utils.SonoNet import SonoNet
from dataset.utility.standardization import standardize_with_gt, standardize_with_model
from doc.info.template import create_template


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

# ADDED with NEW DEFAULT MISMATCHES: Index Order explicit set
def extract_image(filename, index_order='C'):
    """Extract the image into a 3D numpy array [x, y, z]. As it was saved in RAS
    Args:
      filename: Path and name of nifti file.

    Returns:
      data: A 3D numpy array [x, y, z]
      header: An OrderedDict
    """

    data, header = nrrd.read(filename, index_order=index_order)
    if len(data.shape) == 4:
        data=data[:, :, :, 0]

    return data, header

def filter_3d_image(volume, filter_size=3, mode='reflect'):

    low_pass_filter = np.ones((filter_size,) * 3) / (filter_size ** 3)

    # Apply filtering for smoothing
    return ndimage.convolve(volume, low_pass_filter, mode=mode)

def get_matrix_of_lmks(landmarks_df):
    """
    Convert the DataFrame of landmarks into a matrix in LPS (Left-Posterior-Superior) format.
    
    Parameters:
    landmarks_df (pd.DataFrame): DataFrame containing landmark data.
    
    Returns:
    np.ndarray: Numpy array representing the matrix of landmarks in LPS format.
    """
    # Assuming the DataFrame has columns named 'x', 'y', 'z' for the coordinates
    # Modify according to your actual column names
    # x = -landmarks_df['x'].values  # Flip x for LPS
    # y = -landmarks_df['y'].values  # Flip y for LPS
    # z = landmarks_df['z'].values

    # There are some instances that is in RAS but do not need this sign change. Maybe abs is better?
    x = abs(landmarks_df['x'].values)  # Flip x for LPS
    y = abs(landmarks_df['y'].values)  # Flip y for LPS
    z = abs(landmarks_df['z'].values)

    # This works but still the case 39-20s-08 is shifted a bit. 
    coords = np.array([x, y, z])
    lmk_matrix = np.column_stack(coords).astype(np.float32)
    
    return lmk_matrix

def swap_xz_coordinates(lmks):
    """
    Swaps the x and z coordinates of each landmark point in the given array.

    Args:
        lmks (numpy.ndarray or torch.Tensor): A 2D array of landmarks with shape (N, 3), 
                                              where N is the number of landmarks, and the 
                                              coordinates are [x, y, z].
    Returns:
        numpy.ndarray or torch.Tensor: The array with x and z coordinates swapped.
    """
    lmks[:, [0, 2]] = lmks[:, [2, 0]]
    return lmks

def save_transformed_landmarks(Lhat, lmk, output_csv_path, output_fscv_path):
    """
    Save transformed landmarks to CSV and FCSV files with predefined header lines.

    Parameters:
    - Lhat (np.ndarray): Array containing transformed landmark coordinates.
    - lmk (pd.DataFrame): DataFrame to copy structure from.
    - output_path (str): Directory path to save the files.
    - name (str): Base name for the output files.
    """
    # Copy structure of landmarks and update with transformed coordinates
    Lhat_df = lmk.copy()
    for idx in range(Lhat.shape[0]):
        Lhat_df.loc[idx, ['x', 'y', 'z']] = Lhat[idx, :]

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
        f.writelines(','.join(map(str, row.values)) + '\n' for index, row in Lhat_df.iterrows())

def get_spacing(header):
    """
    Extract voxel spacing from the NRRD header.

    Use the first three values from 'spacings' when available. If that
    operation fails, use the diagonal entries of 'space directions'.
    """
    try:
        spacing = header.get("spacings")[:3]

    except Exception:
        spacing = np.array([
            header["space directions"][0, 0],
            header["space directions"][1, 1],
            header["space directions"][2, 2],
        ])

    return spacing

def prepare_dataset_metadata(dataframe):
    """
    Extract filenames and scan directories, load ground-truth plane
    metadata, and initialize the landmark-file availability flag.

    The trimmed directory list is retained from the original code,
    although it is not subsequently used there.
    """
    filenames = list(dataframe["fcaso"].values)
    filedirs = list(dataframe["fscan"].values)

    plane_metadata = json.loads(
        get_file_list("doc/info//gt.txt")[0]
    )

    trimmed_filedirs = []

    for filedir in filedirs:
        parts = filedir.strip("/").split("/")
        trimmed_filedirs.append("/".join(parts[3:]))

    dataframe.loc[:, "csvfound"] = True

    return filenames, filedirs, trimmed_filedirs, plane_metadata

def create_output_directories(params):
    """
    Build and create the original output directory structure.

    Directory creation retains the original existence checks and os.mkdir
    calls rather than introducing recursive directory creation.
    """
    output_dir = os.path.join(
        params.ds.sys,
        params.preprocessing.save_dir,
    )

    landmarks_dir = os.path.join(output_dir, "landmarks")
    volumes_dir = os.path.join(output_dir, "volumes")
    csv_dir = os.path.join(landmarks_dir, "csv")
    fcsv_dir = os.path.join(landmarks_dir, "fcsv")

    directories = [
        output_dir,
        landmarks_dir,
        volumes_dir,
        csv_dir,
        fcsv_dir,
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.mkdir(directory)

    return output_dir, landmarks_dir, volumes_dir, csv_dir, fcsv_dir

def initialize_gtpp_model(params):
    """
    Initialize SonoNet, optionally load its checkpoint, and enable
    evaluation mode.

    In the original pipeline, this step runs when
    params.preprocessing.gtpp is enabled.
    """
    model = SonoNet(
        params.preprocessing.gtpp.model,
        dropout_p=0.25,
    )
    model.to(params.device)

    # Restore the saved model weights when resuming is enabled.
    if params.preprocessing.gtpp.model.resume:
        checkpoint = torch.load(
            os.path.join(
                params.preprocessing.gtpp.model_dir,
                params.preprocessing.gtpp.checkpoint_name,
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model

def prepare_sample_paths(dataframe, index, params):
    """
    Extract sample identifiers and construct input and output paths.

    Preserve the original dataframe access and output-path concatenation.
    """
    image_path = os.path.join(dataframe.loc[index, "fscan"])
    sample_name = dataframe.loc[index, "osid"]

    save_image_path = (
        params.ds.sys
        + params.preprocessing.save_dir
        + "/"
        + dataframe.loc[index, "mscan"]
    )

    save_csv_path = (
        params.ds.sys
        + params.preprocessing.save_dir
        + "/"
        + dataframe.loc[index, "mcsv"]
    )

    save_landmark_path = (
        params.ds.sys
        + params.preprocessing.save_dir
        + "/"
        + dataframe.loc[index, "mlmk"]
    )

    landmark_path = dataframe.loc[index, "_fcsv"]

    return (
        image_path,
        sample_name,
        save_image_path,
        save_csv_path,
        save_landmark_path,
        landmark_path,
    )

def load_landmark_table(
    landmark_path,
    sample_name,
    dataframe,
    index,
    missing_count,
    logger,
):
    """
    Load the landmark table and retain the original failure reporting.

    Return None on loading failure so the future main loop can execute
    the original continue statement.
    """
    try:
        landmark_table = pd.read_csv(landmark_path)

    except Exception as error:
        logger.error(str(error))
        missing_count += 1

        logger.warning(
            f"Could not load landmarks from {landmark_path} "
            f"for {sample_name}! Skipping!! Count: {missing_count}"
        )

        dataframe.loc[index, "csvfound"] = False

        return None, missing_count

    return landmark_table, missing_count

def load_volume_and_landmarks(
    image_path,
    landmark_table,
    index_order='F'
):
    """
    Check landmark count, load the XYZ volume, extract spacing, and
    convert landmark coordinates from millimeters to pixels.

    A count other than 19 produces a warning but does not stop processing.
    """
    landmarks_mm = get_matrix_of_lmks(landmark_table)

    # Read the image in the original XYZ index order.
    volume, header = extract_image(
        image_path,
        index_order=index_order
    )

    spacing = get_spacing(header)
    template = create_template(spacing)
    landmarks_pix = landmarks_mm / spacing

    return (
        volume,
        header,
        spacing,
        template,
        landmarks_mm,
        landmarks_pix,
    )

def downsample_volume(volume, spacing, downsample):
    """
    Apply the original stride-based volume downsampling.

    In the original pipeline, this step runs when preprocessing.gtpp or
    preprocessing.downsample is enabled. Landmark coordinates are not
    modified by this operation.
    """

    spacing = np.multiply(spacing, downsample)

    volume = volume[
        0:volume.shape[0]:downsample,
        0:volume.shape[1]:downsample,
        0:volume.shape[2]:downsample,
    ]

    return volume, spacing

def pad_volume(volume, landmarks_pix, params, logger):
    """
    Pad the volume to the configured size and shift landmark coordinates.

    Preserve the original padding arithmetic and tensor conversion.
    Return a validity flag so the main loop can retain its skip behavior.

    """
    desired_size = (
        params.preprocessing.gtpp.model.desired_size
    )

    diff_size = (
        np.array(desired_size) - np.array(volume.shape[0:3])
    )
    diff_size = diff_size / 2

    a = int(diff_size[2])
    b = int(diff_size[1])
    c = int(diff_size[0])

    # Determine the extra padding applied for odd size differences.
    if (diff_size[2] * 2) % 2 == 0:
        x = 0
    else:
        x = 1

    if (diff_size[1] * 2) % 2 == 0:
        y = 0
    else:
        y = 1

    if (diff_size[0] * 2) % 2 == 0:
        z = 0
    else:
        z = 1

    padding = torch.nn.ConstantPad3d(
        [a + x, a, b + y, b, c + z, c],
        0,
    )

    volume = torch.from_numpy(volume.astype(np.float32))
    volume = padding(volume)

    # Preserve the original dimension check before updating landmarks.
    if not np.array_equal(
        np.array(volume.shape),
        np.array(desired_size),
    ):
        logger.warning("Skipped inconsistent dimension!!!!")
        return volume, landmarks_pix, False

    landmark_padding = np.array(
        [c + z, b + y, a + x],
        dtype=np.float64,
    )
    landmarks_pix += landmark_padding

    return volume, landmarks_pix, True

def resample_volume(volume, spacing, params):
    """
    Resample the volume to the configured spacing and size.

    Preserve the original SimpleITK setup and spacing update.
    Landmark coordinates are not modified in this step.
    """
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)

    new_spacing = params.preprocessing.bspline.spacing
    new_size = params.preprocessing.bspline.size

    # Configure output geometry and the interpolation method.
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    resampled_image = resampler.Execute(image)

    volume = sitk.GetArrayFromImage(resampled_image)
    spacing = new_spacing

    return volume, spacing

def save_processed_sample(
    volume,
    landmarks_pix,
    spacing,
    template,
    landmark_table,
    save_image_path,
    save_csv_path,
    save_landmark_path,
    index_order = 'C'
):
    """
    Convert landmarks to millimeters, restore XYZ coordinate order,
    and save the processed volume and landmark files.

    Preserve the original template and NRRD index-order settings.
    """

    nrrd.write(
        save_image_path,
        volume,
        header=template,
        index_order=index_order
    )

    save_transformed_landmarks(
        landmarks_pix * spacing,
        landmark_table,
        save_csv_path,
        save_landmark_path,
    )

def preprocess(dataframe, params, logger=None):
    if logger is None:
        logger = logging.getLogger('my_project_logger') 

    filenames, filedirs, trimmed_filedirs, plane_metadata = prepare_dataset_metadata(dataframe)
    ct = 0
    ct_not_found = 0
    dataframe.loc[:, 'csvfound'] = True
    create_output_directories(params)
    read_index_order = params.preprocessing.read_index_order
    write_index_order = params.preprocessing.write_index_order

    # Restrict processing to the requested filenames when a sample list is provided.
    sample_list = getattr(params.preprocessing, "sample_list", None)
    print(sample_list)
    for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
        image_path, sample_name, save_image_path, save_csv_path, save_landmark_path, landmark_path = prepare_sample_paths(dataframe, index, params)
        if sample_list is not None and dataframe.loc[index, "npid"] not in sample_list:
                continue
        # Skip processing if image already exists
        if os.path.exists(save_image_path):
            logger.warning("Attention! Image exists!")
            continue

        landmark_table, ct_not_found = load_landmark_table(landmark_path, sample_name, dataframe, index, ct_not_found, logger)
        if landmark_table is None:
            continue
        if len(landmark_table) != 19:
            logger.warning(
                "Something wrong!! Landmark count mismatch: %s",
                filename,
            )

        volume, header, spacing, template, landmarks_mm, landmarks_pix = load_volume_and_landmarks(image_path, landmark_table, read_index_order)
        volume = filter_3d_image(
            volume,
            params.preprocessing.filter.filter_size,
            params.preprocessing.filter.mode,
        )

        if params.preprocessing.path1: # SonoNet 3DFetus
            # Step 1
            volume, spacing = downsample_volume(volume, spacing, downsample=params.preprocessing.downsampling_factor)
            landmarks_pix = landmarks_mm / spacing
            template = create_template(spacing)
            # Step 2
            volume, landmarks_pix, consistent = pad_volume(volume, landmarks_pix, params, logger) 
            if not consistent:
                continue
            # Step 3
            model = initialize_gtpp_model(params)
            # PyTorch transformation accepts ZYX.
            volume = torch.permute(volume, (2, 1, 0))
            volume, landmarks_pix, ct = standardize_with_model(params, model, params.device, volume, landmarks_pix, ct)
            volume = volume.cpu().numpy().astype(np.float32)

        elif params.preprocessing.path2: # GT alignment- PIPPI2026
            volume, spacing = resample_volume(volume, spacing, params)
            landmarks_pix = landmarks_mm / spacing
            template = create_template(spacing)
            # PyTorch transformation accepts ZYX.
            volume = np.transpose(volume, (2, 1, 0))
            volume, landmarks_pix, ct = standardize_with_gt(plane_metadata, sample_name, volume, spacing, landmarks_pix, ct, logger)

        elif params.preprocessing.path3: # downsample-pad-standardize-resample 
            # Step 1
            volume, spacing = downsample_volume(volume, spacing, downsample=params.preprocessing.downsampling_factor)
            landmarks_pix = landmarks_mm / spacing
            template = create_template(spacing)
            # Step 2
            volume, landmarks_pix, consistent = pad_volume(volume, landmarks_pix, params, logger) 
            if not consistent:
                continue
            # Step 3
            model = initialize_gtpp_model(params)
            # PyTorch transformation accepts ZYX.
            volume = torch.permute(volume, (2, 1, 0))
            volume, landmarks_pix, ct = standardize_with_model(params, model, params.device, volume, landmarks_pix, ct)
            volume = volume.cpu().numpy().astype(np.float32)
            landmarks_mm = landmarks_pix * spacing
            # Update spacing and resample volume.
            volume, spacing = resample_volume(volume, spacing, params)
            landmarks_pix = landmarks_mm / spacing
            template = create_template(spacing)
            
        elif params.preprocessing.path5: # resample-swap
            volume, spacing = resample_volume(volume, spacing, params)
            landmarks_pix = landmarks_mm / spacing
            landmarks_pix = swap_xz_coordinates(landmarks_pix)
            template = create_template(spacing)

        save_processed_sample(volume, landmarks_pix, spacing, template, landmark_table, save_image_path, save_csv_path, save_landmark_path, write_index_order)

    return ct, ct_not_found, sample_list, params