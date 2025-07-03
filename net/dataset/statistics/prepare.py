import os
import pandas as pd
from tqdm import tqdm
import nrrd

# List of global week labels used across the dataset
weeks_global = [
    '20 semanas', '26 semanas', '29 semanas', '30 semanas',
    '31 semanas', '32 semanas', '33 semanas', '35 semanas',
    '36 semanas', '38 semanas',
]

def extract_image(filename):
    """
    Extracts image data and header from a NRRD file.
    If the data has 4 dimensions, reduces it to 3D by selecting the first volume.

    Args:
        filename (str): Path to the NRRD file.

    Returns:
        tuple:
            data (numpy.ndarray): 3D image array [x, y, z].
            header (dict): Header information including metadata.
    """
    data, header = nrrd.read(filename)

    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    return data, header


def perform_prepare(params):
    """
    Main function to scan datasets, gather patient and scan metadata,
    and save info CSV files.

    Args:
        params: Parameter object with attributes:
            - dataset (list): List of dataset names to process.
            - sys (str): Base system path.
            - raw_dir (str): Directory where raw data is stored.
            - root (str): Root directory for processed data.

    Returns:
        list: List of dictionaries with scan information (sinfo_df).
    """
    # Initialize lists to hold patient and scan info dictionaries
    pinfo_df = []
    sinfo_df = []

    # Iterate over each dataset specified in params
    for dataset in params.dataset:

        # Process "Casos Mar" dataset
        if dataset == 'Casos Mar':
            raw_files = params.sys + params.raw_dir + dataset
            fnames = os.listdir(os.path.join(raw_files, 'Casos'))

            ct = 0  # Counter for valid landmark pairs

            # Iterate over patient folders with progress bar
            for fname in tqdm(fnames, total=len(fnames), desc='Scanning images......'):
                path_to_im_fold = os.path.join(raw_files, 'Casos', fname)

                # List weeks folders, filter those containing 'semanas'
                weeks = [w for w in os.listdir(path_to_im_fold) if 'semanas' in w]

                # Initialize patient-level info dictionary
                row1 = {
                    'pid': int(fname),
                    'path_to_im_fold': path_to_im_fold,
                    'path_to_se_fold': os.path.join(raw_files, 'Segmentaciones', fname),
                    'num_weeks': len(weeks)
                }

                # Initialize presence flags for all global weeks to False
                for week in weeks_global:
                    row1[week] = False

                # Update True for weeks found in this patient folder
                for week in weeks:
                    row1[week] = True
                    scans = os.listdir(os.path.join(raw_files, 'Casos', fname, week))
                    scans = [s for s in scans if 'nrrd' in s]

                    num_scan_landmark_pairs_ct = 0

                    # Process each scan in the week folder
                    for scan in scans:
                        scan_id = scan.split('.')[0]
                        path_to_nrrd = os.path.join(path_to_im_fold, week, scan)

                        # Extract header only to get metadata
                        _, header = extract_image(path_to_nrrd)

                        row3 = {
                            'pid': fname,
                            'week': week,
                            'scan': scan,
                            'full_id': scan_id,
                            'path_to_nrrd': path_to_nrrd,
                            'orig_size_x': header['sizes'][0],
                            'orig_size_y': header['sizes'][1],
                            'orig_size_z': header['sizes'][2],
                            'spacing_x': header['spacings'][0],
                            'spacing_y': header['spacings'][1],
                            'spacing_z': header['spacings'][2],
                        }

                        lmk_folder = os.path.join(raw_files, 'Segmentaciones', fname, week, 'Lmks_Antonia')
                        row3['landmark_antonia_found'] = True

                        if not os.path.exists(lmk_folder):
                            row3['landmark_antonia_found'] = False
                            continue

                        option_top_fscv = os.path.join(lmk_folder, scan_id + '_modified.fcsv')
                        option_bttm_fscv = os.path.join(lmk_folder, scan_id + '.fcsv')

                        # Check for landmark files in prioritized order
                        if os.path.exists(option_top_fscv):
                            row3['path_to_lmk'] = option_top_fscv
                            ct += 1
                            num_scan_landmark_pairs_ct += 1
                        elif os.path.exists(option_bttm_fscv):
                            row3['path_to_lmk'] = option_bttm_fscv
                            ct += 1
                            num_scan_landmark_pairs_ct += 1
                        else:
                            continue

                        option_top_csv = os.path.join(lmk_folder, scan_id + '_modified.csv')
                        option_bttm_csv = os.path.join(lmk_folder, scan_id + '.csv')

                        if os.path.exists(option_top_csv):
                            row3['path_to_csv'] = option_top_csv
                            ct += 1
                            num_scan_landmark_pairs_ct += 1
                        elif os.path.exists(option_bttm_csv):
                            row3['path_to_csv'] = option_bttm_csv
                            ct += 1
                            num_scan_landmark_pairs_ct += 1
                        else:
                            continue

                        # Add processed file paths and source info
                        row3['processed__vol_path'] = os.path.join(dataset, 'volumes', scan_id + '.nrrd')
                        row3['processed__lmk_path'] = os.path.join(dataset, 'landmarks', 'fcsv', scan_id + '.fcsv')
                        row3['processed__csv_path'] = os.path.join(dataset, 'landmarks', 'csv', scan_id + '.csv')
                        row3['source'] = 'Casos Mar'

                        if not row3['landmark_antonia_found']:
                            continue

                        ct += 1
                        sinfo_df.append(row3)

                row1['source'] = 'Casos Mar'
                pinfo_df.append(row1)

        # Process "Casos Maternitat" dataset
        elif dataset == 'Casos Maternitat':
            raw_files = params.sys + params.raw_dir + dataset
            fnames = os.listdir(os.path.join(raw_files))

            pinfo_df = []
            sinfo_df = []
            ct = 0

            # Iterate over patient folders with progress bar
            for fname in tqdm(fnames, total=len(fnames), desc='Scanning patients......'):
                path_to_im_fold = os.path.join(raw_files, fname, 'nrrd')
                if not os.path.exists(path_to_im_fold):
                    continue

                # Initialize patient-level info dictionary
                row1 = {
                    'pid': str(fname),
                    'path_to_im_fold': path_to_im_fold,
                    'path_to_se_fold': os.path.join(raw_files, fname, 'PLY'),
                    'num_scans': len(os.listdir(path_to_im_fold)),
                    'num_scan_landmark_pairs': -1
                }

                num_scan_landmark_pairs_ct = 0

                # Process each scan in patient's nrrd folder
                for scan in os.listdir(path_to_im_fold):
                    scan_id = scan.split('.')[0]
                    path_to_nrrd = os.path.join(path_to_im_fold, scan)

                    # Extract header only for metadata
                    _, header = extract_image(path_to_nrrd)

                    row2 = {
                        'pid': fname,
                        'full_id': scan_id,
                        'path_to_nrrd': path_to_nrrd,
                        'orig_size_x': header['sizes'][0],
                        'orig_size_y': header['sizes'][1],
                        'orig_size_z': header['sizes'][2],
                    }

                    # Handle spacings info that might be stored differently in header
                    if 'spacings' in header:
                        row2['spacing_x'] = header['spacings'][0]
                        row2['spacing_y'] = header['spacings'][1]
                        row2['spacing_z'] = header['spacings'][2]
                    else:
                        row2['spacing_x'] = header['space directions'][0, 0]
                        row2['spacing_y'] = header['space directions'][1, 1]
                        row2['spacing_z'] = header['space directions'][2, 2]

                    lmk_folder = os.path.join(raw_files, fname, 'Lmks')
                    option_top_fscv = os.path.join(lmk_folder, scan_id + '_modified.fcsv')
                    option_bttm_fscv = os.path.join(lmk_folder, scan_id + '.fcsv')

                    # Check for landmark files in prioritized order
                    if os.path.exists(option_top_fscv):
                        row2['path_to_lmk'] = option_top_fscv
                        num_scan_landmark_pairs_ct += 1
                    elif os.path.exists(option_bttm_fscv):
                        row2['path_to_lmk'] = option_bttm_fscv
                        num_scan_landmark_pairs_ct += 1
                    else:
                        continue

                    option_top_csv = os.path.join(lmk_folder, scan_id + '_modified.csv')
                    option_bttm_csv = os.path.join(lmk_folder, scan_id + '.csv')

                    if os.path.exists(option_top_csv):
                        row2['path_to_csv'] = option_top_csv
                        num_scan_landmark_pairs_ct += 1
                    elif os.path.exists(option_bttm_csv):
                        row2['path_to_csv'] = option_bttm_csv
                        num_scan_landmark_pairs_ct += 1
                    else:
                        continue

                    # Try to load landmark csv to ensure it's valid
                    try:
                        lmk = pd.read_csv(row2['path_to_csv'])
                    except Exception as e:
                        # print(str(e))
                        # print(f"Could not load landmarks from {row2['path_to_csv']}! Skipping!! I used to work on the other version...")
                        continue

                    ct += 1
                    row2['landmark_antonia_found'] = False
                    row2['week'] = -1

                    # Add processed file paths and source info
                    row2['processed__vol_path'] = os.path.join(dataset, 'volumes', scan_id + '.nrrd')
                    row2['processed__lmk_path'] = os.path.join(dataset, 'landmarks', 'fcsv', scan_id + '.fcsv')
                    row2['processed__csv_path'] = os.path.join(dataset, 'landmarks', 'csv', scan_id + '.csv')
                    row2['source'] = 'Casos Maternitat'

                    sinfo_df.append(row2)

                row1['source'] = 'Casos Maternitat'
                row1['num_scan_landmark_pairs'] = num_scan_landmark_pairs_ct
                pinfo_df.append(row1)

    # Save patient and scan info DataFrames as CSV files
    pd.DataFrame.from_records(pinfo_df).to_csv(params.sys + params.root + 'pinfo.csv', index=False)
    pd.DataFrame.from_records(sinfo_df).to_csv(params.sys + params.root + 'sinfo.csv', index=False)
    return sinfo_df
