import os
import pandas as pd
from tqdm import tqdm
import nrrd

weeks_global=[ '20 semanas', '26 semanas', '29 semanas', '30 semanas',
               '31 semanas', '32 semanas', '33 semanas', '35 semanas',
               '36 semanas', '38 semanas',
        ]

def extract_image(filename):
    """Extract the image into a 3D numpy array [x, y, z]. As it was saved in RAS

    Args:
      filename: Path and name of nifti file.

    Returns:
      data: A 3D numpy array [x, y, z]
      pix_dim: pixel spacings

    """

    data, header = nrrd.read(filename)

    if len(data.shape) == 4:
        data=data[:, :, :, 0]

    return data, header

def main(params):
    if len(params.dataset_clinics) !=1:
        print("Not valid!")
        return 
    
    root = params.sys + params.dataset_clinics[0]
    fnames = os.listdir(os.path.join(root, 'Casos'))

    # Initialize dataframes
    pinfo_df = []
    winfo_df = []
    sinfo_df = []
        
    for fname in tqdm(fnames, total=len(fnames)):
        path_to_im_fold = os.path.join(root, 'Casos', fname)
        # Get weeks.
        weeks = os.listdir(path_to_im_fold)
        tmp = []
        for w in weeks:
            if 'semanas' in w:
                tmp.append(w)
        weeks = tmp

        # Initialize patient row.
        row1 = {'pid': int(fname), 'path_to_im_fold': path_to_im_fold, 'path_to_seg_fold': os.path.join(root, 'Segmentaciones', fname), 'num_weeks': len(weeks)}

        for week in weeks_global:
            row1[week] = False
        for week in weeks:
            row1[week] = True
            scans = os.listdir(os.path.join(root, 'Casos', fname, week))
            tmp = []
            for s in scans:
                if 'nrrd' in s:
                    tmp.append(s)
            scans = tmp

            # Initialize patientinfo row
            row2 = {'pid': fname, 
                    'week': week, 
                    'path_to_week_im_folder': os.path.join(path_to_im_fold, week),
                    'path_to_week_seg_folder': os.path.join(root, 'Segmentaciones', fname, week),
                    'num_scans': len(scans),
                    'num_scan_landmark_pairs': -1
                    }
                                
            num_scan_landmark_pairs_ct = 0

            for scan in scans:
                scan_id = scan.split('.')[0]
                path_to_nrrd = os.path.join(path_to_im_fold, week, scan)
                _, header = extract_image(path_to_nrrd)
                row3 = {'pid': fname, 
                        'week': week, 
                        'scan': scan,
                        'full_id': scan_id,
                        'path_to_nrrd' : path_to_nrrd,
                        'orig_size_x' : header['sizes'][0],
                        'orig_size_y' : header['sizes'][1],
                        'orig_size_z' : header['sizes'][2],
                        'spacing_x' : header['spacings'][0],
                        'spacing_y' : header['spacings'][1],
                        'spacing_z' : header['spacings'][2],
                    }

                lmk_folder = os.path.join(root, 'Segmentaciones', fname, week, 'Lmks_Antonia')

                row3['landmark_antonia_found'] = True
                if not os.path.exists(lmk_folder):
                    row2['num_scan_landmark_pairs'] = num_scan_landmark_pairs_ct
                    continue
                else:
                    if str(scan_id + '_modified.fcsv') in os.listdir(lmk_folder):
                        num_scan_landmark_pairs_ct += 1
                        row3['path_to_fcsv_antonia'] = os.path.join(lmk_folder,str(scan_id + '_modified.fcsv'))
                        row3['path_to_csv_antonia'] = os.path.join(lmk_folder,str(scan_id + '.csv'))
                    else: 
                        if str(scan_id + '.fcsv') in os.listdir(lmk_folder):
                            num_scan_landmark_pairs_ct += 1
                            row3['path_to_fcsv_antonia'] = os.path.join(lmk_folder,str(scan_id + '.fcsv'))
                            row3['path_to_csv_antonia'] = os.path.join(lmk_folder,str(scan_id + '.csv'))
                        else: row3['landmark_antonia_found'] = False

                    row2['num_scan_landmark_pairs'] = num_scan_landmark_pairs_ct
                row3['heatmap_dts_map_path'] = os.path.join(params.save_dir, 'heatmaps', scan_id)
                row3['heatmap_dts_vol_path'] = os.path.join(params.save_dir, 'volumes', scan_id + '.nrrd')
                row3['heatmap_dts_lmk_path'] = os.path.join(params.save_dir, 'landmarks/fcsv', scan_id + '.fcsv')
                row3['heatmap_dts_csv_path'] = os.path.join(params.save_dir, 'landmarks/csv', scan_id + '.csv')

                row3['source'] = 'Casos Mar'
                sinfo_df.append(row3) 
            row2['source'] = 'Casos Mar'
            winfo_df.append(row2) 
        row1['source'] = 'Casos Mar'
        pinfo_df.append(row1)

    return sinfo_df
