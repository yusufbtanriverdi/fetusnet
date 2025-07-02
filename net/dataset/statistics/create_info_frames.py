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

def update(sinfo_df, params):
    # Construct full paths
    paths = sinfo_df['processed__vol_path'].apply(lambda x: os.path.join(params.sys + params.root, x))

    # Create a mask for existing files
    mask = paths.apply(os.path.exists)

    # Filter out missing files
    updated_sinfo_df = sinfo_df[mask].reset_index(drop=True)

    return updated_sinfo_df

def main(params):
    if len(params.dataset) !=1:
        print("Not valid!")
        return 
    
    if params.dataset[0] == 'Casos Mar':
        raw_files = params.sys + params.raw_dir + params.dataset[0]
        fnames = os.listdir(os.path.join(raw_files, 'Casos'))

        # Initialize dataframes
        pinfo_df = []
        winfo_df = []
        sinfo_df = []
            
        # ct = 0
        for fname in tqdm(fnames, total=len(fnames), desc='Scanning images......'):
            path_to_im_fold = os.path.join(raw_files, 'Casos', fname)
            # Get weeks.
            weeks = os.listdir(path_to_im_fold)
            tmp = []
            for w in weeks:
                if 'semanas' in w:
                    tmp.append(w)
            weeks = tmp

            # Initialize patient row.
            row1 = {'pid': int(fname), 'path_to_im_fold': path_to_im_fold, 'path_to_se_fold': os.path.join(raw_files, 'Segmentaciones', fname), 'num_weeks': len(weeks)}

            for week in weeks_global:
                row1[week] = False
            for week in weeks:
                row1[week] = True
                scans = os.listdir(os.path.join(raw_files, 'Casos', fname, week))
                tmp = []
                for s in scans:
                    if 'nrrd' in s:
                        tmp.append(s)
                scans = tmp

                # Initialize patientinfo row
                row2 = {'pid': fname, 
                        'week': week, 
                        'path_to_week_im_folder': os.path.join(path_to_im_fold, week),
                        'path_to_week_se_folder': os.path.join(raw_files, 'Segmentaciones', fname, week),
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
                            'path_to_nrrd' : 'Casos/' + '/' + fname + '/' + week + '/' + scan, 
                            'orig_size_x' : header['sizes'][0],
                            'orig_size_y' : header['sizes'][1],
                            'orig_size_z' : header['sizes'][2],
                            'spacing_x' : header['spacings'][0],
                            'spacing_y' : header['spacings'][1],
                            'spacing_z' : header['spacings'][2],
                        }

                    lmk_postfix = 'Segmentaciones/' + fname + '/' + week + '/Lmks_Antonia/'
                    lmk_folder = os.path.join(raw_files, 'Segmentaciones', fname, week, 'Lmks_Antonia')

                    row3['landmark_antonia_found'] = True
                    if not os.path.exists(lmk_folder):
                        row2['num_scan_landmark_pairs'] = num_scan_landmark_pairs_ct
                        continue
                    else:
                        if str(scan_id + '_modified.fcsv') in os.listdir(lmk_folder):
                            num_scan_landmark_pairs_ct += 1
                            row3['path_to_fcsv_antonia'] = lmk_postfix + scan_id + '_modified.fcsv'
                            row3['path_to_csv_antonia'] = lmk_postfix + scan_id + '_modified.csv'
                        else: 
                            if str(scan_id + '.fcsv') in os.listdir(lmk_folder):
                                num_scan_landmark_pairs_ct += 1
                                row3['path_to_fcsv_antonia'] = lmk_postfix + scan_id + '.fcsv'
                                row3['path_to_csv_antonia'] = lmk_postfix + scan_id + '.csv'
                            else: row3['landmark_antonia_found'] = False

                        row2['num_scan_landmark_pairs'] = num_scan_landmark_pairs_ct

                    row3['processed__vol_path'] = os.path.join(params.dataset[0] + '/volumes', scan_id + '.nrrd')
                    row3['processed__lmk_path'] = os.path.join(params.dataset[0] + '/landmarks/fcsv', scan_id + '.fcsv')
                    row3['processed__csv_path'] = os.path.join(params.dataset[0] + '/landmarks/csv', scan_id + '.csv')
                    row3['source'] = 'Casos Mar'

                    sinfo_df.append(row3) 

                row2['source'] = 'Casos Mar'
                winfo_df.append(row2) 

            row1['source'] = 'Casos Mar'
            pinfo_df.append(row1)

            # ct +=1
            # if ct > 3: break

        # Save files
        pd.DataFrame.from_records(pinfo_df).to_csv(params.sys + params.root + 'pinfo.csv', index=False)
        pd.DataFrame.from_records(winfo_df).to_csv(params.sys + params.root + 'winfo.csv', index=False)
        pd.DataFrame.from_records(sinfo_df).to_csv(params.sys + params.root + 'sinfo.csv', index=False)

        return sinfo_df

    if params.dataset[0] == 'Casos Maternitat':
        raw_files = params.sys + params.raw_dir + params.dataset[0]
        fnames = os.listdir(os.path.join(raw_files))

        # Initialize dataframes
        pinfo_df = []
        sinfo_df = []
        for fname in tqdm(fnames, total=len(fnames), desc='Scanning images......'):
            path_to_im_fold = os.path.join(raw_files, fname, 'nrrd')
            # no week information

            ## 
             # Initialize patient row.
            row1 = {'pid': int(fname), 
                    'path_to_im_fold': path_to_im_fold, 
                    'path_to_se_fold': os.path.join(raw_files, fname, 'PLY'), 
                    'num_scans': len(os.listdir(path_to_im_fold)),
                    'num_scan_landmark_pairs': -1
                    }
            
            num_scan_landmark_pairs_ct = 0

            for scan in os.listdir(path_to_im_fold):
                scan_id = scan.split('.')[0]
                path_to_nrrd = os.path.join(path_to_im_fold, scan)
                _, header = extract_image(path_to_nrrd)
                row2 = {'pid': fname, 
                        'week': week, 
                        'scan': scan_id,
                        'path_to_nrrd' : path_to_nrrd,
                        'orig_size_x' : header['sizes'][0],
                        'orig_size_y' : header['sizes'][1],
                        'orig_size_z' : header['sizes'][2],
                        'spacing_x' : header['spacings'][0],
                        'spacing_y' : header['spacings'][1],
                        'spacing_z' : header['spacings'][2],
                    }

                lmk_postfix = ...
                lmk_folder = os.path.join(path_to_im_fold, 'lmks' )

