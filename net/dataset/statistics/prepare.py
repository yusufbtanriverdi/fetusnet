import os
import pandas as pd
from tqdm import tqdm
import nrrd
import json

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

def extract_list_lmks_fromfcsv(file_path: str):
    """
    Reads a .fcsv (Slicer fiducial file) as plain text and extracts all values
    from the 'associatedNodeID' column into a list.
    Empty values are ignored.
    """
    lmks = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip headers or comment lines
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 14:  # 'associatedNodeID' is the 14th column (0-indexed)
                val = parts[11]
                if val:  # only add non-empty entries
                    lmks.append(val)
    return lmks

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
    save_dir = params.sys + params.move_dir
    os.makedirs(save_dir, exist_ok=True)
    # Load ground truth data for standard planes from JSON file
    dicto = json.loads(get_file_list('doc/info//gt.txt')[0])

    ######  CASOS ######
    # Iterate over each dataset specified in params
    for dataset in params.dataset:
        # Process "Casos Mar" dataset
        if dataset == 'Casos Mar':
            ds_id = '1'
            raw_files = params.sys + params.raw_dir + dataset
            casos = os.listdir(os.path.join(raw_files, 'Casos'))
            ds_dict = {'ds_id': ds_id, 
                       'ds': dataset,
                       'num_casos': len(casos)}
            # Iterate over patient folders with progress bar
            for caso in tqdm(casos, total=len(casos), desc='--------------------Scanning subfolders (Casos)..................'):
                caso_id = '0' + str(caso) if int(caso) >= 10 else '00' + str(caso)

                fcaso = os.path.join(raw_files, 'Casos', caso)
                # List weeks folders, filter those containing 'semanas'
                weeks = [w for w in os.listdir(fcaso) if 'semanas' in w]
                npid = str(ds_id) + str(caso_id) 

                # Initialize patient-level info dictionary
                caso_dict = {
                            'opid': caso,
                            'npid': npid,
                            'fcaso': fcaso,
                            'num_weeks': len(weeks), 
                            'list_weeks': weeks
                }

                scan_counter = 0
                # for week in tqdm(weeks, total=len(weeks), desc='--------------------Scanning subfolders (XX semanas)..................'):
                for week in weeks:
                    fweek = os.path.join(raw_files, 'Casos', caso, week)
                    scans = os.listdir(fweek)
                    # It is a scan if only .nrrd
                    scans = [s for s in scans if 'nrrd' in s]
                    scan_counter += len(scans)

                    # Process each scan in the week folder
                    # for scan in tqdm(scans, total=len(scans), desc='--------------------Scanning scans..................'):
                    for scan in scans:
                        osid = scan.split('.')[0]
                        if dicto.get(osid) is None:
                            rot_found = False
                        else: rot_found = True

                        fscan = os.path.join(raw_files, 'Casos', caso, week, scan)

                        # Extract header only to get metadata
                        _, header = extract_image(fscan)
                        nsid = npid + week.split(' ')[0] + osid.split('-')[-1]

                        mscan = os.path.join(dataset, 'volumes', nsid + '.nrrd')
                        scan_dict  = {
                            'nsid': nsid,
                            'npid': npid,
                            'opid': caso,
                            'week': week,
                            'opid': caso,
                            'osid': osid,
                            'fcaso': fcaso,
                            'fweek': fweek,
                            'fscan': fscan,
                            'os0': header['sizes'][0],
                            'os1': header['sizes'][1],
                            'os2': header['sizes'][2],
                            'ovd0': header['spacings'][0],
                            'ovd1': header['spacings'][1],
                            'ovd2': header['spacings'][2],
                            'ds': dataset,
                            # extras to keep consistent
                            'flmk': '',
                            'flmk_full': '',
                            'visibles': [],
                            '_fcsv': '',
                            'fply': '',
                            'mscan': mscan,
                            'mlmk': '',
                            'mcsv': '',
                            'rot_found' : rot_found
                        }
                         
                        lmk_folder = os.path.join(raw_files, 'Segmentaciones', caso, week, 'Lmks_Antonia') 
                        lmks_found =  False
                        if os.path.exists(lmk_folder):
                            flmk = os.path.join(lmk_folder, osid + '.fcsv')
                            flmk_modified = os.path.join(lmk_folder, osid + '_modified.fcsv')
                            fcsv = os.path.join(lmk_folder, osid + '.csv')
                            fcsv_modified = os.path.join(lmk_folder, osid + '_modified.csv')    
                            if os.path.exists(flmk) or os.path.exists(flmk_modified): lmks_found = True
                            if os.path.exists(flmk_modified): scan_dict['flmk_full'] = flmk_modified

                            if os.path.exists(flmk):
                                scan_dict['flmk'] = flmk
                                scan_dict['visibles'] = extract_list_lmks_fromfcsv(flmk)
                            
                            if os.path.exists(fcsv): scan_dict['_fcsv'] = fcsv
                            elif os.path.exists(fcsv_modified): scan_dict['_fcsv'] = fcsv
                            else: scan_dict['_fcsv'] = ''
                        scan_dict['lmks_found'] = lmks_found
                        if lmks_found:                         
                            mlmk = os.path.join(dataset, 'landmarks', 'fcsv', nsid + '.fcsv')
                            mcsv = os.path.join(dataset, 'landmarks', 'csv', nsid + '.csv')
                            scan_dict['mlmk'] = mlmk
                            scan_dict['mcsv'] = mcsv

                        ply_folder = os.path.join(raw_files, 'Segmentaciones', caso, week, 'PLY') 
                        plys_found = False
                        if os.path.exists(ply_folder):
                            fply = os.path.join(ply_folder, osid + '.ply')
                            if os.path.exists(fply):
                                scan_dict['fply'] = fply
                                plys_found = True
                        scan_dict['plys_found'] = plys_found

                        sinfo_df.append(scan_dict)
                        
                caso_dict['num_scans']= scan_counter
                pinfo_df.append(caso_dict)


        if dataset == 'Casos Maternitat':
            ds_id = '2'
            raw_files = params.sys + params.raw_dir + dataset
            casos = os.listdir(os.path.join(raw_files))
            ds_dict = {'ds_id': ds_id, 
                       'ds': dataset,
                       'num_casos': len(casos)}
            # Iterate over patient folders with progress bar
            for caso in tqdm(casos, total=len(casos), desc='--------------------Scanning subfolders (Casos)..................'):
                caso_id = str(caso).split('A')[-1]

                fcaso = os.path.join(raw_files, caso)
                # List weeks folders, filter those containing 'semanas'
                weeks = ['']
                npid = str(ds_id) + str(caso_id) 

                # Initialize patient-level info dictionary
                caso_dict = {
                            'opid': caso,
                            'npid': npid,
                            'fcaso': fcaso,
                            'num_weeks': 0, 
                            'list_weeks': weeks
                }
                scan_counter = 0
                # for week in tqdm(weeks, total=len(weeks), desc='--------------------Scanning subfolders (XX semanas)..................'):
                for week in weeks:
                    fweek = os.path.join(raw_files, caso, 'nrrd')
                    if not os.path.exists(fweek): break
                    scans = os.listdir(fweek)
                    # It is a scan if only .nrrd
                    scans = [s for s in scans if 'nrrd' in s]
                    scan_counter += len(scans)

                    # Process each scan in the week folder
                    # for scan in tqdm(scans, total=len(scans), desc='--------------------Scanning scans..................'):
                    for scan in scans:
                        osid = scan.split('.')[0]
                        if dicto.get(osid) is None:
                            rot_found = False
                        else: rot_found = True
                        
                        fscan = os.path.join(raw_files, caso, 'nrrd', scan)
                        # Extract header only to get metadata
                        _, header = extract_image(fscan)
                        nsid = npid + '00' + osid.split('_')[-1]
                        mscan = os.path.join(dataset, 'volumes', nsid + '.nrrd')

                        scan_dict  = {
                            'nsid': nsid,
                            'npid': npid,
                            'opid': caso,
                            'week': week,
                            'opid': caso,
                            'osid': osid,
                            'fcaso': fcaso,
                            'fweek': fweek,
                            'fscan': fscan,
                            'os0': header['sizes'][0],
                            'os1': header['sizes'][1],
                            'os2': header['sizes'][2],
                            'ovd0': header['spacings'][0] if 'spacings' in header else header['space directions'][0, 0],
                            'ovd1': header['spacings'][1] if 'spacings' in header else header['space directions'][1, 1],
                            'ovd2': header['spacings'][2] if 'spacings' in header else header['space directions'][2, 2],
                            'ds': dataset,
                            # extras to keep consistent
                            'flmk': '',
                            'flmk_full': '',
                            'visibles': [],
                            '_fcsv': '',
                            'fply': '',
                            'mscan': mscan,
                            'mlmk': '',
                            'mcsv': '',
                            'rot_found' : rot_found
                        }
                         
                        lmk_folder = os.path.join(raw_files, caso, 'Lmks') 
                        lmks_found =  False
                        if os.path.exists(lmk_folder):
                            flmk = os.path.join(lmk_folder, osid + '.fcsv')
                            flmk_modified = os.path.join(lmk_folder, osid + '_modified.fcsv')
                            fcsv = os.path.join(lmk_folder, osid + '.csv')
                            fcsv_modified = os.path.join(lmk_folder, osid + '_modified.csv')    
                            if os.path.exists(flmk) or os.path.exists(flmk_modified): lmks_found = True
                            if os.path.exists(flmk_modified): scan_dict['flmk_full'] = flmk_modified

                            if os.path.exists(flmk):
                                scan_dict['flmk'] = flmk
                                scan_dict['visibles'] = extract_list_lmks_fromfcsv(flmk)
                            
                            if os.path.exists(fcsv): scan_dict['_fcsv'] = fcsv
                            elif os.path.exists(fcsv_modified): scan_dict['_fcsv'] = fcsv
                            else: scan_dict['_fcsv'] = ''
                        scan_dict['lmks_found'] = lmks_found
                        if lmks_found:                         
                            mlmk = os.path.join(dataset, 'landmarks', 'fcsv', nsid + '.fcsv')
                            mcsv = os.path.join(dataset, 'landmarks', 'csv', nsid + '.csv')
                            scan_dict['mlmk'] = mlmk
                            scan_dict['mcsv'] = mcsv

                        ply_folder = os.path.join(raw_files, caso, week, 'PLY') 
                        plys_found = False
                        if os.path.exists(ply_folder):
                            fply = os.path.join(ply_folder, osid + '.ply')
                            if os.path.exists(fply):
                                scan_dict['fply'] = fply
                                plys_found = True
                        scan_dict['plys_found'] = plys_found

                        sinfo_df.append(scan_dict)
                caso_dict['num_scans']= scan_counter
                pinfo_df.append(caso_dict)

        if dataset == 'Estudio Dexeus':
            ds_id = '3'
            raw_files = params.sys + params.raw_dir + dataset
            casos = os.listdir(os.path.join(raw_files, 'Casos'))
            ds_dict = {'ds_id': ds_id, 
                       'ds': dataset,
                       'num_casos': len(casos)}
            # Iterate over patient folders with progress bar
            for caso in tqdm(casos, total=len(casos), desc='--------------------Scanning subfolders (Casos)..................'):
                caso_id = caso.split('-')[-1]
                if int(caso_id) < 10: caso_id = '00' + caso_id
                elif int(caso_id) >=10 and int(caso_id) <100: caso_id = '0' + caso_id

                fcaso = os.path.join(raw_files, 'Casos', caso)
                # List weeks folders, filter those containing 'semanas'
                weeks = [w for w in os.listdir(fcaso) if 'semanas' in w]
                npid = str(ds_id) + str(caso_id) 

                # Initialize patient-level info dictionary
                caso_dict = {
                            'opid': caso,
                            'npid': npid,
                            'fcaso': fcaso,
                            'num_weeks': len(weeks), 
                            'list_weeks': weeks
                }

                scan_counter = 0
                # for week in tqdm(weeks, total=len(weeks), desc='--------------------Scanning subfolders (XX semanas)..................'):
                for week in weeks:
                    fweek = os.path.join(raw_files, 'Casos', caso, week)
                    scans = os.listdir(fweek)
                    # It is a scan if only .nrrd
                    scans = [s for s in scans if 'nrrd' in s]
                    scan_counter += len(scans)

                    # Process each scan in the week folder
                    # for scan in tqdm(scans, total=len(scans), desc='--------------------Scanning scans..................'):
                    for scan in scans:
                        osid = scan.split('.')[0]
                        if dicto.get(osid) is None:
                            rot_found = False
                        else: rot_found = True

                        fscan = os.path.join(raw_files, 'Casos', caso, week, scan)

                        # Extract header only to get metadata
                        _, header = extract_image(fscan)
                        nsid = npid + week.split(' ')[0] + osid.split('s-')[-1]

                        mscan = os.path.join(dataset, 'volumes', nsid + '.nrrd')
                        scan_dict  = {
                            'nsid': nsid,
                            'npid': npid,
                            'opid': caso,
                            'week': week,
                            'opid': caso,
                            'osid': osid,
                            'fcaso': fcaso,
                            'fweek': fweek,
                            'fscan': fscan,
                            'os0': header['sizes'][0],
                            'os1': header['sizes'][1],
                            'os2': header['sizes'][2],
                            'ovd0': header['spacings'][0],
                            'ovd1': header['spacings'][1],
                            'ovd2': header['spacings'][2],
                            'ds': dataset,
                            # extras to keep consistent
                            'flmk': '',
                            'flmk_full': '',
                            'visibles': [],
                            '_fcsv': '',
                            'fply': '',
                            'mscan': mscan,
                            'mlmk': '',
                            'mcsv': '',
                            'rot_found' : rot_found

                        }
                         
                        lmk_folder = os.path.join(raw_files, 'Segmentaciones', caso, week, 'Lmks_Ricardo') 
                        if not os.path.exists(lmk_folder): lmk_folder = os.path.join(raw_files, 'Segmentaciones', caso, week, 'Lmks_Gerard') 
                        lmks_found =  False
                        if os.path.exists(lmk_folder):
                            flmk = os.path.join(lmk_folder, osid + '.fcsv')
                            flmk_modified = os.path.join(lmk_folder, osid + '_modified.fcsv')
                            fcsv = os.path.join(lmk_folder, osid + '.csv')
                            fcsv_modified = os.path.join(lmk_folder, osid + '_modified.csv')    
                            if os.path.exists(flmk) or os.path.exists(flmk_modified): lmks_found = True
                            if os.path.exists(flmk_modified): scan_dict['flmk_full'] = flmk_modified

                            if os.path.exists(flmk):
                                scan_dict['flmk'] = flmk
                                scan_dict['visibles'] = extract_list_lmks_fromfcsv(flmk)
                            
                            if os.path.exists(fcsv): scan_dict['_fcsv'] = fcsv
                            elif os.path.exists(fcsv_modified): scan_dict['_fcsv'] = fcsv
                            else: scan_dict['_fcsv'] = ''
                        scan_dict['lmks_found'] = lmks_found
                        if lmks_found:                         
                            mlmk = os.path.join(dataset, 'landmarks', 'fcsv', nsid + '.fcsv')
                            mcsv = os.path.join(dataset, 'landmarks', 'csv', nsid + '.csv')
                            scan_dict['mlmk'] = mlmk
                            scan_dict['mcsv'] = mcsv

                        ply_folder = os.path.join(raw_files, 'Segmentaciones', caso, week, 'PLY') 
                        plys_found = False
                        if os.path.exists(ply_folder):
                            fply = os.path.join(ply_folder, osid + '.ply')
                            if os.path.exists(fply):
                                scan_dict['fply'] = fply
                                plys_found = True
                        scan_dict['plys_found'] = plys_found

                        sinfo_df.append(scan_dict)
                caso_dict['num_scans']= scan_counter
                pinfo_df.append(caso_dict)

    sinfo_df = pd.DataFrame.from_records(sinfo_df)

    # Save patient and scan info DataFrames as CSV files
    pd.DataFrame.from_records(pinfo_df).to_csv(params.sys + params.root + 'pinfo_all.csv', index=False)
    sinfo_df.to_csv(params.sys + params.root + 'sinfo_all.csv', index=False)
    sinfo = sinfo_df[sinfo_df['rot_found'] & sinfo_df['lmks_found']]
    pd.DataFrame.from_records(sinfo).to_csv(params.sys + params.root + 'sinfo.csv', index=False)

    return sinfo_df
