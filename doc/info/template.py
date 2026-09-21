import numpy as np
import nrrd
import glob
import os
from tqdm import tqdm 
from dataset.utility.rotation import extract_image

def create_template(spacings):
    """ To save template header for your image. 
    This is to ensure you have correct numpy version as sometimes it might be a matter of conflict."""
    header = {
        "space": "right-anterior-superior",
        "space directions": np.array([
            [spacings[0], 0.0, 0.0],
            [0.0, spacings[1], 0.0],
            [0.0, 0.0, spacings[2]],
        ]),
        "kinds": ["domain", "domain", "domain"],
        "space units": ["mm", "mm", "mm"],
        "space origin": np.array([0.0, 0.0, 0.0]),
        "encoding": "raw",
    }
    # with open("templates/2.pkl", "wb") as f:
    #     pickle.dump(header, f)

    return header

def perform_header_matching(new_data_dir, old_data_dir):
    # Ensure output directory exists
    os.makedirs(new_data_dir, exist_ok=True)
    # Find all .nrrd files in volumes/
    nrrd_files = glob.glob(os.path.join(old_data_dir,"*.nrrd"))
    
    for name in tqdm(nrrd_files):
        data, header = extract_image(os.path.join(old_data_dir, name))
        try:
            p = header.get('spacings')[:3]
        except Exception as e:
            p = np.array([header['space directions'][0, 0],
                          header['space directions'][1, 1],
                          header['space directions'][2, 2]])
        template = create_template(spacings=p)
        nrrd.write(os.path.join(new_data_dir, name), data, header=template, index_order="F")
