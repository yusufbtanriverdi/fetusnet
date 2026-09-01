import pickle
import nrrd
import glob
import os
from tqdm import tqdm 

def create_template(path_to_sample):
    """ To save template header for your image. 
    This is to ensure you have correct numpy version as sometimes it might be a matter of conflict."""
    _, header = nrrd.read(path_to_sample)

    header["space dimension"] = 3
    header["space directions"] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    header["space origin"] = [0.0, 0.0, 0.0]
    header["space units"] = ["mm", "mm", "mm"]

    # optional: remove old/simple spacing if ImFusion complains
    # header.pop("spacings", None)
    # Save
    with open("templates/1.pkl", "wb") as f:
        pickle.dump(header, f)
    # Load
    with open("templates/1.pkl", "rb") as f:
        loaded_d = pickle.load(f)
    print(loaded_d)
    print(type(loaded_d))

def main(base="/media/yusuf/HDD 4TB/DATA/"):
    # Load template header
    with open("templates/1.pkl", "rb") as f:
        template_header = pickle.load(f)

    # Ensure output directory exists
    os.makedirs(os.path.join(base, "volumes2"), exist_ok=True)

    # Find all .nrrd files in volumes/
    nrrd_files = glob.glob(os.path.join(base, "volumes", "*.nrrd"))
    
    for path in tqdm(nrrd_files):
        # Read data and original header
        data, _ = nrrd.read(path)

        # Use template header (copy to avoid accidental mutation sharing)
        header = template_header.copy()

        # Optional: ensure consistency with data shape if needed
        # header["sizes"] = list(data.shape)

        # Save to volumes2 with same filename
        out_path = os.path.join(base, "volumes2", os.path.basename(path))
        nrrd.write(out_path, data, header)

        # Optional: remove print if tqdm progress is enough
        # print(f"Saved: {out_path}")

if __name__ == "__main__":
    # create_template("/media/yusuf/HDD 4TB/DATA/volumes/10012001.nrrd")
    main(base='/home/yusuf/Source/fetusnet/runs/2026-05-19/kendall6/eval')