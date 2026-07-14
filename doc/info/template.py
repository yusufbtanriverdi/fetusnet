import pickle
import nrrd

def main(path_to_sample):
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


if __name__ == "__main__":
    main("/media/yusuf/HDD 4TB/DATA/volumes/10012001.nrrd")