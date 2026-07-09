import pickle
import nrrd

def main():
    """ To save template header for your image. 
    This is to ensure you have correct numpy version as sometimes it might be a matter of conflict."""
    _, header = nrrd.read('path/to/your/sample/nrrd/image')
    # Save
    with open("templates/1.pkl", "wb") as f:
        pickle.dump(header, f)
    # Load
    with open("templates/1.pkl", "rb") as f:
        loaded_d = pickle.load(f)

    print(loaded_d)
    print(type(loaded_d))


if __name__ == "__main__":
    main()