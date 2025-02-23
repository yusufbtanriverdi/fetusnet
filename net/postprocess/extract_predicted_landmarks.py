import numpy as np

def find_peak_coord(output):
    # TODO: What is output shape?

    coord = np.unravel_index(np.argmax(output[0]), output[0].shape)
    return coord