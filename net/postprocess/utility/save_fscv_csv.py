import numpy as np
import pandas as pd

# Global row template for landmarks
gb_row = {
    '0': 0,  # ID (can be set to landmark name or a unique identifier)
    'x': 0,  # X-coordinate in physical units
    'y': 0,  # Y-coordinate in physical units
    'z': 0,  # Z-coordinate in physical units
    'ow': 0,  # Orientation w-component
    'ox': 0,  # Orientation x-component
    'oy': 0,  # Orientation y-component
    'oz': 1,  # Orientation z-component
    'vis': 1,  # Visibility flag
    'sel': 1,  # Selection flag
    'lock': 0,  # Lock flag
    'desc': '',  # Description
    'associatedNodeID': ''  # Associated node ID
}

# Mapping of landmark names to their IDs
landmark_to_id_table = {'exR': 0, 
                        'enR': 1, 
                        'n': 2, 
                        'enL': 3, 
                        'exL': 4, 
                        'acR': 5,
                        'aR': 6,
                        'prn': 7, 
                        'aL': 8,
                        'acL': 9,
                        'sn': 10,
                        'chR': 11,
                        'cphR': 12,
                        'ls': 13,
                        'cphL': 14,
                        'chL': 15, 
                        'li': 16,
                        'sl': 17, 
                        'pg': 18,}  # Example mapping, can be extended+


def create_df(coords, selected_lmks, spacing=1.0):
    """    
    Create a DataFrame containing information about a single landmark.

        coords (array-like): The coordinates of the landmark (typically as a NumPy array or list).
        selected_lmk (str): The name or identifier of the selected landmark.
        spacings (float or array-like, optional): Spacing values for converting voxel coordinates to physical units. Default is 1.0.

        pd.DataFrame: A DataFrame with a single row containing the landmark's coordinates (in physical units), metadata, and associated information.

    Notes:
        - The function expects global variables `gb_row` (a template row as a dict) and `landmark_to_id_table` (a mapping from landmark names to IDs) to be defined.
        - The DataFrame columns include: ['0', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel', 'lock', 'label', 'desc', 'associatedNodeID'].
    """
    # Define the columns for the DataFrame
    columns = ['0', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel', 'lock', 'label', 'desc', 'associatedNodeID']

    # Create a copy of the global row template
    row = gb_row.copy()

    for i, coord in enumerate(coords):
        # Convert coordinates to physical units using spacings
        row['x'], row['y'], row['z'] = coord * spacing

        # Set the label for the landmark
        row['label'] = selected_lmks[i]

        # Set the ID for the landmark (optional)
        row['0'] = landmark_to_id_table[selected_lmks[i]]

        # Use pd.concat instead of append
        if i == 0:
            landmarks_as_df = pd.DataFrame([row], columns=columns, index=[0])
        else:
            landmarks_as_df = pd.concat([landmarks_as_df, pd.DataFrame([row], columns=columns, index=[i])])

    return landmarks_as_df

def write_csv_file(out, landmarks_as_df):
    """
    Write the landmarks DataFrame to a CSV file.

    Parameters:
        out (str): Output file path (without extension).
        landmarks_as_df (pd.DataFrame): DataFrame containing landmark data.
    """
    # Define the output CSV file path
    csv_path = out + '.csv'

    # Save the DataFrame to a CSV file, dropping any 'Unnamed: 0' column
    landmarks_as_df.drop(columns=['Unnamed: 0'], errors='ignore').to_csv(csv_path, index=True)

def write_fcsv_file(out, landmarks_as_df):
    """
    Write the landmarks DataFrame to an FCSV file with predefined header lines.

    Parameters:
        out (str): Output file path (without extension).
        landmarks_as_df (pd.DataFrame): DataFrame containing landmark data.
    """
    # Predefined header lines for the FCSV file
    predefined_lines = [
        "# Markups fiducial file version = 5.2\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    ]

    # Define the output FCSV file path
    fcsv_path = out + '.fcsv'

    # Write the FCSV file
    with open(fcsv_path, 'w') as f:
        # Write the predefined header lines
        f.writelines(predefined_lines)

        # Write each row of the DataFrame as a comma-separated string
        for _, row in landmarks_as_df.iterrows():
            f.write(','.join(map(str, row.values)) + '\n')



def save_fscv_csv(out, coords, selected_lmks, spacing=1.0):  

    """
    Save landmark coordinates to both CSV and FCSV files.

    Parameters:
        out (str): Output file path (without extension).
        coords (array-like): Coordinates of the landmark.
        selected_lmk (str): Name or identifier of the selected landmark.
        spacing (float or array-like, optional): Spacing values for converting voxel coordinates to physical units. Default is 1.0.
    """
    # Create a DataFrame with the landmark data
    landmarks_as_df = create_df(coords, selected_lmks, spacing)

    # Write the DataFrame to a CSV file
    write_csv_file(out, landmarks_as_df)

    # Write the DataFrame to an FCSV file
    write_fcsv_file(out, landmarks_as_df)