import numpy as np
import pandas as pd

# Global row template for landmarks
gb_row = {
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
landmark_to_id_table = {'prn': 7}

def create_df(spacings, coords, selected_lmks=['prn']):
    """
    Create a DataFrame for landmarks with their coordinates and metadata.

    Parameters:
        spacings (array-like): Spacing values for converting voxel to physical units.
        coords (dict): Dictionary containing landmark coordinates.
        selected_lmks (list): List of selected landmarks to include.

    Returns:
        pd.DataFrame: DataFrame containing landmark data.
    """
    # Define the columns for the DataFrame
    columns = ['0', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel', 'lock', 'label', 'desc', 'associatedNodeID']
    landmarks_as_df = pd.DataFrame(columns=columns)

    for landmark in selected_lmks:
        # Create a copy of the global row template
        row = gb_row.copy()

        # Convert coordinates to physical units using spacings
        row['x'], row['y'], row['z'] = coords[landmark] * spacings

        # Set the label for the landmark
        row['label'] = landmark

        # Set the ID for the landmark (optional)
        row['0'] = landmark_to_id_table[landmark]

        # Append the row to the DataFrame
        landmarks_as_df = landmarks_as_df.append(row, ignore_index=True)

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