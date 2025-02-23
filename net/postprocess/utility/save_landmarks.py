import numpy as np
import pandas as pd

# Global row
gb_row = {'ow': 0, 'ox': 0, 'oy': 0, 'oz': 1, 'vis': 1, 'sel': 1, 'lock': 0, 'desc': '', 'associatedNodeID': ''}
landmark_to_id_table = {'prn': 7}

def create_df(spacings, coords, selected_lmks = ['prn']):
    # coords --> dictionary with either label or id 
    columns = ['0','x','y','z','ow','ox','oy','oz','vis','sel','lock','label','desc','associatedNodeID']
    landmarks_as_df = pd.DataFrame(columns=columns)

    for landmark in selected_lmks:
        # Convert coord to physical units
        row = gb_row.copy()
        row['x'], row['y'], row['z'] = coords[landmark] * spacings
        row['label'] = landmark

        i = landmark_to_id_table[landmark]
        row['0'] =  i # Optional

    return landmarks_as_df

def write_csv_file(out, landmarks_as_df):
    # Save CSV file
    csv_path = out + '.csv'
    landmarks_as_df.drop(columns=['Unnamed: 0']).to_csv(csv_path, index=True)

def write_fcsv_file(out, landmarks_as_df):
    # Define the predefined lines for FCSV format
    predefined_lines = [
        "# Markups fiducial file version = 5.2\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    ]

    # Save FCSV file with predefined lines
    fcsv_path = out + '.fcsv'
    with open(fcsv_path, 'w') as f:
        # Write predefined lines
        f.writelines(predefined_lines)
        # Write each row as a comma-separated string
        for _, row in landmarks_as_df.iterrows():
            f.write(','.join(map(str, row.values)) + '\n')