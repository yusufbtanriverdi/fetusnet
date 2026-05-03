import os
import nrrd
import numpy as np
import pyvista as pv
import pandas as pd
import ast

def numpy_sigmoid(arr: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function element-wise to a 3D array."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if arr.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array.")
    return 1 / (1 + np.exp(-arr))


def load_landmarks_as_point_cloud(csv_path: str):
    """Load lmk coordinates from CSV → PyVista point cloud + labels."""
    df = pd.read_csv(csv_path)
    points = df[["x", "y", "z"]].to_numpy()
    labels = df["label"].to_numpy()
    cloud = pv.PolyData(points)
    cloud.point_data["labels"] = labels
    return cloud, labels


def project_landmarks_to_surface(point_cloud: pv.PolyData, mesh: pv.PolyData) -> pv.PolyData:
    """Project landmarks onto the closest surface points."""
    projected = [mesh.points[mesh.find_closest_point(p)] for p in point_cloud.points]
    return pv.PolyData(np.array(projected))


def load_heatmap_as_grid(nrrd_path: str) -> pv.ImageData:
    """Load a .nrrd heatmap file and convert to PyVista ImageData."""
    data, header = nrrd.read(nrrd_path)
    spacing = header.get("spacings", [1, 1, 1])  # fallback if missing
    grid = pv.ImageData()
    grid.dimensions = np.array(data.shape)
    grid.origin = [0, 0, 0]
    grid.spacing = spacing
    grid.point_data["heatmap"] = data.flatten(order="F")
    return grid



def perform_plot_3d(df, experiment_dir, params):
    """ Visualize predicted vs ground truth heatmaps from a DataFrame row.

    Expected columns in row:
        - 'nsid': str
        - 'lmk': str
        - 'mcsv': str
        - 'pred_nrrd': str
        - 'gt_nrrd': str
        - 'stl': str

    Args:
        row (pd.Series): Row containing paths and metadata.
        out_dir (str): Directory to save HTML exports.

    Returns:
        float: Euclidean distance between predicted max and ground-truth lmk.

    """
    df['lmks_array'] = df['visibles'].apply(ast.literal_eval)
    row = df.loc[0] # for now

    nsid = row["nsid"]
    lmk = row["lmks_array"][0] # visibles? then loop 
    
    # --- Load inputs ---
    point_cloud, labels = load_landmarks_as_point_cloud(os.path.join(params.root, row['mcsv']))
    grid_pred = load_heatmap_as_grid(os.path.join(experiment_dir, 'evals', nsid+lmk+'.nrrd')) 
    grid_gt = load_heatmap_as_grid(os.path.join(experiment_dir, 'evals', nsid+lmk+'_target.nrrd')) 

    if row['plys_found']:
        mesh = pv.read(row["fply"].replace(".ply", "_rotated.ply"))

    # --- Sample heatmaps on surface ---
    mesh_pred = mesh.sample(grid_pred)
    mesh_gt = mesh.sample(grid_gt)

    # --- Find max prediction ---
    max_idx = np.argmax(mesh_pred["heatmap"])
    max_point_pred = mesh_pred.points[max_idx].reshape(1, 3)

    # --- PyVista plot ---
    pl = pv.Plotter(shape=(1, 2))
    point_size = 10

    # Prediction subplot
    pl.subplot(0, 0)
    pl.add_title("Prediction")
    pl.add_mesh(mesh_pred, scalars="heatmap", cmap="hot", show_scalar_bar=True)
    cloud_proj = project_landmarks_to_surface(point_cloud, mesh_pred)
    pl.add_points(cloud_proj.points[labels != lmk], color="white", point_size=point_size, render_points_as_spheres=True)
    pl.add_points(cloud_proj.points[labels == lmk], color="lightgreen", point_size=point_size, render_points_as_spheres=True)
    pl.add_points(max_point_pred, color="blue", point_size=point_size, render_points_as_spheres=True)

    # Ground truth subplot
    pl.subplot(0, 1)
    pl.add_title("Ground Truth")
    pl.add_mesh(mesh_gt, scalars="heatmap", cmap="hot", show_scalar_bar=True)
    pl.add_points(point_cloud.points[labels != lmk], color="white", point_size=point_size, render_points_as_spheres=True)
    pl.add_points(point_cloud.points[labels == lmk], color="lightgreen", point_size=point_size, render_points_as_spheres=True)
    pl.add_points(max_point_pred, color="blue", point_size=point_size, render_points_as_spheres=True)

    pl.link_views()
    out_html = os.path.join(os.path.join(experiment_dir, 'evals'), f"{nsid}_{lmk}.html")
    pl.export_html(out_html)
    pl.show()

    # --- Distance calculation ---
    gt_point = np.array(point_cloud.points[labels == lmk])
    pred_point = np.array(max_point_pred)
    distance = np.linalg.norm(gt_point - pred_point)
    print(f"[{nsid} - {lmk}] Distance: {distance:.4f}")
    return distance