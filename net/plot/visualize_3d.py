import nrrd
import numpy as np
import pyvista as pv
import os
import pandas as pd

def numpy_sigmoid(arr):
    """
    Applies the sigmoid function element-wise to a 3D NumPy array.

    Parameters:
        arr (np.ndarray): A 3D NumPy array.

    Returns:
        np.ndarray: The result of applying the sigmoid function.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if arr.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array.")
    
    return 1 / (1 + np.exp(-arr))

def load_landmarks_as_point_cloud(scan_name: str):
    """
    Load all landmark coordinates from a CSV and return a PyVista point cloud.

    Parameters:
        scan_name (str): The name of the scan (used to locate the CSV).

    Returns:
        pv.PolyData: A PyVista point cloud of all landmarks.
        list: A list of landmark labels in the same order as the points.
    """
    landmark_path = f'tmps/{scan_name}.csv'
    landmark_df = pd.read_csv(landmark_path)

    # Extract coordinates and labels
    points = landmark_df[['x', 'y', 'z']].to_numpy()
    labels = landmark_df['label'].tolist()

    # Create a PyVista PolyData object with points
    point_cloud = pv.PolyData(points)

    # Optionally attach labels as point data (can be used for labeling)
    point_cloud.point_data['labels'] = np.array(labels)

    return point_cloud, labels

def project_landmarks_to_surface(point_cloud, mesh):
    projected = []
    for i, point in enumerate(point_cloud.points):
        # print(i)
        # print(point)
        # print(mesh.find_closest_point(point))
        # print(mesh.points[mesh.find_closest_point(point)])
        projected.append(mesh.points[mesh.find_closest_point(point)])
    return pv.PolyData(np.array(projected))

def visualize_heatmaps(scan_name: str, landmark: str):
    # Construct file paths
    pred_nrrd_path = f'tmps/{scan_name}_{landmark}_19.nrrd'
    gt_nrrd_path   = f'tmps/{scan_name}_{landmark}_gen.nrrd'
    mesh_path      = f'tmps/{scan_name}_SEG2.stl'
    # Load the landmark file containing coordinates
    point_cloud, labels = load_landmarks_as_point_cloud(scan_name)
    # Read NRRD files
    heatmap_pred, header_pred = nrrd.read(pred_nrrd_path)
    heatmap_gt, header_gt = nrrd.read(gt_nrrd_path)
    
    # Sigmoid 
    # heatmap_pred = numpy_sigmoid(heatmap_pred)

    # Load volume from .nrrd file
    # volume_path = f"tmps/{scan_name}.nrrd"  # Replace with your path
    # volume, header = nrrd.read(volume_path)

    # Read mesh
    mesh = pv.read(mesh_path)

    # Extract spacing and define origin
    spacing_pred = header_pred.get('spacings', [1, 1, 1])  # fallback to 1s
    origin = [0, 0, 0]

    # Convert prediction heatmap to UniformGrid and sample onto mesh
    grid_pred = pv.ImageData()
    grid_pred.dimensions = np.array(heatmap_pred.shape)
    grid_pred.origin = origin
    grid_pred.spacing = spacing_pred
    grid_pred.point_data['heatmap'] = heatmap_pred.flatten(order='F')
    mesh_with_heatmap_pred = mesh.sample(grid_pred)

    # Repeat for ground truth
    spacing_gt = header_gt.get('spacings', [1, 1, 1])
    grid_gt = pv.ImageData()
    grid_gt.dimensions = np.array(heatmap_gt.shape)
    grid_gt.origin = origin
    grid_gt.spacing = spacing_gt
    grid_gt.point_data['heatmap'] = heatmap_gt.flatten(order='F')
    mesh_with_heatmap_gt = mesh.sample(grid_gt)

    # Find max heatmap values (landmark positions)
    max_idx_pred = np.argmax(mesh_with_heatmap_pred['heatmap'])
    max_point_pred = mesh_with_heatmap_pred.points[max_idx_pred].reshape(1, 3)

    # # max_point_gt = mesh_with_heatmap_gt.points[max_idx_gt].reshape(1, 3)
    # volume[volume <= 50] = 0
    # pl = pv.Plotter()
    # pl.add_volume(volume)
    # pl.add_points(point_cloud, color='red', point_size=10)
    # pl.show()

    # Plotting
    pl = pv.Plotter(shape=(1, 2))
    point_size = 10
    pl.subplot(0, 0)
    pl.add_title('Prediction')
    pl.add_mesh(mesh_with_heatmap_pred, scalars='heatmap', cmap='hot', show_scalar_bar=True)

    point_cloud_pred = project_landmarks_to_surface(point_cloud, mesh_with_heatmap_pred)
    labels = np.array(labels)
    pl.add_points(point_cloud_pred.points[labels!=landmark], color='white', point_size=point_size, render_points_as_spheres=True)
    pl.add_points(point_cloud_pred.points[labels==landmark], color='lightgreen', point_size=point_size, render_points_as_spheres=True)
    pl.add_points(max_point_pred, color='blue', point_size=point_size, render_points_as_spheres=True)
    # pl.add_point_labels(max_point_pred, [landmark], point_size=0, font_size=5,)
    # Add labels
    for i, label in enumerate(labels):
        # pl.add_point_labels(point_cloud.points[i:i+1], [label], font_size=5, point_size=0)
        pass 

    pl.subplot(0, 1)
    pl.add_title('Ground Truth')
    
    pl.add_mesh(mesh_with_heatmap_gt, scalars='heatmap', cmap='hot', show_scalar_bar=True)
    # pl.add_points(max_point_gt, color='black', point_size=20)
    # point_cloud_gt = project_landmarks_to_surface(point_cloud, mesh_with_heatmap_gt)
    point_cloud_gt = point_cloud
    pl.add_points(point_cloud_gt.points[labels!=landmark], color='white', point_size=point_size, render_points_as_spheres=True)
    pl.add_points(point_cloud_gt.points[labels==landmark], color='lightgreen', point_size=point_size, render_points_as_spheres=True)
    pl.add_points(max_point_pred, color='blue', point_size=point_size, render_points_as_spheres=True)
    # pl.add_point_labels(max_point_pred, [landmark], point_size=0, font_size=5,)
    for i, label in enumerate(labels):
        # pl.add_point_labels(point_cloud.points[i:i+1], [label], font_size=5, point_size=0)
        pass


    # 🔗 Synchronize views (camera linked)
    pl.link_views()
    # Export to HTML (interactive)
    pl.export_html(f"{scan_name}_{landmark}.html")  # or backend='pythreejs'
    pl.show()

    print(max_point_pred, point_cloud_gt.points[labels==landmark])


    p1 = np.array(point_cloud_gt.points[labels==landmark])
    p2 = np.array(max_point_pred)

    distance = np.linalg.norm(p1 - p2)
    print(f"Distance: {distance:.6f}")

# Example usage
if __name__ == "__main__":
    scan_name = "37-26s-01"   # Replace with your scan name
    landmark = "enL"         # Replace with your landmark name
    visualize_heatmaps(scan_name, landmark)
