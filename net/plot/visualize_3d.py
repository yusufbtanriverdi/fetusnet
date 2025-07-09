import nrrd
import numpy as np
import pyvista as pv
import os

def visualize_heatmaps(scan_name: str, landmark: str):
    # Construct file paths
    pred_nrrd_path = f'tmps/{scan_name}_{landmark}_pred.nrrd'
    gt_nrrd_path   = f'tmps/{scan_name}_{landmark}_gen.nrrd'
    mesh_path      = f'tmps/{scan_name}_SEG.stl'

    # Read NRRD files
    heatmap_pred, header_pred = nrrd.read(pred_nrrd_path)
    heatmap_gt, header_gt = nrrd.read(gt_nrrd_path)

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

    # max_point_gt = mesh_with_heatmap_gt.points[max_idx_gt].reshape(1, 3)

    # Plotting
    pl = pv.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_title('Prediction')
    pl.add_mesh(mesh_with_heatmap_pred, scalars='heatmap', cmap='jet', show_scalar_bar=True)
    pl.add_points(max_point_pred, color='white', point_size=20)
    # pl.add_points(max_point_gt, color='black', point_size=20)
    pl.add_point_labels(max_point_pred, [landmark], point_size=20, font_size=20)

    pl.subplot(0, 1)
    pl.add_title('Ground Truth')
    pl.add_mesh(mesh_with_heatmap_gt, scalars='heatmap', cmap='jet', show_scalar_bar=True)
    pl.add_points(max_point_pred, color='white', point_size=20)
    # pl.add_points(max_point_gt, color='black', point_size=20)
    # pl.add_point_labels(max_point_gt, [landmark], point_size=20, font_size=20)

    pl.show()


# Example usage
if __name__ == "__main__":
    scan_name = "FLA094_11"   # Replace with your scan name
    landmark = "sn"         # Replace with your landmark name
    visualize_heatmaps(scan_name, landmark)
