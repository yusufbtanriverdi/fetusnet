import nrrd
import numpy as np
import pyvista as pv
import pandas as pd

heatmap, heatmap_header = nrrd.read('notebooks/tmps/23-20s-02_enR_predS.nrrd')
heatmap2, heatmap_header = nrrd.read('notebooks/tmps/23-20s-02_enR_gen.nrrd')
# heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
mesh = pv.read('notebooks/tmps/Output Volume_4-models/23-20s-02.stl')
## WHAT TO DO HERE TO RESAMPLE HEATMAP COORDINATES TO MESH ?? ##
#print(heatmap_header)
# Convert heatmap to PyVista UniformGrid
spacing = heatmap_header['spacings']  # Typically a (3,3) array
origin = [0, 0, 0]       # Typically a (3,) array
grid = pv.ImageData()
grid.dimensions = np.array(heatmap.shape) 
grid.origin = origin
grid.spacing = spacing
grid.point_data['heatmap'] = heatmap.flatten(order='F')  # Fortran order for VTK compatibility
# Sample heatmap values onto mesh points
mesh_with_heatmap = mesh.sample(grid)

spacing = heatmap_header['spacings']  # Typically a (3,3) array
origin = [0, 0, 0]       # Typically a (3,) array
grid = pv.ImageData()
grid.dimensions = np.array(heatmap2.shape) 
grid.origin = origin
grid.spacing = spacing
grid.point_data['heatmap'] = heatmap2.flatten(order='F')  # Fortran order for VTK compatibility
# Sample heatmap values onto mesh points
mesh_with_heatmap2 = mesh.sample(grid)

# For mesh_with_heatmap (Prediction)
heatmap1 = mesh_with_heatmap['heatmap']
max_idx1 = np.argmax(heatmap1)
peak1 = np.array(np.nonzero(heatmap == heatmap.max()))


# Load the landmark file containing coordinates
landmark_path = 'notebooks/tmps/23-20s-02.csv'
landmark_df = pd.read_csv(landmark_path)
landmark_row = landmark_df[landmark_df['label'] == 'enR']
coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()

max_point1 = np.asarray(peak1).reshape(1, 3).astype(np.float32)
max_point2 = np.asarray(coord).reshape(1, 3).astype(np.float32)
print(peak1, coord, max_point1, max_point2)


pl = pv.Plotter(shape=(1, 2))
# Prediction
# Extract the edges of the grid to make a boxy grid look
outline = grid.outline()
pl.subplot(0, 0)
pl.add_mesh(outline, color='black', line_width=0.5, opacity=0.4)  # on prediction
pl.add_title('Prediction')
pl.add_mesh(mesh_with_heatmap, scalars='heatmap', cmap='jet', show_scalar_bar=True, opacity=0.99)
pl.add_points(max_point1, color='black', point_size=10)
pl.add_point_labels(max_point1, ['enR'], point_size=0, font_size=10)

# Ground Truth
pl.subplot(0, 1)
pl.add_mesh(outline, color='black', line_width=0.5, opacity=0.4)  # on prediction
pl.add_title('Ground Truth')
pl.add_mesh(mesh_with_heatmap2, scalars='heatmap', cmap='jet', show_scalar_bar=True, opacity=0.99)
pl.add_points(max_point2, color='white', point_size=10)
pl.add_point_labels(max_point2, ['enR'], point_size=0, font_size=10,)
# pl.show()


pl = pv.Plotter()
# Prediction
# Extract the edges of the grid to make a boxy grid look
# outline = grid.outline()
# pl.add_mesh(outline, color='black', line_width=0.5, opacity=0.4)  # on prediction
pl.add_title('Quantization Error')

# Set landmark and crop box size (in world units)
center = max_point1[0]  # Use max_point2[0] for ground truth
crop_size = np.array(spacing) * 5  # Half-size of crop region: 10 voxels

# Create bounding box around the center
bounds = [
    center[0] - crop_size[0], center[0] + crop_size[0],
    center[1] - crop_size[1], center[1] + crop_size[1],
    center[2] - crop_size[2], center[2] + crop_size[2]
]
# Clip mesh with box
cropped_mesh_pred = mesh_with_heatmap.clip_box(bounds, invert=False)
# pl.add_mesh(cropped_mesh_pred, scalars='heatmap', cmap='jet', show_scalar_bar=True, opacity=0.99)
pl.add_points(max_point1, color='pink', point_size=15, render_points_as_spheres=True)
pl.add_points(max_point2, color='blue', point_size=15, render_points_as_spheres=True)

# pl.add_point_labels(max_point1, [f'Prediction'], point_size=0, font_size=5)
# pl.add_point_labels(max_point2, [f'Ground Truth'], point_size=0, font_size=5)


# Convert landmark to voxel index
center_idx = np.round((max_point2[0] - origin) / spacing).astype(int)
cx, cy, cz = center_idx
crop_size = 2  # Half width of 10x10x10

for i in range(cx - crop_size, cx + crop_size):
    for j in range(cy - crop_size, cy + crop_size):
        for k in range(cz - crop_size, cz + crop_size):
            # Compute voxel min corner
            voxel_origin = np.array([i, j, k]) * spacing + origin

            # Build cube using 8 corners (wireframe only)
            cube = pv.Cube(
                center=voxel_origin + spacing / 2,
                x_length=spacing[0],
                y_length=spacing[1],
                z_length=spacing[2]
            )

            pl.add_mesh(cube, style='wireframe', color='black', line_width=0.6, opacity=1.0)

rng = np.random.default_rng(seed=0)
cent = max_point1
direction = max_point2-max_point1
mag = np.linalg.norm(max_point1 - max_point2)
print(cent, direction, mag)
pl.add_arrows(cent, direction, mag=0.5, color='magenta', opacity=0.5, point_size=0)
# pl.add_arrows(max_point1, max_point2, mag=0.5)
pl.show()

# mesh.point_data['heatmap'] = heatmap_values

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, scalars='heatmap', cmap='jet', show_scalar_bar=True)
# plotter.show()
