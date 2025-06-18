import nrrd
import numpy as np
import pyvista as pv

heatmap, heatmap_header = nrrd.read('notebooks/tmps/1-29s-06_acL_pred.nrrd')
heatmap2, heatmap_header = nrrd.read('notebooks/tmps/1-29s-06_acL_gen.nrrd')
# heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
mesh = pv.read('notebooks/tmps/1_29s_06-models/1_29s_06.stl')
## WHAT TO DO HERE TO RESAMPLE HEATMAP COORDINATES TO MESH ?? ##
print(heatmap_header)
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
max_point1 = mesh_with_heatmap.points[max_idx1]

# For mesh_with_heatmap2 (Ground Truth)
heatmap2 = mesh_with_heatmap2['heatmap']
max_idx2 = np.argmax(heatmap2)
max_point2 = mesh_with_heatmap2.points[max_idx2]

max_point1 = np.asarray(max_point1).reshape(1, 3)
max_point2 = np.asarray(max_point2).reshape(1, 3)

pl = pv.Plotter(shape=(1, 2))
# Prediction
pl.subplot(0, 0)
pl.add_title('Prediction')
pl.add_mesh(mesh_with_heatmap, scalars='heatmap', cmap='jet', show_scalar_bar=True)
pl.add_points(max_point1, color='white', point_size=10)
pl.add_point_labels(max_point1, ['acL'], point_size=0, font_size=10)

# Ground Truth
pl.subplot(0, 1)
pl.add_title('Ground Truth')
pl.add_mesh(mesh_with_heatmap2, scalars='heatmap', cmap='jet', show_scalar_bar=True)
pl.add_points(max_point2, color='white', point_size=10)
pl.add_point_labels(max_point2, ['acL'], point_size=0, font_size=10,)

pl.show()

# mesh.point_data['heatmap'] = heatmap_values

# plotter = pv.Plotter()
# plotter.add_mesh(mesh, scalars='heatmap', cmap='jet', show_scalar_bar=True)
# plotter.show()
