# import io
# import os

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import imageio.v2 as imageio
# from cairosvg import svg2png
# import torch
# from tqdm import tqdm
# from matplotlib.ticker import NullFormatter
# from net.dataset.target.gaussian_heatmap import create_gaussian_heatmap

# sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
# sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

# def _softmax_numpy(volume):
#     exp_volume = np.exp(volume - np.max(volume))
#     return exp_volume / np.sum(exp_volume)

# def create_disc_figure(test_dl, w, save_dir='figures/'):
#     """
#     Create a figure showing the optimal transport distance on a moving disc in a 50x50 grid.

#     Args:
#         test_dl (DataLoader): DataLoader containing the test dataset.
#         w (float): Weight for the distance penalty.
#         save_dir (str): Directory to save the generated figure.
#     """ 
#     batch = next(iter(test_dl)) # Get the first batch from the DataLoader
#     # Move input data and targets to the specified device
#     # images = batch['image']['data']
#     targets = batch['target'].cpu()
#     target_heatmap = targets[0, 0, 0].cpu().numpy() # {!} Assuming batch size of 1, using the first landmark
#     distance_map = targets[0, 0, 1].cpu().numpy() # {!} Assuming batch size of 1   
#     # target_coords = batch['coords'][0][0].cpu().numpy().ravel()[:3].astype(int)  # Get the voxel coordinates of the target landmark, assuming batch size of 1
#     # print("Target coordinates (voxel): ", target_coords
#     volume = torch.zeros((128, 128, 128), dtype=torch.float32)  # Create an empty volume to generate the heatmap and distance map for testing
#     target_coords = torch.tensor([30, 30, 30], dtype=torch.float32)  # Center of the grid (0, 0, 0) in normalized coordinates
#     # target_heatmap, distance_map = create_gaussian_heatmap(target_coords, volume, alpha=3, clip=False, mask=False)
#     # target_heatmap = target_heatmap.cpu().numpy()
#     # distance_map = distance_map.cpu().numpy()
#     target_coords = target_coords.numpy().astype(int)
#     print(target_heatmap.shape, distance_map.shape, target_coords)

#     D, H, W = target_heatmap.shape
#     max_shift = np.array([D, H, W], dtype=float) * target_coords/np.array([D, H, W], dtype=float)  # Maximum shift in each direction (half the size of the grid)
#     ts = [target_coords] + [target_coords + max_shift * t for t in np.linspace(0, 1, 128)]  # Create a series of shifted coordinates for testing
#     # Compute L2 distances and Wasserstein
#     Images = [target_heatmap]
#     L2Dists = [0.0]
#     DistPenalty = [0.0]
#     SoftmaxCE = [0.0]
#     for i, t in enumerate(ts):
#         shift_vox = np.round(t - target_coords).astype(int)
#         I = np.roll(target_heatmap, shift_vox, axis=(0, 1, 2))
#         if i > 0:
#             Images.append(I)
#             L2Dists.append(np.sum((I - target_heatmap)**2))
#             DistPenalty.append(np.sum(_softmax_numpy(I) * distance_map ** w))
#             SoftmaxCE.append(np.sum(- _softmax_numpy(target_heatmap) * np.log(_softmax_numpy(I))))

#     L2Dists = np.array(L2Dists)
#     DistPenalty = np.array(DistPenalty)
#     SoftmaxCE = np.array(SoftmaxCE)

#     L2Dists /= np.max(L2Dists)
#     DistPenalty /= np.max(DistPenalty)
#     SoftmaxCE /= np.max(SoftmaxCE)
    
#     print("L2Dists: ", L2Dists)
#     print("DistPenalty: ", DistPenalty)
#     print("SoftmaxCE: ", SoftmaxCE)
    
#     fig = plt.figure(figsize=(15, 10))
#     I0 = Images[0]
#     displacements = np.sqrt(2)*(ts - ts[0])[..., 0]  # Calculate the displacement in voxel units (assuming isotropic spacing for simplicity)
#     # print("DistPenalty: ", DistPenalty[1:])
#     for i, I in tqdm(enumerate(Images), total=len(Images)):
#         I0_x = I0[target_coords[0], :, :]
#         I_x = I[:, :, target_coords[0]]
#         Dx = np.concatenate((I0_x[:, :, None], I_x[:, :, None], 0*I_x[:, :, None]), 2)
#         Dx = Dx*255/np.max(I0)
#         Dx = np.array(Dx, dtype=np.uint8)
        
#         I0_y = I0[:, target_coords[1], :]
#         I_y = I[:, :, target_coords[1]]
#         Dy = np.concatenate((I0_y[:, :, None], I_y[:, :, None], 0*I_y[:, :, None]), 2)
#         Dy = Dy*255/np.max(I0)
#         Dy = np.array(Dy, dtype=np.uint8)

#         I0_z = I0[:, :, target_coords[2]]
#         I_z = I[:, :, target_coords[2]]
#         Dz = np.concatenate((I0_z[:, :, None], I_z[:, :, None], 0*I_z[:, :, None]), 2)
#         Dz = Dz*255/np.max(I0)
#         Dz = np.array(Dz, dtype=np.uint8)
        
#         plt.clf()
#         ax1img = plt.subplot(231)
#         ax1img.imshow(Dx)
#         ax2img = plt.subplot(232)
#         ax2img.imshow(Dy)
#         ax3img = plt.subplot(233)
#         ax3img.imshow(Dz)

#         ax4 = plt.subplot(234)
#         ax4.plot(displacements, L2Dists, 'r-')
#         ax4.set_yscale('log')
#         ax4.fill_between(displacements[:i+1], L2Dists[:i+1], color='r', alpha=0.3)
#         ax4.set_xlabel("Displacements")
#         ax4.set_title("L2 Distance") 

#         ax5 = plt.subplot(235)
#         ax5.plot(displacements, DistPenalty, 'b-')
#         ax5.plot(displacements, SoftmaxCE, 'k--', alpha=0.3)
#         ax5.set_yscale('log', base=10)
#         ax5.fill_between(displacements[:i+1], DistPenalty[:i+1], color='b', alpha=0.3)
#         ax5.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.1)
#         ax5.set_xlabel("Displacements")
#         ax5.set_title("Distance Penalty")

#         ax6 = plt.subplot(236)
#         ax6.plot(displacements, SoftmaxCE, 'k-')
#         ax6.set_yscale('log', base=10)
#         ax6.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.3)
#         ax6.set_xlabel("Displacements")
#         ax6.set_title("Softmax CE")
#         for ax, values in zip([ax4, ax5, ax6], [L2Dists, DistPenalty, SoftmaxCE]):
#             ymax = np.max(values)
#             ymin = np.min(np.array(values)[np.array(values) > 0])
#             ax.tick_params(axis='y', which='both', labelleft=False, labeltop=True)
#             ax.set_yticks([ymin, ymax])
#             ax.yaxis.set_major_formatter(NullFormatter())
#             ax.yaxis.set_minor_formatter(NullFormatter())

#         # plt.suptitle(f"NSID: {nsid}, Visible: {visibles}")  
#         os.makedirs("tmp", exist_ok=True)
#         fig.savefig("tmp/%i.svg"%i, format='svg')
#         # plt.show()

#     # Create a GIF from the saved SVG frames
#     os.makedirs(save_dir, exist_ok=True)
#     svg_files = [f"tmp/{i}.svg" for i in range(len(Images))]
#     frames = []
#     for svg_path in svg_files:
#         png_bytes = svg2png(url=svg_path)
#         frames.append(imageio.imread(io.BytesIO(png_bytes)))

#     gif_path = os.path.join(save_dir, f"disc_concept_w{w:.2f}.gif")
#     imageio.mimsave(gif_path, frames, duration=0.1)
#     print(f"Saved GIF: {gif_path}")

import io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from tqdm import tqdm
from matplotlib.ticker import NullFormatter

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

def _softmax_numpy(volume):
    exp_volume = np.exp(volume - np.max(volume))
    return exp_volume / np.sum(exp_volume)

def generate_dummy_data(shape=(128, 128, 128), center=(16, 16, 16), sigma=6.):
    """Generates a 3D Gaussian heatmap and a Euclidean distance map."""
    z, y, x = np.indices(shape)
    dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    distance_map = np.sqrt(dist_sq)
    return heatmap.astype(np.float32), distance_map.astype(np.float32)

def safe_normalize(arr):
    """Normalizes an array to 0-1 safely to avoid division by zero."""
    max_val = np.max(arr)
    return arr / max_val if max_val > 0 else arr

def create_disc_figure(w=2, save_dir='figures/', num_steps=128):
    """
    Create a figure showing the optimal transport distance on a moving disc.
    """
    volume_shape = (64, 40, 64)
    start_coords = np.array([16, 10, 16])    # Top-Left-Front
    end_coords = np.array([50, 20, 50])   # Bottom-Right-Back
    
    print(f"Generating dummy data starting at {start_coords}...")
    target_heatmap, distance_map = generate_dummy_data(shape=volume_shape, center=start_coords)
    
    # Create a linear trajectory from start_coords to end_coords
    ts = [start_coords + (end_coords - start_coords) * t for t in np.linspace(0, 1, num_steps)]
    
    Images = [target_heatmap]
    L2Dists = [0.0]
    DistPenalty = [0.0]
    SoftmaxCE = [0.0]
    
    print("Calculating displacements and metrics...")
    for i, t in enumerate(ts):
        shift_vox = np.round(t - start_coords).astype(int)
        
        # np.roll wraps around, which is fine since we stay within the 128 bounds
        I = np.roll(target_heatmap, shift_vox, axis=(0, 1, 2))
        
        if i > 0:
            Images.append(I)
            L2Dists.append(np.sum((I - target_heatmap)**2))
            DistPenalty.append(np.sum(_softmax_numpy(I) * (distance_map ** 0.5)))
            
            # Add a tiny epsilon (1e-12) to avoid np.log(0) resulting in -inf
            pred_softmax = _softmax_numpy(I)
            target_softmax = _softmax_numpy(target_heatmap)
            ce = np.sum(-target_softmax * np.log(pred_softmax + 1e-12))
            SoftmaxCE.append(ce)

    # Convert to arrays and safely normalize
    L2Dists = safe_normalize(np.array(L2Dists))
    DistPenalty = safe_normalize(np.array(DistPenalty))
    SoftmaxCE = safe_normalize(np.array(SoftmaxCE))

    fig = plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(2, 2) 
    I0 = Images[0]
    displacements = np.sqrt(np.sum((np.array(ts) - ts[0])**2, axis=1))

    print("Generating frames...")
    os.makedirs("tmp", exist_ok=True)
    
    for i, I in tqdm(enumerate(Images), total=len(Images)):
        # Use Maximum Intensity Projections (MIP) instead of static slices
        # This ensures the target remains visible even as it moves across axes
        I0_x, I_x = np.max(I0, axis=0), np.max(I, axis=0)
        Dx = np.stack([I0_x, I_x, np.zeros_like(I_x)], axis=-1)
        Dx = (Dx * 255 / np.max(I0)).astype(np.uint8)
        
        I0_y, I_y = np.max(I0, axis=1), np.max(I, axis=1)
        Dy = np.stack([I0_y, I_y, np.zeros_like(I_y)], axis=-1)
        Dy = (Dy * 255 / np.max(I0)).astype(np.uint8)

        I0_z, I_z = np.max(I0, axis=2), np.max(I, axis=2)
        Dz = np.stack([I0_z, I_z, np.zeros_like(I_z)], axis=-1)
        Dz = (Dz * 255 / np.max(I0)).astype(np.uint8)
        
        plt.clf()
        ax1img = plt.subplot(221)
        ax1img.imshow(Dx, aspect='auto')
        ax1img.set_title("Projection in 2D")
        # L2 Distance Plot
        ax4 = plt.subplot(222)
        ax4.plot(displacements, L2Dists, 'r-')
        ax4.fill_between(displacements[:i+1], L2Dists[:i+1], color='r', alpha=0.3)
        ax4.set_yscale('log')
        ax4.set_xlabel("Displacement (Voxels)")
        ax4.set_title("L2 Distance") 

        # Softmax CE Plot
        ax5 = plt.subplot(223)
        ax5.plot(displacements, SoftmaxCE, 'g-')
        ax5.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.3)
        ax5.set_yscale('log')
        ax5.set_xlabel("Displacement (Voxels)")
        ax5.set_title("Softmax CE")

        # Distance Penalty Plot
        ax6 = plt.subplot(224)
        ax6.plot(displacements, DistPenalty, 'b-')
        ax6.plot(displacements, SoftmaxCE, 'k--', alpha=0.3)
        ax6.fill_between(displacements[:i+1], DistPenalty[:i+1], color='b', alpha=0.3)
        ax6.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.1)
        ax6.set_yscale('log')
        ax6.set_xlabel("Displacement (Voxels)")
        ax6.set_title("Distance Penalty")
        
        for ax, values in zip([ax4, ax5, ax6], [L2Dists, DistPenalty, SoftmaxCE]):
            ymax = np.max(values) if np.max(values) > 0 else 1.0
            ymin = np.min(values[values > 0]) if len(values[values > 0]) > 0 else 0.0
            ax.tick_params(axis='y', which='both', labelleft=False, labeltop=True)
            ax.set_yticks([ymin, ymax])
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            # Force the plot box to be a perfect square to match imshow
            # ax.set_box_aspect(1)
        plt.tight_layout()
        fig.savefig(f"tmp/{i}.png", format='png')

    # Create GIF
    os.makedirs(save_dir, exist_ok=True)
    png_files = [f"tmp/{i}.png" for i in range(len(Images))]
    frames = []
    
    print("Compiling GIF...")
    for png_path in tqdm(png_files):
        # png_bytes = svg2png(url=svg_path)
        frames.append(imageio.imread(png_path, format='png'))

    gif_path = os.path.join(save_dir, f"disc_concept_w{w:.2f}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"Saved GIF: {gif_path}")

# Run the function to test
if __name__ == "__main__":
    create_disc_figure()    