import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from matplotlib.ticker import NullFormatter

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

def create_disc_figure(save_dir, temp_dir, w=2, **kwargs):
    """
    Create a figure showing the optimal transport distance on a moving disc.
    """
    volume_shape = (64, 40, 64)
    start_coords = np.array([16, 10, 16])    # Top-Left-Front
    end_coords = np.array([50, 20, 50])   # Bottom-Right-Back
    num_samples = kwargs.get('num_samples', 128)

    print(f"Generating dummy data starting at {start_coords}...")
    target_heatmap, distance_map = generate_dummy_data(shape=volume_shape, center=start_coords)
    
    # Create a linear trajectory from start_coords to end_coords
    ts = [start_coords + (end_coords - start_coords) * t for t in np.linspace(0, 1, num_samples)]
    
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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8)) 
    I0 = Images[0]
    displacements = np.sqrt(np.sum((np.array(ts) - ts[0])**2, axis=1))

    print("Generating frames...")
    os.makedirs(temp_dir, exist_ok=True)
    
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
        ax6.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.2)
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
        fig.savefig(f"{temp_dir}/{i}.png", format='png')

    # Create GIF
    os.makedirs(save_dir, exist_ok=True)
    png_files = [f"{temp_dir}/{i}.png" for i in range(len(Images))]
    frames = []
    
    print("Compiling GIF...")
    for png_path in tqdm(png_files):
        # png_bytes = svg2png(url=svg_path)
        frames.append(imageio.imread(png_path, format='png'))

    gif_path = os.path.join(save_dir, f"disc_concept_w{w:.2f}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"Saved GIF: {gif_path}")

# # Run the function to test
# if __name__ == "__main__":
#     create_disc_figure()    