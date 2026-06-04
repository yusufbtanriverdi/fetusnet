import io
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imageio.v2 as imageio
from cairosvg import svg2png
from tqdm import tqdm
from matplotlib.ticker import NullFormatter

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

def _softmax_numpy(volume):
    exp_volume = np.exp(volume - np.max(volume))
    return exp_volume / np.sum(exp_volume)

def create_disc_figure(test_dl, w, save_dir='figures/'):
    """
    Create a figure showing the optimal transport distance on a moving disc in a 50x50 grid.

    Args:
        test_dl (DataLoader): DataLoader containing the test dataset.
        w (float): Weight for the distance penalty.
        save_dir (str): Directory to save the generated figure.
    """ 
    batch = next(iter(test_dl)) # Get the first batch from the DataLoader
    # Move input data and targets to the specified device
    # images = batch['image']['data']
    targets = batch['target'].cpu()
    target_heatmap = targets[0, 0, 0].cpu().numpy() # {!} Assuming batch size of 1, using the first landmark
    target_coords = batch['coords'][0][0].cpu().numpy().ravel()[:3].astype(int)  # Get the voxel coordinates of the target landmark, assuming batch size of 1            
    distance_map = targets[0, 0, 1].cpu().numpy() # {!} Assuming batch size of 1   
    D, H, W = target_heatmap.shape
    max_shift = np.array([D, H, W], dtype=float) * 0.1
    ts = [target_coords] + [target_coords + max_shift * t for t in np.linspace(0, 1, 128)]  # Create a series of shifted coordinates for testing
    # Compute L2 distances and Wasserstein
    Images = [target_heatmap]
    L2Dists = [0.0]
    DistPenalty = [0.0]
    SoftmaxCE = [0.0]
    for i, t in enumerate(ts):
        shift_vox = np.round(t - target_coords).astype(int)
        I = np.roll(target_heatmap, shift_vox, axis=(0, 1, 2))
        if i > 0:
            Images.append(I)
            L2Dists.append(np.sum((I - target_heatmap)**2))
            DistPenalty.append(np.sum(_softmax_numpy(I) * distance_map ** w))
            SoftmaxCE.append(np.sum(- _softmax_numpy(target_heatmap) * np.log(_softmax_numpy(I))))

    L2Dists = np.array(L2Dists)
    DistPenalty = np.array(DistPenalty)
    SoftmaxCE = np.array(SoftmaxCE)

    L2Dists /= np.max(L2Dists)
    DistPenalty /= np.max(DistPenalty)
    SoftmaxCE /= np.max(SoftmaxCE)
    fig = plt.figure(figsize=(15, 10))
    I0 = Images[0]
    displacements = np.sqrt(2)*(ts - ts[0])[..., 0]  # Calculate the displacement in voxel units (assuming isotropic spacing for simplicity)
    # print("DistPenalty: ", DistPenalty[1:])
    for i, I in tqdm(enumerate(Images), total=len(Images)):
        I0_x = I0[target_coords[0], :, :]
        I_x = I[:, :, target_coords[0]]
        Dx = np.concatenate((I0_x[:, :, None], I_x[:, :, None], 0*I_x[:, :, None]), 2)
        Dx = Dx*255/np.max(I0)
        Dx = np.array(Dx, dtype=np.uint8)
        
        I0_y = I0[:, target_coords[1], :]
        I_y = I[:, :, target_coords[1]]
        Dy = np.concatenate((I0_y[:, :, None], I_y[:, :, None], 0*I_y[:, :, None]), 2)
        Dy = Dy*255/np.max(I0)
        Dy = np.array(Dy, dtype=np.uint8)

        I0_z = I0[:, :, target_coords[2]]
        I_z = I[:, :, target_coords[2]]
        Dz = np.concatenate((I0_z[:, :, None], I_z[:, :, None], 0*I_z[:, :, None]), 2)
        Dz = Dz*255/np.max(I0)
        Dz = np.array(Dz, dtype=np.uint8)
        
        plt.clf()
        ax1img = plt.subplot(231)
        ax1img.imshow(Dx)
        ax2img = plt.subplot(232)
        ax2img.imshow(Dy)
        ax3img = plt.subplot(233)
        ax3img.imshow(Dz)

        ax4 = plt.subplot(234)
        ax4.plot(displacements, L2Dists, 'r-')
        ax4.set_yscale('log')
        ax4.fill_between(displacements[:i+1], L2Dists[:i+1], color='r', alpha=0.3)
        ax4.set_xlabel("Displacements")
        ax4.set_title("L2 Distance") 

        ax5 = plt.subplot(235)
        ax5.plot(displacements, DistPenalty, 'b-')
        ax5.plot(displacements, SoftmaxCE, 'k--', alpha=0.3)
        ax5.set_yscale('log', base=10)
        ax5.fill_between(displacements[:i+1], DistPenalty[:i+1], color='b', alpha=0.3)
        ax5.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.1)
        ax5.set_xlabel("Displacements")
        ax5.set_title("Distance Penalty")

        ax6 = plt.subplot(236)
        ax6.plot(displacements, SoftmaxCE, 'k-')
        ax6.set_yscale('log', base=10)
        ax6.fill_between(displacements[:i+1], SoftmaxCE[:i+1], color='g', alpha=0.3)
        ax6.set_xlabel("Displacements")
        ax6.set_title("Softmax CE")
        for ax, values in zip([ax4, ax5, ax6], [L2Dists, DistPenalty, SoftmaxCE]):
            ymax = np.max(values)
            ymin = np.min(np.array(values)[np.array(values) > 0])
            ax.tick_params(axis='y', which='both', labelleft=False, labeltop=True)
            ax.set_yticks([ymin, ymax])
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

        # plt.suptitle(f"NSID: {nsid}, Visible: {visibles}")  
        os.makedirs("tmp", exist_ok=True)
        fig.savefig("tmp/%i.svg"%i, format='svg')
        # plt.show()

    # Create a GIF from the saved SVG frames
    os.makedirs(save_dir, exist_ok=True)
    svg_files = [f"tmp/{i}.svg" for i in range(len(Images))]
    frames = []
    for svg_path in svg_files:
        png_bytes = svg2png(url=svg_path)
        frames.append(imageio.imread(io.BytesIO(png_bytes)))

    gif_path = os.path.join(save_dir, f"disc_concept_w{w:.2f}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"Saved GIF: {gif_path}")

    