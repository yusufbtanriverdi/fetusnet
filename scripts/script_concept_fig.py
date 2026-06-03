import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

def fig_A(radius=5, shape = 'circle', colors = ['red', 'blue'], ):
    pass

def testMovingDisc():
    """
    Show optimal transport on a moving disc in a 50x50 grid
    """
    ## Step 1: Setup problem
    pix = np.linspace(0, 1, 256)
    # Setup grid
    X, Y = np.meshgrid(pix, pix)
    ts = np.linspace(0, 1, 256)
    
    ## Step 2: Compute L2 distances and Wasserstein
    Images = []
    radius = 0.4
    L2Dists = [0.0]
    RegSCEDists = [0.0]
    WassDists = [0.0]
    for i, t in enumerate(ts):
        # I = 1e-5 + np.array((X-t)**2 + (Y-t)**2 < radius**2, dtype=float)
        # I /= np.sum(I)
        dist = np.sqrt(np.array((X-t)**2 + (Y-t)**2)) / (2*3**2)
        I = np.exp(-dist) # * np.array((X-t)**2 + (Y-t)**2 < radius**2)
        # I = 1e-5 + np.array((X-t)**2 + (Y-t)**2 < radius**2) * dist
        I /= np.sum(I)
        Images.append(I)
        if i > 0:
            L2Dists.append(np.sum((I-Images[0])**2))
            WassDists.append(np.sum(- Images[0] * np.log(I + 1e-15)))
            RegSCEDists.append(np.sum(- Images[0] * np.log(I + 1e-15)) + np.sum(I * dist*35))
            #  + np.sum(- Images[0] * np.log(I + 1e-15)) + 
            # wass = ot.sinkhorn2(Images[0].flatten(), I.flatten(), M, 1.0)
            # print(wass)
            # WassDists.append(wass)

    
    ## Step 3: Make Animation
    WassDists = np.array(WassDists)
    I0 = Images[0]
    plt.figure(figsize=(15, 5))
    displacements = np.sqrt(2)*(ts - ts[0])
    for i, I in tqdm(enumerate(Images), total=len(Images)):
        plt.clf()
        D = np.concatenate((I0[:, :, None], I[:, :, None], 0*I[:, :, None]), 2)
        D = D*255/np.max(I0)
        D = np.array(D, dtype=np.uint8)
        plt.subplot(141)
        plt.imshow(D, extent = (pix[0], pix[-1], pix[-1], pix[0]))
        plt.subplot(142)
        plt.plot(displacements, L2Dists)
        plt.stem([displacements[i]], [L2Dists[i]])
        plt.xlabel("Displacements")
        plt.ylabel("L2 Dist")
        plt.title("L2 Dist")
        plt.subplot(143)
        plt.plot(displacements, WassDists)
        plt.stem([displacements[i]], [WassDists[i]])
        plt.xlabel("Displacements")
        plt.ylabel("Wasserstein Dist")
        plt.title("Wasserstein Dist")
        plt.subplot(144)
        plt.plot(displacements, RegSCEDists)
        plt.stem([displacements[i]], [RegSCEDists[i]])
        plt.xlabel("Displacements")
        plt.ylabel("Reg. Softmax Cross Entropy")
        plt.title("Reg. Softmax Cross Entropy")
        plt.savefig("tmp/%i.png"%i, bbox_inches='tight')
        # plt.show()
if __name__ == '__main__':
    testMovingDisc()