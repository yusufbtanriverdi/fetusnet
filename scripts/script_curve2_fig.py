import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style(
    'whitegrid',
    {'font.family': 'sans-serif', 'font.sans-serif': 'Verdana'}
)
sns.set_theme(
    'paper',
    'whitegrid',
    font_scale=2,
    palette='husl'
)
def plot_aela_figure(radii, edrs, titles, dict, save_dir='assets/ela2.svg', show=False, colors=[None, None]):
    """
    Plot the average expected local accuracy (AELA) figure.

    Args:
        radii (list or torch.Tensor): Radii in mm.
        edr (list or torch.Tensor): Average expected local accuracy values.
        save_dir (str): Path to save the figure.
    """
    upper_limit = [0.75 * r for r in radii]  # 3/4 upper limit
    markersize=5
    fig = plt.figure(figsize=(10, 6))
    fills = ['full', 'full', 'none', 'none']
    for edr, title, color, fill in zip(edrs, titles, colors, fills):
        plt.plot(radii, edr, label=dict[title], marker='D', fillstyle=fill, markersize=markersize, linewidth=1.5, color=color,)

    plt.plot(radii, upper_limit, linestyle='dashed', color='blue', label='Upper Limit', fillstyle='none', markersize=markersize)
    plt.legend(loc='center right')
    # leg = plt.legend(loc='right', bbox_to_anchor=(1.0, 1))
    # fig.add_artist(leg)    
    plt.xlabel('Radius (mm)')
    plt.ylabel('Average Expected Local Accuracy (AELA)')
    plt.yscale('log')
    plt.xlim(1, 100)
    plt.grid(True)
    
    if save_dir:
        plt.savefig(save_dir, format='svg')

    if show:
        plt.show()

    plt.close()  # Free memory


def get_edr_lmks(directory):
    """
    """

    curves = {}
    for exp_name in os.listdir(directory):
        csv_file = os.path.join(directory, exp_name)
        curve = np.loadtxt(csv_file, delimiter=",")
        curves[exp_name] = curve
        if len(curves) == 0:
            continue

    return curves

if __name__ == '__main__':
    radii = np.linspace(1, 100, 100)
    experiment_means = get_edr_lmks('assets/_plot2/')
    dict = {'enR.csv': '$enR$', 
            'enL.csv': '$enL$', 
            's_enR.csv': '$enR$ swapped with $enL$', 
            's_enL.csv': '$enL$ swapped with $enR$', 
            }
    plot_aela_figure(radii, experiment_means.values(), experiment_means.keys(), dict, show=True, 
                     colors=["#62E8F4", "#ED38C0",  '#62E8F4', '#ED38C0'])