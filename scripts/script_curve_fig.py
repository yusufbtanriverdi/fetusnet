import os
from glob import glob
from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np
import seaborn as sns

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

def plot_aela_figure(radii, edrs, titles, dict, save_dir='ela.svg', show=False):
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
    for edr, title in zip(edrs, titles):
        plt.plot(radii, edr, label=dict[title], marker=9, fillstyle='top', markersize=markersize, linewidth=1.5)
        # plt.hlines(
        #     y=edr[-1],
        #     xmin=radii[0],
        #     xmax=radii[-1],
        #     linestyles='solid',
        #     linewidth=0.3,
        # )
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

def average_edr_per_experiment(directory):
    """
    Returns
    -------
    dict
        {
            'exp1': mean_curve,
            'exp2': mean_curve,
            ...
        }
    """

    experiment_means = {}

    for exp_name in os.listdir(directory):
        exp_dir = os.path.join(directory, exp_name)

        if not os.path.isdir(exp_dir):
            continue

        curves = []

        for csv_file in glob(os.path.join(exp_dir, "*_curve_mean.csv")):
            curve = np.loadtxt(csv_file, delimiter=",")
            curves.append(curve)

        if len(curves) == 0:
            continue

        experiment_means[exp_name] = np.nanmean(
            np.stack(curves),
            axis=0
        )

    return experiment_means

if __name__ == '__main__':
    radii = np.linspace(1, 100, 100)
    experiment_means = average_edr_per_experiment('C:/Users/user/Projeler/Ph.D/Research/source/plot_')
    dict = {'kendall1': 'Softmax CE', 
            'kendall2_1': 'EMD Penalty (w=2)', 
            'kendall2_2': 'EMD Penalty (w=0.5)', 
            'kendall3': 'SSE', 
            'kendall4_2': 'EMD-regularized (w=2)',
            'kendall6': 'EMD-regularized (w=0.5)'
            }
    plot_aela_figure(radii, experiment_means.values(), experiment_means.keys(), dict, show=True)