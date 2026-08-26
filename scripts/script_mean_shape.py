from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os 
import pyvista as pv

def plot_mean_shape(
    loader,
    device,
    lmks,
    output_dir='.',
    temp_dir='.',
    df_name='mean_shape_coordinates.csv',
    fig_name='mean_shape_plot.png',
    html_name='mean_shape_3D.html',
    progress_bar=True,
    check_visibility=True,
    recompute=False,
    palette='tab20c',
    exp_dir=None,
    ):
    """Collect landmark coordinates, compute means, and plot them.

    This function performs two passes:
    1. Iterate over the DataLoader and store visible landmark coordinates in a pandas DataFrame.
    2. Iterate over the saved DataFrame and plot every coordinate in blue.

    If the CSV already exists and recompute is False, the first iteration is skipped.
    The resulting figure and DataFrame are both saved to disk.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_path = os.path.join(temp_dir, df_name)
    fig_path = os.path.join(output_dir, fig_name)

    if not exp_dir:
        if os.path.exists(df_path) and not recompute:
            print(f"Loading existing target landmark coordinates from {df_path}...")
            df = pd.read_csv(df_path)
        else:
            print(f"Collecting target landmark coordinates from DataLoader and saving to {df_path}...")
            records = []
            iterable = tqdm(loader, desc='Collecting landmark coordinates', total=len(loader)) if progress_bar else loader

            for batch_index, batch in enumerate(iterable):
                if len(batch['image']['data']) != 1:
                    print(f"Skipping batch {batch_index} due to unexpected batch size: {len(batch['image']['data'])}")
                    continue

                nsid = batch['name'][0]
                visibles = batch['visibles'][0]
                target_coord_tensor = batch['coords'][0].to(device)
                if progress_bar:
                    iterable.set_description(f"Processing {nsid}")

                for lmk_index, lmk in enumerate(lmks):
                    if check_visibility and lmk not in visibles:
                        continue

                    coord = target_coord_tensor[lmk_index].cpu().numpy()
                    records.append(
                        {
                            'nsid': nsid,
                            'lmk': lmk,
                            'x': float(coord[0]),
                            'y': float(coord[1]),
                            'z': float(coord[2]),
                        }
                    )

            df = pd.DataFrame.from_records(records)
            df.to_csv(df_path, index=False)

    if exp_dir:
        df_path = os.path.join(exp_dir, df_name)
        if os.path.exists(df_path) and not recompute:
            print(f"Loading existing predicted mean shape coordinates from {exp_dir}...")
            df = pd.read_csv(df_path)
        else:
            print(f"Recomputing predicted mean shape coordinates from {exp_dir}...")
            records = []
            iterable = tqdm(loader, desc='Collecting landmark coordinates', total=len(loader)) if progress_bar else loader
            for batch_index, batch in enumerate(iterable):
                if len(batch['image']['data']) != 1:
                    print(f"Skipping batch {batch_index} due to unexpected batch size: {len(batch['image']['data'])}")
                    continue

                nsid = batch['name'][0]
                visibles = batch['visibles'][0]
                output_coord_tensor = pd.read_csv(os.path.join(exp_dir, f"eval/{nsid}.csv"))
                if progress_bar:
                    iterable.set_description(f"Processing {nsid}")

                for lmk_index, lmk in enumerate(lmks):
                    if check_visibility and lmk not in visibles:
                        continue

                    coord = output_coord_tensor.iloc[lmk_index]
                    records.append(
                        {
                            'nsid': nsid,
                            'lmk': lmk,
                            'x': float(coord['x']),
                            'y': float(coord['y']),
                            'z': float(coord['z']),
                        }
                    )

            df = pd.DataFrame.from_records(records)
            df.to_csv(df_path, index=False)

    if df.empty:
        raise ValueError('No landmark coordinates were collected. Check the loader contents or visibility settings.')

    projection_specs = [
        ('x', 'y', 'X-Y projection'),
        ('x', 'z', 'X-Z projection'),
        ('y', 'z', 'Y-Z projection'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    colors = plt.get_cmap(palette).colors[:len(lmks)]  # Get a color for each landmark
    for lmk, color in zip(lmks, colors):
        lmk_df = df[df['lmk'] == lmk]
        if lmk_df.empty:
            print(f"Warning: No coordinates found for landmark '{lmk}'.")
        means = {
            'x': lmk_df['x'].mean(),
            'y': lmk_df['y'].mean(),
            'z': lmk_df['z'].mean(),
        }
        for ax, (axis_x, axis_y, title) in zip(axes, projection_specs):
            ax.scatter(lmk_df[axis_x], lmk_df[axis_y], c=color, alpha=0.35, s=20, label='all coordinates')
            ax.scatter(
                means[axis_x],
                means[axis_y],
                c='red',
                s=40,
                marker='X',
                label='mean coordinate',
            )
            ax.set_title(title)
            ax.set_xlabel(axis_x.upper())
            ax.set_ylabel(axis_y.upper())
            ax.grid(True, linestyle=':', alpha=0.5)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(unique.values(), unique.keys(), fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=False, ncol=5)

    # fig.suptitle('Mean Landmark Shape and Coordinate Cloud', fontsize=18)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    pv.set_plot_theme('document')
    pl = pv.Plotter(off_screen=True)
    for lmk, color in zip(lmks, colors):
        lmk_df = df[df['lmk'] == lmk]
        if lmk_df.empty:
            print(f"Warning: No coordinates found for landmark '{lmk}' in 3D plot.")
            continue
        points = lmk_df[['x', 'y', 'z']].values
        pl.add_points(points, color=color, point_size=5, render_points_as_spheres=True, opacity=0.35)
        mean_point = df[df['lmk'] == lmk][['x', 'y', 'z']].mean().values
        pl.add_mesh(pv.Sphere(radius=0.5, center=mean_point), color='red', label='mean coordinate')
    pl.add_legend()
    pl.export_html(os.path.join(output_dir, html_name))
    pl.show()
    return df, fig_path, df_path
