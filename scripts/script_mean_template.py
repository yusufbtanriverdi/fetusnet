from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import torch
import nrrd

from doc.info.template import create_template

def plot_mean_volume(
    loader,
    device,
    output_dir='.',
    temp_dir='.',
    df_name='mean_templates_info.csv',
    fig_name='mean_volume_plot.png',
    progress_bar=True,
    recompute=True,
    ):
    """Collect training volumes, compute mean volumes, and plot them.


    This function performs two passes:
    1. Iterate over the DataLoader and compute mean volumes for each gestational week.
    2. Compute mean volumes for the three gestational-age categories: 20, 20-30, and 30+.


    If the saved mean volumes already exist and recompute is False, the first iteration is skipped.
    The resulting figure, mean volumes, and DataFrame are all saved to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    df_path = os.path.join(temp_dir, df_name)
    week_output_dir = os.path.join(output_dir, 'weeks')
    category_output_dir = os.path.join(output_dir, 'categories')
    os.makedirs(week_output_dir, exist_ok=True)
    os.makedirs(category_output_dir, exist_ok=True)

    mean_volume_path = os.path.join(temp_dir, 'mean_volumes')
    fig_path = os.path.join(output_dir, fig_name)


    if os.path.exists(mean_volume_path) and not recompute:
        print(f"Loading existing mean volumes from {mean_volume_path}...")
        pass

    else:
        print(f"Collecting volumes from the DataLoader and saving to {mean_volume_path}...")
        records = []
        week_volume_sums = {}
        category_volume_sums = {}
        week_counts = {}
        category_counts = {}
        iterable = tqdm(loader, desc='Collecting volumes', total=len(loader)) if progress_bar else loader

        for batch_index, batch in enumerate(iterable):
            if len(batch['image']['data']) != 1:
                print(f"Skipping batch {batch_index} due to unexpected batch size: {len(batch['image']['data'])}")
                continue

            nsid = batch['name'][0]
            volume = batch['image']['data'][0].to(device)
            week_group = int(batch['week'][0].replace('semanas', '').strip())
            p = batch['spacings'][0][0] # Assuming ISO p for simplicity
            template = create_template(p)

            if week_group <= 20:
                category = '20'
            elif week_group <= 30:
                category = '20-30'
            else:
                category = '30+'

            if progress_bar:
                iterable.set_description(f"Processing {nsid}")

            if week_group not in week_volume_sums:
                week_volume_sums[week_group] = torch.zeros_like(volume, device=device)
                week_counts[week_group] = 0

            if category not in category_volume_sums:
                category_volume_sums[category] = torch.zeros_like(volume, device=device)
                category_counts[category] = 0

            week_volume_sums[week_group] += volume
            week_counts[week_group] += 1

            category_volume_sums[category] += volume
            category_counts[category] += 1

            records.append(
                {
                    'nsid': nsid,
                    'week_group': week_group,
                    'category': category,
                }
            )
            break

        df = pd.DataFrame.from_records(records)
        df.to_csv(df_path, index=False)
        if df.empty:
            raise ValueError('No training volumes were collected. Check the loader contents and gestational-age key.')

        week_mean_volumes = {}
        for week_group in sorted(week_volume_sums):
            week_mean_volumes[week_group] = (
                week_volume_sums[week_group] / week_counts[week_group]
            ).cpu()

        category_mean_volumes = {}
        for category in ['20', '20-30', '30+']:
            if category not in category_volume_sums:
                print(f"Warning: No volumes found for category '{category}'.")
                continue

            category_mean_volumes[category] = (
                category_volume_sums[category] / category_counts[category]
            ).cpu()

        mean_volume_data = {
            'week_mean_volumes': week_mean_volumes,
            'category_mean_volumes': category_mean_volumes,
            'week_counts': week_counts,
            'category_counts': category_counts,
        }
        torch.save(mean_volume_data, mean_volume_path)


    if not week_mean_volumes and not category_mean_volumes:
        raise ValueError('No mean volumes were computed. Check the loader contents and gestational-age values.')

    for week_group in sorted(week_mean_volumes):
        output_path = os.path.join(
            week_output_dir,
            f'mean_volume_week_{week_group}.nrrd',
        )
        nrrd.write(output_path, week_mean_volumes[week_group], header=template)

    for category in ['20', '20-30', '30+']:
        if category not in category_mean_volumes:
            continue
        category_name = category.replace('-', '_').replace('+', '_plus')
        output_path = os.path.join(
            category_output_dir,
            f'mean_volume_{category_name}.nrrd',
        )
        nrrd.write(output_path, category_mean_volumes[category], header=template)

    return df, df_path