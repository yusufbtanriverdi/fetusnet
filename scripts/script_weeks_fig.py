import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def load_and_process_data(filepath):
    """Load CSV and process week column."""
    df = pd.read_csv(filepath)
    df['week'] = (
        df['week']
        .astype(str)
        .str.replace('semanas', '', regex=False)
        .str.strip()
        .astype(int)
    )
    return df


def count_unique_weeks(df):
    """Count unique samples per week."""
    week_counts = (
        df.groupby(['week'])['nsid']
          .nunique()
          .reset_index(name='n_unique_weeks')
    )
    week_counts['label'] = week_counts['week'].astype(str)
    return week_counts


def plot_and_save(week_counts, output_path='assets/weeks.svg'):
    """Create and save bar plot."""
    fig = plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=week_counts,
        x='label',
        y='n_unique_weeks'
    )
    
    plt.xlabel('Gestational Age (weeks)', labelpad=15)
    plt.ylabel('Count', labelpad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path, format='svg')


def create_weeks_plot(save_dir, df=None, **kwargs):
    """Main execution."""    
    if df is None:
        df = load_and_process_data("C:/Users/user/Projeler/Ph.D/Research/source/sinMaternitat__fold0.csv")
    week_counts = count_unique_weeks(df)
    print(week_counts)
    plot_and_save(week_counts, output_path=f"{save_dir}/weeks.svg")


# if __name__ == '__main__':
#     create_weeks_plot(save_dir='assets')