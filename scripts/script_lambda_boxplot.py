import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=2, palette='husl')

color = "#337255"

df_lambda = pd.read_csv('assets/lambdas.csv')
path = r'C:/Users/user/Projeler/Ph.D/Research/source/plot'
fig, ax = plt.subplots(figsize=(10, 10))

for file in df_lambda['Name'].values:
    experiment_name = file
    file += '.csv'
    df = pd.read_csv(os.path.join(path, file))
    lambda_val = df_lambda.loc[
        df_lambda['Name'] == experiment_name,
        'lambdas'
    ].iloc[0]
    df['lambda'] = lambda_val
    fill = True
    if lambda_val == 0.2:
        fill=False
        df = df[:157]
    sns.boxplot(
        data=df,
        x='lambda',
        y='dmean',
        color=color,
        fill=fill,
        ax=ax # Explicitly draw onto our captured axis
    )

ax.tick_params(axis='y', labelleft=True)
plt.ylabel('d-mean Score (mm)', labelpad=15)
plt.xticks(ha='right')
plt.xlabel('Coefficient (\u03BB) of EMD-regularization')
plt.tight_layout()
plt.show()  
fig.savefig('assets/boxplot.svg', format='svg')

