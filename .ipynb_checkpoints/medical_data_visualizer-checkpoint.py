import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
data = pd.read_csv('medical_examination.csv')
df = pd.DataFrame(data)

# Add 'overweight' column
df['overweight'] = ((df['weight'] /
                     ((df['height'] * 0.01)**2)) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = df['cholesterol'].apply(lambda cho: 0 if cho == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda gluc: 0 if gluc == 1 else 1)


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  cols = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
  df_cat = pd.melt(df,
                   id_vars=['cardio'],
                   value_vars=cols,
                   var_name='Category',
                   value_name='value')

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = sns.catplot(data=long_df,
                       kind='count',
                       x='Category',
                       hue='value',
                       col='cardio',
                       height=5,
                       aspect=1.2).set_axis_labels('Variables', 'total')

  # Draw the catplot with 'sns.catplot()'
  plt.show()

  # Get the figure for the output
  fig = plt.gcf()

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df[(df['ap_lo'] <= df['ap_hi'])
               & (df['height'] >= df['height'].quantile(0.025)) &
               (df['height'] <= df['height'].quantile(0.975)) &
               (df['weight'] >= df['weight'].quantile(0.025)) &
               (df['weight'] <= df['weight'].quantile(0.975))]

  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.triu(np.ones_like(corr, dtype=bool))

  # Set up the matplotlib figure
  plt.figure(figsize=(10, 8))
  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(correlation_matrix,
              cmap='coolwarm',
              mask=mask,
              annot=True,
              fmt=".1f")

  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig
