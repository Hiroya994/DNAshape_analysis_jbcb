
import matplotlib.pyplot as plt

def create_boxplot(dataframe, column_name, width, height):
    # Calculate median and standard deviation for the specified column
    median = dataframe[column_name].median()
    std_dev = dataframe[column_name].std()

    plt.figure(figsize=(width, height))
    plt.boxplot(dataframe[column_name], vert=True, patch_artist=True,
                boxprops=dict(facecolor='white'),
                medianprops=dict(color="black"),
                widths=0.6)
    # Use calculated median and standard deviation for the axhline
    plt.axhline(y=median + std_dev, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=median - std_dev, color='black', linestyle='--', linewidth=1)
    plt.ylim(0, 1)
    plt.yticks(fontsize=22)
    plt.gca().xaxis.set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.savefig('median_sd_boxplot.png', dpi=300, format='png', bbox_inches='tight')

import pandas as pd

df = pd.read_csv('results_df.csv')
create_boxplot(df, 'AUC', 2, 8)

import matplotlib.pyplot as plt

def create_boxplot(dataframe, column_name, width, height):
    # Calculate median and standard deviation for the specified column
    median = dataframe[column_name].median()
    std_dev = dataframe[column_name].std()

    plt.figure(figsize=(width, height))
    plt.boxplot(dataframe[column_name], vert=True, patch_artist=True,
                boxprops=dict(facecolor='white'),
                medianprops=dict(color="black"),
                widths=0.6)
    # Use calculated median and standard deviation for the axhline
    plt.axhline(y=median + std_dev, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=median - std_dev, color='black', linestyle='--', linewidth=1)
    plt.ylim(0, 1)
    plt.yticks(fontsize=22)
    plt.gca().xaxis.set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.savefig('median_sd_boxplot_AUPRC.png', dpi=300, format='png', bbox_inches='tight')

import pandas as pd

df = pd.read_csv('results_df.csv')
create_boxplot(df, 'AUPRC', 2, 8)
