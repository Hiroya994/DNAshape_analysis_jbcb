
# lasso

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to load a CSV file, create a boxplot, and save the result as a PNG file.

def create_boxplot_and_save(data, top_n=10):
    
    # Calculate the average importance of each feature
    feature_mean_importance = data.groupby('Feature')['Importance'].mean().reset_index()
    
    # Sort features by their average importance in descending order
    sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)
    
    # Select the top_n features with the highest mean importance
    top_features = sorted_features.head(top_n)['Feature']
    
    # Extract the importance data for these top features
    top_features_df = data[data['Feature'].isin(top_features)]
    
    # Draw the boxplot
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=top_features_df, 
        x='Feature', 
        y='Importance', 
        order=top_features[::-1],  # Reverse the order to place highest values on the right
        width=0.6,
        boxprops=dict(edgecolor='black', facecolor='white'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black')
    )

    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)

    # Remove axis label
    plt.ylabel('')
    plt.xlabel('')
    
    # Save the plot as a PNG file
    plt.savefig('feature_importance_boxplot.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()  # Close the figure to prevent further modifications



data = pd.read_csv('feature_importance_df.csv')

create_boxplot_and_save(data)


# svm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to load a CSV file, create a boxplot, and save the result as a PNG file.

def create_boxplot_and_save(data, top_n=10):
    
    # Calculate the average importance of each feature
    feature_mean_importance = data.groupby('Feature')['Coefficient'].mean().reset_index()
    
    # Sort features by their average importance in descending order
    sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)
    
    # Select the top_n features with the highest mean importance
    top_features = sorted_features.head(top_n)['Feature']
    
    # Extract the importance data for these top features
    top_features_df = data[data['Feature'].isin(top_features)]
    
    # Draw the boxplot
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=top_features_df, 
        x='Feature', 
        y='Coefficient', 
        order=top_features[::-1],  # Reverse the order to place highest values on the right
        width=0.6,
        boxprops=dict(edgecolor='black', facecolor='white'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black')
    )

    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14)

    # Remove axis label
    plt.ylabel('')
    plt.xlabel('')
    
    # Save the plot as a PNG file
    plt.savefig('feature_importance_boxplot.png', dpi=300, format='png', bbox_inches='tight')
    plt.close()  # Close the figure to prevent further modifications



data = pd.read_csv('feature_importance_df.csv')

create_boxplot_and_save(data)
