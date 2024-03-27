

## import library

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Support Vector Classifier for SVM models
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

## preparation

def svm_balance(data, output_dir='./output/', random_state_base=42):

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare DataFrames for storing results
    results_df = pd.DataFrame(columns=['Trial_num', 'Accuracy', 'AUC'])
    feature_coefficients_df = pd.DataFrame()

    # Retrieve data for the minority class
    minority_data = data[data['Gene_type'] == 'T']
    n = len(minority_data)

    plt.figure(figsize=(10, 8))

    for i in range(10):
        
        random_state = random_state_base + i

        # Balance dataset by selecting equal number of samples from each class
        majority_data = data[data['Gene_type'] == 'FP'].sort_values(by='ED', ascending=True).head(n).drop(columns='ED')
        balanced_data = pd.concat([minority_data.drop(columns='ED'), majority_data])

        # Split into features and target
        X = balanced_data.drop('Gene_type', axis=1)
        y = balanced_data['Gene_type'].apply(lambda x: 1 if x == 'T' else 0)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Use GridSearchCV for tuning SVM parameters
        param_grid = {'C': [0.1, 1, 10, 100]}
        grid_search = GridSearchCV(SVC(kernel='linear', probability=True, random_state=random_state), param_grid, cv=5)
        grid_search.fit(X_train_scaled, y_train)

        # Retrain model with the best parameters
        model = grid_search.best_estimator_
        model.fit(X_train_scaled, y_train)

        # Evaluate the model on test data
        accuracy = model.score(X_test_scaled, y_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Get probabilities for the positive class
        auc_score = roc_auc_score(y_test, y_pred_proba)  # Calculate AUC
        
        # Store results
        results_df = pd.concat([results_df, pd.DataFrame({'Trial_num': [i+1], 'Accuracy': [accuracy], 'AUC': [auc_score]})], ignore_index=True)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='black', alpha=0.3)

        # Extract and store feature importances
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            temp_df = pd.DataFrame({
                'Feature': X.columns,
                'Trial': [i + 1] * len(X.columns),  # Replicate trial number for each feature
                'Coefficient': coefficients
            })
            feature_coefficients_df = pd.concat([feature_coefficients_df, temp_df], ignore_index=True)

    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.grid(False)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Save and output results
    average_accuracy = results_df['Accuracy'].mean()
    average_auc = results_df['AUC'].mean()
    summary_df = pd.DataFrame({'Average Accuracy': [average_accuracy], 'Average AUC': [average_auc]})

    print(f"Average Accuracy: {results_df['Accuracy'].mean()}")
    print(f"Average AUC: {results_df['AUC'].mean()}")
    
    plt.savefig(os.path.join(output_dir, 'average_roc_curve.png'), dpi=300, format='png')

    if not feature_coefficients_df.empty:
        feature_mean_coefficients = feature_coefficients_df.groupby('Feature')['Coefficient'].mean().reset_index()

    # Save the mean coefficients to a CSV file
        feature_mean_coefficients.to_csv(os.path.join(output_dir, 'feature_mean_coefficients.csv'), index=False)
    else:
        feature_mean_coefficients = pd.DataFrame()  # In case no coefficients were collected

    results_df.to_csv(os.path.join(output_dir, 'results_df.csv'), index=False)
    feature_coefficients_df.to_csv(os.path.join(output_dir, 'feature_importance_df.csv'), index=False)
    feature_mean_coefficients.to_csv(os.path.join(output_dir, 'feature_mean_importance.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'summary_df.csv'), index=False)

    return results_df, feature_mean_coefficients, summary_df



# sense
# all
# import data
data = pd.read_csv('all_sense_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_all.csv', index=False)


# single
# import data
data = pd.read_csv('all_sense_single_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_single.csv', index=False)


# multi
# import data
data = pd.read_csv('all_sense_multi_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_multi.csv', index=False)


# antisense
# all
# import data
data = pd.read_csv('all_antisense_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_all_antisense.csv', index=False)


# single
# import data
data = pd.read_csv('all_antisense_single_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_single_antisense.csv', index=False)


# multi
# import data
data = pd.read_csv('all_antisense_multi_ED.csv')
results_df, feature_mean_importance, summary_df = svm_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Coefficient', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = svm_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('svm_results_sum_multi_antisense.csv', index=False)