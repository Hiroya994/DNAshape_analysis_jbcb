import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

def lasso_balance(data, output_dir='./output/', random_state_base=42):

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare DataFrames for storing results
    results_df = pd.DataFrame(columns=['Trial_num', 'Accuracy', 'AUPRC', 'AUC'])
    feature_importance_df = pd.DataFrame(columns=['Feature', 'Trial', 'Importance'])
    roc_curves = []
    pr_curves = []

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

        # Initialize Lasso model with cross-validation for parameter tuning
        model = GridSearchCV(LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=random_state),
                             param_grid={'C': np.logspace(-4, 4, 20)},
                             cv=5,
                             scoring='roc_auc')
        
        # Fit the model
        model.fit(X_train_scaled, y_train)
        best_model = model.best_estimator_
        
        # Evaluate the model on test data
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] # Get probabilities for the positive class
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = auc(recall, precision)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        roc_curves.append((fpr, tpr, roc_auc))
        pr_curves.append((recall, precision, pr_auc))

        # Store results
        results_df = pd.concat([results_df, pd.DataFrame({'Trial_num': [i+1], 'Accuracy': [accuracy], 'AUPRC': [auprc], 'AUC': [auc_score]})], ignore_index=True)

        # Extract and store coefficients instead of feature importances
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(best_model.coef_[0]), 'Trial': [i + 1] * len(X.columns)})
        feature_importance_df = pd.concat([feature_importance_df, importance_df], ignore_index=True)


    # Calculate average feature importance
    feature_mean_importance = feature_importance_df.groupby('Feature')['Importance'].mean().reset_index()

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    for fpr, tpr, roc_auc in roc_curves:
        plt.plot(fpr, tpr, color='black', lw=2, alpha=0.3, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.grid(False)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    # Plot PR curve
    plt.figure(figsize=(10, 8))
    for recall, precision, pr_auc in pr_curves:
        plt.plot(recall, precision, color='black', lw=2, alpha=0.3, label=f'AUPRC = {pr_auc:.2f}')
    ticks = np.arange(0, 1.1, 0.2)
    plt.xticks(fontsize=22)
    plt.yticks(ticks, fontsize=22)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
    plt.close()

    # Save and output results
    average_accuracy = results_df['Accuracy'].mean()
    average_auprc = results_df['AUPRC'].mean()
    average_auc = results_df['AUC'].mean()
    
    summary_df = pd.DataFrame({'Average Accuracy': [average_accuracy], 'Average AUPRC': [average_auprc], 'Average AUC': [average_auc]})

    print(f"Average Accuracy: {results_df['Accuracy'].mean()}")
    print(f"Average AUC: {results_df['AUC'].mean()}")
    print(f"Average AUPRC: {results_df['AUPRC'].mean()}")

    results_df.to_csv(os.path.join(output_dir, 'results_df.csv'), index=False)
    feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance_df.csv'), index=False)
    feature_mean_importance.to_csv(os.path.join(output_dir, 'feature_mean_importance.csv'), index=False)
    summary_df.to_csv(os.path.join(output_dir, 'summary_df.csv'), index=False)

    return results_df, feature_mean_importance, summary_df


# sense
# all
# import data
data = pd.read_csv('all_sense_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_all.csv', index=False)


# single
# import data
data = pd.read_csv('all_sense_single_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_single.csv', index=False)


# multi
# import data
data = pd.read_csv('all_sense_multi_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_multi.csv', index=False)


# antisense
# all
# import data
data = pd.read_csv('all_antisense_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_all_antisense.csv', index=False)


# single
# import data
data = pd.read_csv('all_antisense_single_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_single_antisense.csv', index=False)


# multi
# import data
data = pd.read_csv('all_antisense_multi_ED.csv')
results_df, feature_mean_importance, summary_df = lasso_balance(data, output_dir='./all/')

# order the importance by feature_mean_importance
sorted_features = feature_mean_importance.sort_values(by='Importance', ascending=False)

# Initialize an empty DataFrame to aggregate results
aggregate_results_df = pd.DataFrame()

for i in range(2, 11): # Starting from top 2 to top 10 features
    
    # Retrieve top N feature names based on their importance
    top_features = sorted_features.head(i)['Feature'].tolist()
    top_features += ['Gene_type', 'ED']

    # Select only these features from the original DataFrame
    selected_data = data[top_features]

    directory_path = f"./{i}"
    results_df, feature_mean_importance, summary_df = lasso_balance(selected_data, output_dir=directory_path)
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

# Optionally, save the aggregated results to a CSV file
aggregate_results_df.to_csv('lasso_results_sum_multi_antisense.csv', index=False)
