
import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc

def lasso_simple_test(data, random_state_base=42):

    # Prepare DataFrames for storing results
    results_df = pd.DataFrame(columns=['Trial_num', 'Accuracy', 'AUPRC', 'AUC'])

    # Retrieve data for the minority class
    minority_data = data[data['Gene_type'] == 'T']
    n = len(minority_data)

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

        # Store results
        results_df = pd.concat([results_df, pd.DataFrame({'Trial_num': [i+1], 'Accuracy': [accuracy], 'AUPRC': [auprc], 'AUC': [auc_score]})], ignore_index=True)

    # Save and output results
    average_accuracy = results_df['Accuracy'].mean()
    average_auprc = results_df['AUPRC'].mean()
    average_auc = results_df['AUC'].mean()
    
    summary_df = pd.DataFrame({'Average Accuracy': [average_accuracy], 'Average AUPRC': [average_auprc], 'Average AUC': [average_auc]})

    return summary_df


# shuffle test

all_data = pd.read_csv('all_sense_single_ED.csv')
selected_parameters = ['down_23_ProT', 'up_13_MGW', 'down_8_MGW', 'down_11_MGW', 'up_19_MGW']
test_parameter = 'down_17_ProT'

selected_columns = selected_parameters + [test_parameter, 'Gene_type', 'ED']
data_selected = all_data[selected_columns]

summary_df = lasso_simple_test(data_selected, random_state_base=42)
summary_df['Selected_Parameter'] = test_parameter

all_columns_list = set(all_data.columns.tolist())
excluded_columns = set(selected_parameters + [test_parameter, 'Gene_type', 'ED'])
columns_list = list(all_columns_list - excluded_columns)


# Initialize an empty DataFrame to aggregate results
aggregate_results_df = summary_df

for i in range(10):

    selected_item = random.choice(columns_list)
    selected_columns = selected_parameters + [selected_item, 'Gene_type', 'ED']
    data_selected = all_data[selected_columns]

    summary_df = lasso_simple_test(data_selected, random_state_base=42)
    summary_df['Selected_Parameter'] = selected_item
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

aggregate_results_df.to_csv('aggregate_results.csv', index=False)


# shuffle test

# 2

all_data = pd.read_csv('all_sense_ED.csv')
selected_parameters = ['up_13_MGW', 'down_7_Roll', 'down_17_ProT', 'down_5_ProT', 'down_7_HelT', 'down_13_MGW', 'down_10_ProT', 'down_5_Roll', 'up_23_Roll']
test_parameter = 'down_23_ProT'

selected_columns = selected_parameters + [test_parameter, 'Gene_type', 'ED']
data_selected = all_data[selected_columns]

summary_df = lasso_simple_test(data_selected, random_state_base=42)
summary_df['Selected_Parameter'] = test_parameter

all_columns_list = set(all_data.columns.tolist())
excluded_columns = set(selected_parameters + [test_parameter, 'Gene_type', 'ED'])
columns_list = list(all_columns_list - excluded_columns)


# Initialize an empty DataFrame to aggregate results
aggregate_results_df = summary_df

for i in range(10):

    selected_item = random.choice(columns_list)
    selected_columns = selected_parameters + [selected_item, 'Gene_type', 'ED']
    data_selected = all_data[selected_columns]

    summary_df = lasso_simple_test(data_selected, random_state_base=42)
    summary_df['Selected_Parameter'] = selected_item
    aggregate_results_df = pd.concat([aggregate_results_df, summary_df], ignore_index=True)

aggregate_results_df.to_csv('aggregate_results_v2.csv', index=False)