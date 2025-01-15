import csv
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure y_test is a 1D array
y_test = y_test.values.ravel()

# List of model file paths
model_files = [
    'random_forest.pkl', 'logistic_regression.pkl', 'ada_boost.pkl', 
    'knn.pkl', 'decision_tree.pkl', 'extra_tree.pkl', 'mlp.pkl'
]

# Collect scores
results = []
for model_file in model_files:
    try:
        print(f"Evaluating {model_file}...")
        model = joblib.load(model_file)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        results.append({'Model': model_file, 'Accuracy': accuracy, 'ROC-AUC': roc_auc, 'F1-Score': f1})

    except FileNotFoundError:
        print(f"Model file {model_file} not found. Skipping...")
    except Exception as e:
        print(f"An error occurred while evaluating {model_file}: {e}")

# Save to CSV
output_file = 'eve_model_scores.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Model', 'Accuracy', 'ROC-AUC', 'F1-Score'])
    writer.writeheader()
    writer.writerows(results)

print(f"Scores saved to {output_file}.")