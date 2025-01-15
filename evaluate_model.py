import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load test data
try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')['Churn']  # Ensure the target column matches training
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Test data files not found. Please check 'X_test.csv' and 'y_test.csv'.")
    exit()

# List of model file paths
model_files = [
    'random_forest.pkl', 'logistic_regression.pkl', 'ada_boost.pkl', 
    'knn.pkl', 'decision_tree.pkl', 'extra_tree.pkl', 'mlp.pkl'
]

# Dictionary to store evaluation results
results = {}

# Evaluate each model
for model_file in model_files:
    try:
        print(f"Evaluating {model_file}...")
        model = joblib.load(model_file)
        print(f"Model {model_file} loaded successfully.")
        
        # Ensure X_test matches the training features
        X_test_filtered = X_test[model.feature_names_in_]

        # Generate predictions and probabilities
        y_pred = model.predict(X_test_filtered)
        y_proba = model.predict_proba(X_test_filtered)[:, 1] if hasattr(model, "predict_proba") else None
        print(f"Predictions generated for {model_file}.")
        
        # Calculate metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
        print(f"Metrics calculated for {model_file}.")
        
        # Save metrics to results dictionary
        results[model_file] = {
            "accuracy": classification_rep["accuracy"],
            "precision": classification_rep["weighted avg"]["precision"],
            "recall": classification_rep["weighted avg"]["recall"],
            "f1-score": classification_rep["weighted avg"]["f1-score"],
            "roc_auc": roc_auc
        }
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc}")
        print("\n" + "-"*50 + "\n")
    
    except FileNotFoundError:
        print(f"Model file {model_file} not found. Skipping...")
    except ValueError as ve:
        print(f"ValueError while evaluating {model_file}: {ve}")
    except Exception as e:
        print(f"Error evaluating {model_file}: {e}. Skipping...")

# Save metrics to CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv("evaluation_metrics.csv")
print("Metrics saved to 'evaluation_metrics.csv'.")

# Visualize the results
metrics = ["accuracy", "precision", "recall", "f1-score", "roc_auc"]
for metric in metrics:
    plt.figure(figsize=(8, 5))
    
    # Filter results for models with valid metrics
    metric_values = [results[model][metric] for model in results.keys() if metric in results[model] and results[model][metric] != "N/A"]
    model_names = [model.replace(".pkl", "").replace("_", " ").capitalize() for model in results.keys() if metric in results[model] and results[model][metric] != "N/A"]
    
    if metric_values:  # Only plot if there are valid metrics
        plt.bar(model_names, metric_values)
        plt.title(f"Model Comparison: {metric.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No valid data for {metric}. Skipping plot.")