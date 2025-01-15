import feature_importance
import importlib

importlib.reload(feature_importance)  # Force reload of the module

from feature_importance import FeatureImportanceAnalyzer
import joblib
import pandas as pd

# Load the model and data
model = joblib.load('final_model.pkl')
data = pd.read_csv('final_dataset_cleaned.csv')

# Use feature names from the model
if hasattr(model, "feature_names_in_"):
    features = model.feature_names_in_
else:
    features = data.drop(columns=['Churn']).columns  # Fallback to manual extraction

# Initialize the analyzer
analyzer = FeatureImportanceAnalyzer(model, feature_names=features)

# Get and plot feature importance
try:
    print("Calculating feature importance...")
    importance_df = analyzer.plot_importance()

    # Save importance data to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance data saved to 'feature_importance.csv'.")
except ValueError as e:
    print(f"Error: {e}")