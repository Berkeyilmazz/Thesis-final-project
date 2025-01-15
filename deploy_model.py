import joblib
import pandas as pd

# Step 1: Load the Saved Model
try:
    print("Loading the saved model...")
    model = joblib.load('final_model.pkl')
    print("Model loaded successfully.\n")
except FileNotFoundError:
    print("Error: Model file 'final_model.pkl' not found.")
    exit(1)

# Step 2: Load and Preprocess New Data
try:
    print("Loading new data...")
    new_data = pd.read_csv('new_data.csv') 
    print("New data loaded successfully.\n")
except FileNotFoundError:
    print("Error: Data file 'new_data.csv' not found.")
    exit(1)

# Ensure the new data has the same features as the training data
if hasattr(model, 'feature_names_in_'):
    required_features = model.feature_names_in_
else:
    print("Error: The model does not contain feature information.")
    exit(1)

# Check for missing features in new data
missing_features = set(required_features) - set(new_data.columns)
if missing_features:
    print(f"Error: Missing required features in new data: {missing_features}")
    exit(1)

# Select only the required features
new_data = new_data[required_features]

print("New data preprocessing complete.\n")

# Step 3: Make Predictions
print("Making predictions...")
predictions = model.predict(new_data)

# Step 4: Output Results
output_file = 'eve_predictions.csv'
new_data['ChurnPrediction'] = predictions
new_data.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'.\n")