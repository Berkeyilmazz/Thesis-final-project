# predict.py
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('final_model.pkl')

# Example input data 
input_data = {
    'Monetary-2022-01': [500],  # Monetary value in January 2022
    'Frequency-2022-01': [20],  # Frequency of purchases in January 2022
    'ItemVariety-2022-01': [5],  # Number of different items purchased in January 2022
    'StoreVariety-2022-01': [3],  # Number of stores visited in January 2022
    'SubscriptionDuration': [12],  # Duration of subscription in months
    'Recency': [2],  # Recency of last purchase (number of months since last purchase)
    'Monetary-2022-02': [300],  # Monetary value in February 2022
    'Frequency-2022-02': [15],  # Frequency of purchases in February 2022
    'ItemVariety-2022-02': [4],  # Number of different items purchased in February 2022
    'StoreVariety-2022-02': [2],  # Number of stores visited in February 2022
    'Recency-2022-02': [3],  # Recency of last purchase in February 2022
}

# Convert the input data to a DataFrame
df = pd.DataFrame(input_data)

# Ensure columns match the training features
try:
    df = df[model.feature_names_in_]  # Ensure the input data matches the model's expected features
except KeyError as e:
    print(f"Error: Missing feature(s) in input data: {str(e)}")
    exit()

# Make predictions
predictions = model.predict(df)

# Print the predictions
print(f"Predictions: {predictions}")