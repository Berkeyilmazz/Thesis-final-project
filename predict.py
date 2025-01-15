import joblib
import pandas as pd

# Load the saved model
model = joblib.load('final_model.pkl')

df = pd.read_csv('synthesized_data_with_churn.csv')

# Ensure that the columns match the training features
try:
    # Align the input data to match the model's expected features
    df = df[model.feature_names_in_]
except KeyError as e:
    print(f"Error: Missing feature(s) in input data: {str(e)}")
    exit()


# Make predictions
predictions = model.predict(df)

# Print the predictions
print(f"Predictions: {predictions}")

# Save predictions to a CSV file if needed
predictions_df = pd.DataFrame(predictions, columns=["ChurnPrediction"])
predictions_df.to_csv('predictions.csv', index=False)
