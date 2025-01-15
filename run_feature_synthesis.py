import pandas as pd
from feature_synthesis import FeatureSynthesizer
import os

# Initialize the synthesizer with a reference date
reference_date = '2024-06-01'  
synthesizer = FeatureSynthesizer(reference_date=reference_date)

# Directory where monthly data is stored
monthly_dir = "monthly_segments"
monthly_files = [f for f in os.listdir(monthly_dir) if f.endswith('.csv')]

# Process each monthly file
all_features = []
for file in monthly_files:
    print(f"Processing file: {file}")
    month_data = pd.read_csv(os.path.join(monthly_dir, file))
    month = file.split('_')[-1].split('.')[0]  # Extract month from filename

    # Synthesize features
    features = synthesizer.synthesize_features([(month, month_data)])
    all_features.append(features)

# Combine all synthesized features
final_features = pd.concat(all_features, axis=0)

# Check the columns in final_features to ensure they are as expected
print("Columns in synthesized features:", final_features.columns)

# Print some data to check if the columns have expected values
print(final_features[['Recency', 'Monetary-2022-01']].head())

# Adjust churn logic based on actual data distribution
final_features['Churn'] = ((final_features['Recency'] > 3) & (final_features['Monetary-2022-01'] < 150))  

# Save the final dataset with churn labels
final_features.to_csv('synthesized_data_with_churn.csv', index=False)
print("Feature synthesis and churn labeling complete. Data saved to 'synthesized_data_with_churn.csv'.")