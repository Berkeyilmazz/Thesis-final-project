from feature_synthesis import FeatureSynthesizer
import pandas as pd
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

# Assign churn labels
final_dataset = synthesizer.assign_churn_labels(final_features)
final_dataset.to_csv("final_dataset.csv", index=False)
print("Final dataset saved as 'final_dataset.csv'.")