import pandas as pd
from feature_synthesis import FeatureSynthesizer

# Example: Load monthly data with full paths
monthly_data = [
    ('2022-01', pd.read_csv('/Users/berkeyilmaz/Desktop/eveModeller/monthly_segments/monthly_data_2022-01.csv')),
    ('2022-02', pd.read_csv('/Users/berkeyilmaz/Desktop/eveModeller/monthly_segments/monthly_data_2022-02.csv')),
    # Add more months as needed
]

# Initialize the synthesizer
reference_date = '2024-06-01'
synthesizer = FeatureSynthesizer(reference_date)

# Synthesize features
print("Synthesizing features...")
synthesized_data = synthesizer.synthesize_features(monthly_data)

# Assign churn labels
print("Assigning churn labels...")
churn_labeled_data = synthesizer.assign_churn_labels(synthesized_data)

# Save results to a file
churn_labeled_data.to_csv('synthesized_data_with_churn.csv', index=False)
print("Feature synthesis and churn labeling complete. Data saved to 'synthesized_data_with_churn.csv'.")