import pandas as pd

# Load the dataset
data = pd.read_csv("final_dataset.csv")

# Display the dataset structure
print(f"Original dataset shape: {data.shape}")
print("Missing values per column:")
print(data.isnull().sum())

# Step 1: Drop rows only if all features are NaN
data_cleaned = data.dropna(how='all')

# Step 2: Fill remaining NaNs with default values
for col in data_cleaned.columns:
    if data_cleaned[col].dtype in ['float64', 'int64']:
        # For numerical columns, fill NaN with the column mean
        mean_value = data_cleaned[col].mean()
        data_cleaned[col] = data_cleaned[col].fillna(mean_value)
    else:
        # For categorical columns, fill NaN with 'Unknown'
        data_cleaned[col] = data_cleaned[col].fillna('Unknown')

# Step 3: Drop duplicate rows (if any)
data_cleaned = data_cleaned.drop_duplicates()

# Step 4: Remove non-essential columns (if needed)
if 'CustomerID' in data_cleaned.columns:
    data_cleaned = data_cleaned.drop(columns=['CustomerID'])

# Step 5: Convert date columns to datetime (if applicable)
for col in ['FirstPurchaseDate', 'LastPurchaseDate']:
    if col in data_cleaned.columns:
        data_cleaned[col] = pd.to_datetime(data_cleaned[col], errors='coerce')
        data_cleaned[col] = data_cleaned[col].fillna(pd.to_datetime('1900-01-01'))

# Step 6: Perform undersampling
# Separate the classes
non_churned = data_cleaned[data_cleaned['Churn'] == 0]
churned = data_cleaned[data_cleaned['Churn'] == 1]

# Get the counts of each class
non_churned_count = len(non_churned)
churned_count = len(churned)

print(f"Non-Churned Count: {non_churned_count}")
print(f"Churned Count: {churned_count}")

# Dynamically adjust undersampling size to match the smaller class
undersampling_size = min(non_churned_count, churned_count)

# Perform undersampling on the non-churned class
undersampled_non_churned = non_churned.sample(n=undersampling_size, random_state=42)

# Combine the churned and undersampled non-churned data
balanced_data = pd.concat([undersampled_non_churned, churned])

# Shuffle the dataset to mix the classes
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Final step: Save the cleaned and balanced dataset
balanced_data.to_csv("final_dataset_balanced.csv", index=False)

# Print summary
print(f"Cleaned and balanced dataset saved as 'final_dataset_balanced.csv'. Final shape: {balanced_data.shape}")
print("Class distribution after balancing:")
print(balanced_data['Churn'].value_counts())