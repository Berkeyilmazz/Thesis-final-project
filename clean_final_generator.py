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
# For numerical columns, use the column mean; for categorical columns, use a placeholder.
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

# Final step: Save the cleaned dataset
data_cleaned.to_csv("final_dataset_cleaned.csv", index=False)

# Print summary
print(f"Cleaned dataset saved as 'final_dataset_cleaned.csv'. Final shape: {data_cleaned.shape}")
print("Missing values after cleaning:")
print(data_cleaned.isnull().sum())
