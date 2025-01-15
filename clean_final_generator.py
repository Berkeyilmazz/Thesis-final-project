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
# For numerical columns, use 0 or the column mean; for categorical columns, use a placeholder.
for col in data_cleaned.columns:
    if data_cleaned[col].dtype in ['float64', 'int64']:
        data_cleaned[col] = data_cleaned[col].fillna(0)  # Replace NaNs with 0
    else:
        data_cleaned[col] = data_cleaned[col].fillna('Unknown')  # Replace NaNs with 'Unknown'

# Step 3: Drop duplicate rows (if any)
data_cleaned = data_cleaned.drop_duplicates()

# Step 4: Remove non-essential columns
if 'CustomerID' in data_cleaned.columns:
    data_cleaned = data_cleaned.drop(columns=['CustomerID'])

# Step 5: Convert date columns to datetime (if applicable)
for col in ['FirstPurchaseDate', 'LastPurchaseDate']:
    if col in data_cleaned.columns:
        data_cleaned[col] = pd.to_datetime(data_cleaned[col], errors='coerce')

# Final step: Save the cleaned dataset
data_cleaned.to_csv("final_dataset_cleaned.csv", index=False)

# Print summary
print(f"Cleaned dataset saved as 'final_dataset_cleaned.csv'. Final shape: {data_cleaned.shape}")