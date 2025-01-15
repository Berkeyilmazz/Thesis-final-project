import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
data = pd.read_csv("final_dataset_cleaned.csv")

# Shuffle the data to randomize the order
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check basic info about the dataset
print("Dataset Shape:", data.shape)
print("Dataset Columns:", data.columns.tolist())

# Ensure there are no missing values and fill missing values appropriately
if data.isnull().sum().any():
    print("Handling missing values...")

    
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:  
            data[col] = data[col].fillna(data[col].mean())  
        else:  
            data[col] = data[col].fillna(data[col].mode()[0])  

# Check that 'Churn' is a binary target
if "Churn" not in data.columns:
    raise ValueError("Target column 'Churn' not found in the dataset.")
if data["Churn"].nunique() != 2:
    raise ValueError("Target column 'Churn' must be binary.")

# Split features and target
X = data.drop(columns=["Churn"])  # Features
y = data["Churn"]  # Target

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check the shapes of the resulting datasets
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Save the splits for later use
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Train-test split saved as CSV files.")
