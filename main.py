from first_preprocess import DataSanitizer
from outlier_filter import OutlierFilter
from monthly_segmenter import MonthlySegmenter
from feature_importance import FeatureImportanceAnalyzer
from train_model import ModelTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV  
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp

# Configure Logging
logging.basicConfig(filename='model_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Step 1: Load and Clean the Data
logging.info("Step 1: Loading and Cleaning the Data...")
sanitizer = DataSanitizer(file_path='eve_data.csv')
cleaned_dataset = sanitizer.clean_data()
logging.info("Data cleaning complete.\n")

# Step 2: Filter Outliers
logging.info("Step 2: Filtering Outliers...")
outlier_filter = OutlierFilter(max_transactions=360, max_monetary=60000, max_item_variety=500)
filtered_dataset = outlier_filter.filter_outliers(cleaned_dataset)
logging.info("Outlier filtering complete.\n")

# Add a synthetic StoreID column with random store identifiers
filtered_dataset = filtered_dataset.copy()  # Create a copy to avoid SettingWithCopyWarning
filtered_dataset['StoreID'] = np.random.randint(1, 21, size=len(filtered_dataset))  

# Step 3: Split Data by Month
logging.info("Step 3: Splitting Data by Month...")
monthly_segmenter = MonthlySegmenter(start_date='2022-01-01', end_date='2024-06-01')
monthly_data = monthly_segmenter.split_by_month(filtered_dataset)
monthly_segmenter.save_monthly_data(monthly_data, output_dir="monthly_segments")
logging.info("Monthly data splitting complete. Saved to 'monthly_segments/'.\n")

# Step 4: Load the final dataset
logging.info("Step 4: Preparing Final Dataset...")
data = pd.read_csv('final_dataset.csv')

# Remove non-numerical columns from features
logging.info("Validating dataset columns for training...")
non_numerical_columns = ['CustomerID', 'TRANS_DATE', 'FirstPurchaseDate', 'LastPurchaseDate']
X = data.drop(columns=['Churn'] + non_numerical_columns, errors='ignore')  # Exclude target and known non-numerical columns
y = data['Churn']  # Target

# Handle missing values
X = X.fillna(X.mean()) 


# Validate that X contains only numerical columns
if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    logging.error("Non-numerical columns found in features. Please check the dataset.")
    print(f"Columns in X: {X.columns.tolist()}")
    raise ValueError("Non-numerical columns found in features.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using BorderlineSMOTE
smote = BorderlineSMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
logging.info("Data preprocessing for model training complete.\n")

# Step 5: Train the Best Model with GridSearchCV
logging.info("Step 5: Tuning the Best Model (Random Forest) using GridSearchCV...")

# Define the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model with the best hyperparameters
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters from GridSearchCV
best_params = grid_search.best_params_
logging.info(f"Best hyperparameters found: {best_params}")

# Train the model with the best hyperparameters
best_model = grid_search.best_estimator_

logging.info("Random Forest model training complete with best hyperparameters.\n")

# Save the trained model
joblib.dump(best_model, 'best_model.pkl')
logging.info("Trained model saved as 'best_model.pkl'.\n")

# Save the trained model
joblib.dump(model, 'final_model.pkl')
logging.info("Trained model saved as 'final_model.pkl'.\n")

def monitor_predictions(new_data_path, actuals_path=None):
    import logging
    import pandas as pd
    from scipy.stats import ks_2samp

    logging.info(f"Loading new data from {new_data_path}...")
    new_data = pd.read_csv(new_data_path)

    # Ensure the new data matches training features
    required_features = model.feature_names_in_
    missing_features = [feature for feature in required_features if feature not in new_data.columns]
    for feature in missing_features:
        logging.warning(f"Feature {feature} is missing in new_data. Setting default value.")
        new_data[feature] = 0  # Default value for missing features

    # Drop any extra columns not in the training set
    new_data = new_data[required_features]

    # Detect data drift
    logging.info("Checking for data drift...")
    drift_detected = False
    for feature in required_features:
        stat, p_value = ks_2samp(X_train[feature], new_data[feature])
        if p_value < 0.05:
            logging.warning(f"Significant data drift detected in feature: {feature}")
            drift_detected = True

    # Make predictions
    logging.info("Making predictions...")
    predictions = model.predict(new_data)
    new_data['ChurnPrediction'] = predictions
    new_data.to_csv('predictions.csv', index=False)
    logging.info("Predictions saved to 'predictions.csv'.\n")

    # If actual outcomes are provided, evaluate model performance
    if actuals_path:
        logging.info(f"Loading actual outcomes from {actuals_path}...")
        actuals = pd.read_csv(actuals_path)

        # Verify the presence of 'CustomerID'
        if 'CustomerID' not in actuals.columns:
            raise ValueError("The 'CustomerID' column is missing in actual_outcomes.csv.")

        # Merge predictions with actual outcomes
        merged = new_data.merge(actuals, on='CustomerID', how='inner')
        print(classification_report(merged['ActualChurn'], merged['ChurnPrediction']))
        print(confusion_matrix(merged['ActualChurn'], merged['ChurnPrediction']))
        logging.info("Model performance evaluation complete.\n")

    # Trigger retraining if drift is detected
    if drift_detected:
        logging.info("Retraining model due to data drift...")
        updated_data = pd.concat([data, new_data], axis=0)
        updated_data.to_csv('updated_dataset.csv', index=False)
        retrain_model('updated_dataset.csv')
