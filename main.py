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
from logistic_regression import train_logistic_regression
from knn_model import train_knn
from adaboost_model import train_adaboost
from extra_tree_model import train_extra_tree
from decision_tree_model import train_decision_tree
from mlp_classifier_model import train_mlp_classifier
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
filtered_dataset = filtered_dataset.copy()
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
X = data.drop(columns=['Churn'] + non_numerical_columns, errors='ignore')
y = data['Churn']

# Handle missing values
X = X.fillna(X.mean())

# Validate that X contains only numerical columns
if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
    logging.error("Non-numerical columns found in features. Please check the dataset.")
    raise ValueError("Non-numerical columns found in features.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using BorderlineSMOTE
smote = BorderlineSMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
logging.info("Data preprocessing for model training complete.\n")

# Train and Evaluate Logistic Regression
logging.info("Step 5: Training Logistic Regression...")
best_lr_model = train_logistic_regression(X_train_resampled, y_train_resampled, X_test, y_test)

# Train and Evaluate Random Forest
logging.info("Step 6: Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 8, 11],
    'min_samples_leaf': [2, 5, 8, 11]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train_resampled, y_train_resampled)
best_rf_model = rf_grid_search.best_estimator_

# Train and Evaluate KNN
logging.info("Step 7: Training KNN...")
best_knn_model = train_knn(X_train_resampled, y_train_resampled, X_test, y_test)

# Train and Evaluate AdaBoost
logging.info("Step 8: Training AdaBoost...")
best_ab_model = train_adaboost(X_train_resampled, y_train_resampled, X_test, y_test)

# Train and Evaluate Extra Trees
logging.info("Step 9: Training Extra Trees...")
best_et_model = train_extra_tree(X_train_resampled, y_train_resampled, X_test, y_test)

# Train and Evaluate Decision Tree
logging.info("Step 10: Training Decision Tree...")
best_dt_model = train_decision_tree(X_train_resampled, y_train_resampled, X_test, y_test)

# Train and Evaluate MLP Classifier
logging.info("Step 11: Training MLP Classifier...")
best_mlp_model = train_mlp_classifier(X_train_resampled, y_train_resampled, X_test, y_test)

# Compare Models
logging.info("Step 12: Comparing Models...")

# Logistic Regression
y_pred_lr = best_lr_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
y_pred_rf = best_rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# KNN
y_pred_knn = best_knn_model.predict(X_test)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# AdaBoost
y_pred_ab = best_ab_model.predict(X_test)
print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ab))

# Extra Trees
y_pred_et = best_et_model.predict(X_test)
print("Extra Trees Classification Report:")
print(classification_report(y_test, y_pred_et))

# Decision Tree
y_pred_dt = best_dt_model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# MLP Classifier
y_pred_mlp = best_mlp_model.predict(X_test)
print("MLP Classifier Classification Report:")
print(classification_report(y_test, y_pred_mlp))