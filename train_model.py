import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class ModelTrainer:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None
        self.models = {}

    def load_data(self):
        """
        Loads the dataset from the provided path and logs basic details.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logging.info(f"Dataset loaded successfully. Shape: {self.data.shape}")

            # Verify the presence of the target column
            if 'Churn' not in self.data.columns:
                raise ValueError("Target column 'Churn' is missing from the dataset.")

            # Log the target distribution
            logging.info(f"Target distribution:\n{self.data['Churn'].value_counts(normalize=True)}")

        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.data_path}. Exiting.")
            raise

    def preprocess_data(self):
        """
        Preprocesses the data by handling missing values, dropping non-numerical columns,
        and splitting into training and testing sets. Class imbalance is addressed using BorderlineSMOTE.
        """
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Separate features and target
        self.X = self.data.drop(columns=['Churn'])
        self.y = self.data['Churn']

        # Validate that the target is binary
        if not set(self.y.unique()).issubset({0, 1}):
            raise ValueError("Target column 'Churn' must contain only binary values (0 and 1).")

        # Identify and drop non-numerical columns
        non_numerical_columns = self.X.select_dtypes(include=['object']).columns.tolist()
        if non_numerical_columns:
            logging.warning(f"Dropping non-numerical columns: {non_numerical_columns}")
            self.X = self.X.drop(columns=non_numerical_columns)

        # Ensure no non-numerical columns remain
        if self.X.select_dtypes(include=['object']).shape[1] > 0:
            raise ValueError("Non-numerical columns still present in features.")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

        # Log split details
        logging.info(f"Training size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        logging.info(f"Training class distribution:\n{y_train.value_counts(normalize=True)}")
        logging.info(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")

        # Handle class imbalance using BorderlineSMOTE
        smote = BorderlineSMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Save processed data for debugging
        X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=self.X.columns)
        X_test_df = pd.DataFrame(X_test, columns=self.X.columns)
        X_train_resampled_df.to_csv("X_train_resampled.csv", index=False)
        X_test_df.to_csv("X_test.csv", index=False)
        y_train_resampled.to_csv("y_train_resampled.csv", index=False)
        y_test.to_csv("y_test.csv", index=False)
        logging.info("Processed data saved for verification.")

        logging.info("Data preprocessing complete.")
        return X_train_resampled, X_test, y_train_resampled, y_test

    def train_models(self, X_train, y_train):
        """
        Trains multiple models on the training data and saves them with descriptive filenames.
        """
        # Define models
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=1),
            'ada_boost': AdaBoostClassifier(n_estimators=117, learning_rate=1, random_state=self.random_state),
            'extra_tree': ExtraTreesClassifier(n_estimators=130, random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=2, random_state=self.random_state),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.05, solver='adam', 
                                  learning_rate='constant', max_iter=1000, random_state=self.random_state)
        }

        # Train models and save them
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.data_path.split("/")[-1].split(".")[0]  # Extract dataset name
            model_filename = f"{dataset_name}_{name}_{timestamp}.pkl"
            joblib.dump(model, model_filename)
            logging.info(f"{name} model trained and saved as {model_filename}.")
        return self.models

    def evaluate_models(self, X_test, y_test):
        """
        Evaluates trained models on the test data and logs metrics. Results are saved to a CSV file.
        """
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

            # Log detailed metrics
            logging.info(f"\n{name} Model Evaluation:")
            logging.info(classification_report(y_test, y_pred))
            logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            if y_proba is not None:
                logging.info(f"{name} ROC-AUC Score: {roc_auc}")

            results.append({
                'Model': name,
                'Accuracy': classification_rep["accuracy"],
                'Precision': classification_rep["weighted avg"]["precision"],
                'Recall': classification_rep["weighted avg"]["recall"],
                'F1-Score': classification_rep["weighted avg"]["f1-score"],
                'ROC-AUC': roc_auc
            })

        # Save evaluation metrics
        eval_df = pd.DataFrame(results)
        eval_filename = 'model_evaluation.csv'
        eval_df.to_csv(eval_filename, index=False)
        logging.info(f"Model evaluation results saved to '{eval_filename}'.")
        logging.info(f"Evaluation preview:\n{eval_df.head()}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate machine learning models.")
    parser.add_argument('--data_path', type=str, default="final_dataset_cleaned.csv", help="Path to the dataset.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Test split ratio.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Initialize the ModelTrainer
    trainer = ModelTrainer(data_path=args.data_path, test_size=args.test_size, random_state=args.random_state)

    # Load data
    trainer.load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = trainer.preprocess_data()

    # Train models
    trainer.train_models(X_train, y_train)

    # Evaluate models
    trainer.evaluate_models(X_test, y_test)