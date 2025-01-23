import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_decision_tree(X_train, y_train, X_test, y_test):
    logging.info("Training Decision Tree Model with GridSearchCV...")

    # Define the Decision Tree model
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Define the hyperparameters to tune
    param_grid = {
        'min_samples_leaf': list(range(1, 7)),   # Values from 1 to 6
        'min_samples_split': list(range(2, 7))  # Values from 2 to 6
    }

    # Set up GridSearchCV
    grid_search_dt = GridSearchCV(
        estimator=decision_tree,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search_dt.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_dt_params = grid_search_dt.best_params_
    best_dt_model = grid_search_dt.best_estimator_

    logging.info(f"Best Decision Tree parameters: {best_dt_params}")

    # Evaluate the best Decision Tree model on the test set
    y_pred_dt = best_dt_model.predict(X_test)
    logging.info("Best Decision Tree Model evaluation complete.\n")

    # Classification report and confusion matrix
    print("Decision Tree Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_dt))

    print("Decision Tree Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_dt))

    # Save the trained Decision Tree model
    joblib.dump(best_dt_model, 'decision_tree_model_best.pkl')
    logging.info("Best Decision Tree model saved as 'decision_tree_model_best.pkl'.\n")

    return best_dt_model