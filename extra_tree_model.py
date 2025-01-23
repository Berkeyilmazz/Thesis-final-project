import logging
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_extra_tree(X_train, y_train, X_test, y_test):
    logging.info("Training Extra Trees Model with GridSearchCV...")

    # Define the Extra Trees model
    extra_tree = ExtraTreesClassifier(random_state=42)

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': list(range(50, 151, 10))  # Example range for n_estimators: 50, 60, ..., 150
    }

    # Set up GridSearchCV
    grid_search_et = GridSearchCV(
        estimator=extra_tree,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search_et.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_et_params = grid_search_et.best_params_
    best_et_model = grid_search_et.best_estimator_

    logging.info(f"Best Extra Trees parameters: {best_et_params}")

    # Evaluate the best Extra Trees model on the test set
    y_pred_et = best_et_model.predict(X_test)
    logging.info("Best Extra Trees Model evaluation complete.\n")

    # Classification report and confusion matrix
    print("Extra Trees Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_et))

    print("Extra Trees Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_et))

    # Save the trained Extra Trees model
    joblib.dump(best_et_model, 'extra_tree_model_best.pkl')
    logging.info("Best Extra Trees model saved as 'extra_tree_model_best.pkl'.\n")

    return best_et_model