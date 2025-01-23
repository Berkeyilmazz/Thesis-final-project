import logging
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_adaboost(X_train, y_train, X_test, y_test):
    logging.info("Training AdaBoost Model with GridSearchCV...")

    # Define the AdaBoost model
    adaboost = AdaBoostClassifier(random_state=42)

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': list(range(50, 200, 10)),  # Example range for n_estimators
        'learning_rate': [0.01, 0.1, 1]           # Learning rates to test
    }

    # Set up GridSearchCV
    grid_search_ab = GridSearchCV(
        estimator=adaboost,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search_ab.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_ab_params = grid_search_ab.best_params_
    best_ab_model = grid_search_ab.best_estimator_

    logging.info(f"Best AdaBoost parameters: {best_ab_params}")

    # Evaluate the best AdaBoost model on the test set
    y_pred_ab = best_ab_model.predict(X_test)
    logging.info("Best AdaBoost Model evaluation complete.\n")

    # Classification report and confusion matrix
    print("AdaBoost Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_ab))

    print("AdaBoost Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_ab))

    # Save the trained AdaBoost model
    joblib.dump(best_ab_model, 'adaboost_model_best.pkl')
    logging.info("Best AdaBoost model saved as 'adaboost_model_best.pkl'.\n")

    return best_ab_model