import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def train_logistic_regression(X_train, y_train, X_test, y_test):
    logging.info("Tuning Logistic Regression Model with GridSearchCV...")

    # Define the Logistic Regression model
    logistic_model = LogisticRegression(max_iter=1000, random_state=42, solver='saga')  # Use 'saga' for elasticnet support

    # Define the hyperparameters to tune
    logistic_param_grid = {
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Regularization strength
        'penalty': ['l2', 'elasticnet', 'none'],       # Regularization type
        'class_weight': ['balanced', None],           # Class weight options
        'l1_ratio': [0.5] if 'elasticnet' in ['penalty'] else [None]  # L1 ratio only for elasticnet
    }

    # Set up GridSearchCV
    grid_search_lr = GridSearchCV(
        estimator=logistic_model,
        param_grid=logistic_param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the resampled training data
    grid_search_lr.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_lr_params = grid_search_lr.best_params_
    best_lr_model = grid_search_lr.best_estimator_

    logging.info(f"Best Logistic Regression parameters: {best_lr_params}")

    # Evaluate the best Logistic Regression model on the test set
    y_pred_lr = best_lr_model.predict(X_test)
    logging.info("Best Logistic Regression Model evaluation complete.\n")

    # Classification report and confusion matrix
    print("Logistic Regression Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_lr))

    print("Logistic Regression Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_lr))

    # Save the trained Logistic Regression model
    joblib.dump(best_lr_model, 'logistic_model_best.pkl')
    logging.info("Best Logistic Regression model saved as 'logistic_model_best.pkl'.\n")

    return best_lr_model