import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_knn(X_train, y_train, X_test, y_test):
    logging.info("Training KNN Model with GridSearchCV...")

    # Define the KNN model
    knn = KNeighborsClassifier()

    # Define the hyperparameters to tune
    param_grid = {
        'n_neighbors': list(range(1, 21))  # Values from 1 to 20
    }

    # Set up GridSearchCV
    grid_search_knn = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search_knn.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_knn_params = grid_search_knn.best_params_
    best_knn_model = grid_search_knn.best_estimator_

    logging.info(f"Best KNN parameters: {best_knn_params}")

    # Evaluate the best KNN model on the test set
    y_pred_knn = best_knn_model.predict(X_test)
    logging.info("Best KNN Model evaluation complete.\n")

    # Classification report and confusion matrix
    print("KNN Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_knn))

    print("KNN Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_knn))

    # Save the trained KNN model
    joblib.dump(best_knn_model, 'knn_model_best.pkl')
    logging.info("Best KNN model saved as 'knn_model_best.pkl'.\n")

    return best_knn_model