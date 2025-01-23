import logging
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_mlp_classifier(X_train, y_train, X_test, y_test):
    logging.info("Training MLP Classifier with GridSearchCV...")

    # Define the MLP Classifier
    mlp = MLPClassifier(max_iter=1000, random_state=42)

    # Define the hyperparameters to tune
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
        'activation': ['tanh', 'relu'],         # Activation functions
        'solver': ['sgd', 'adam'],              # Optimization solvers
        'alpha': [0.0001, 0.05],                # L2 regularization term
        'learning_rate': ['constant', 'adaptive']  # Learning rate schedules
    }

    # Set up GridSearchCV
    grid_search_mlp = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=5,                # 5-fold cross-validation
        scoring='accuracy',  # Evaluation metric
        verbose=2,
        n_jobs=-1            # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search_mlp.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_mlp_params = grid_search_mlp.best_params_
    best_mlp_model = grid_search_mlp.best_estimator_

    logging.info(f"Best MLP Classifier parameters: {best_mlp_params}")

    # Evaluate the best MLP Classifier model on the test set
    y_pred_mlp = best_mlp_model.predict(X_test)
    logging.info("Best MLP Classifier evaluation complete.\n")

    # Classification report and confusion matrix
    print("MLP Classifier Classification Report (Best Model):")
    print(classification_report(y_test, y_pred_mlp))

    print("MLP Classifier Confusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_mlp))

    # Save the trained MLP Classifier model
    joblib.dump(best_mlp_model, 'mlp_classifier_model_best.pkl')
    logging.info("Best MLP Classifier model saved as 'mlp_classifier_model_best.pkl'.\n")

    return best_mlp_model