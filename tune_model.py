import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HyperparameterTuner:
    def __init__(self, models):
        self.models = models
        self.best_params = {}

    def tune_model(self, model_name, model, param_grid, X_train, y_train):
        """
        Tunes a given model using GridSearchCV.
        """
        print(f"Tuning {model_name}...")
        try:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            self.best_params[model_name] = grid_search.best_params_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        except Exception as e:
            print(f"An error occurred while tuning {model_name}: {e}")
            return None

    def tune_all_models(self, X_train, y_train):
        """
        Tunes all models with their respective parameter grids.
        """
        param_grids = {
            'LogisticRegression': {'C': [1, 5, 10], 'penalty': ['l2'], 'class_weight': ['balanced', None]},
            'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]},
            'KNN': {'n_neighbors': [3, 5, 7]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1]},
            'ExtraTree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
            'DecisionTree': {'max_depth': [10, 20], 'min_samples_split': [2, 5]},
            'MLP': {'hidden_layer_sizes': [(50,), (100, 50)], 'activation': ['relu', 'tanh'], 'solver': ['adam']}
        }

        best_models = {}
        for name, model in self.models.items():
            print(f"\nTuning {name}...")
            best_model = self.tune_model(name, model, param_grids[name], X_train, y_train)
            if best_model:
                best_models[name] = best_model

        return best_models

# Main execution
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.neural_network import MLPClassifier

    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'ExtraTree': ExtraTreeClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'MLP': MLPClassifier(max_iter=500)
    }

    # Simulate training data (replace with your real data)
    print("Generating example data...")
    np.random.seed(42)
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.randint(0, 2, size=100)  # Binary target
    print("Example data generated.")

    # Initialize HyperparameterTuner
    tuner = HyperparameterTuner(models)

    # Tune all models
    print("\nStarting model tuning...")
    best_models = tuner.tune_all_models(X_train, y_train)

    # Display the results
    print("\nTuned Models:")
    for name, model in best_models.items():
        print(f"{name}: {model}")

    print("\nBest Parameters:")
    print(tuner.best_params)