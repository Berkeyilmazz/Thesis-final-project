import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FeatureImportanceAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def get_importance(self):
        """
        Extracts feature importance from tree-based models or coefficients for linear models.
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return abs(self.model.coef_[0])  # Absolute value of coefficients for interpretability
        else:
            raise ValueError("Feature importance is not available for this model type.")

    def remove_zero_importance_features(self, importance):
        """
        Removes features with zero importance or constant values.
        """
        # Filter out features that have zero importance
        filtered_importance = importance > 0.0001  # Threshold to remove very low importance features
        return filtered_importance

    def plot_importance(self):
        """
        Visualizes feature importance as a bar plot.
        """
        importance = self.get_importance()
        
        if len(importance) != len(self.feature_names):
            raise ValueError("Mismatch between number of features and importances.")
        
        # Remove features with zero or near-zero importance
        filtered_importance = self.remove_zero_importance_features(importance)
        
        importance_df = pd.DataFrame({
            'Feature': np.array(self.feature_names)[filtered_importance],
            'Importance': importance[filtered_importance]
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.gca().invert_yaxis()
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

        return importance_df
