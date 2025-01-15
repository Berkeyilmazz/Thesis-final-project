import matplotlib.pyplot as plt
import pandas as pd

class OutlierFilter:
    def __init__(self, max_transactions=360, max_monetary=60000, max_item_variety=500):
        self.max_transactions = max_transactions
        self.max_monetary = max_monetary
        self.max_item_variety = max_item_variety

    def filter_outliers(self, data, save_path="filtered_data.csv"):
        """
        Filters customers based on upper thresholds for transactions, spending, and product variety.
        Saves the filtered dataset to 'filtered_data.csv' by default.

        Args:
            data (pd.DataFrame): The input dataset.
            save_path (str, optional): Path to save the filtered dataset as a CSV file. Defaults to 'filtered_data.csv'.

        Returns:
            pd.DataFrame: The filtered dataset.
        """
        filtered_data = data[
            (data['TotalTransactions'] <= self.max_transactions) &  # Keep <= 360 transactions
            (data['MonetaryValue'] <= self.max_monetary) &          # Keep <= 60,000 TL spent
            (data['ItemVariety'] < self.max_item_variety)           # Keep < 500 distinct items
        ]
        
        # Save the filtered data to a CSV file
        filtered_data.to_csv(save_path, index=False)
        print(f"Filtered dataset saved to: {save_path}")
        
        return filtered_data

    def plot_distribution(self, data, column, title):
        """
        Visualizes the distribution of a specific column in the dataset.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(data[column], bins=50, alpha=0.7, color='blue', label=column)
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()