import pandas as pd

class FeatureSynthesizer:
    def __init__(self, reference_date):
        """
        Initializes the FeatureSynthesizer with a reference date.

        Parameters:
        reference_date (str): The reference date for feature calculations and churn labeling.
        """
        self.reference_date = pd.to_datetime(reference_date)

    def calculate_weekday_weekend(self, data):
        """
        Calculates weekday vs. weekend purchase percentages for customers.

        Parameters:
        data (DataFrame): The transaction data containing 'TRANS_DATE' and 'CustomerID'.

        Returns:
        tuple: Two Series, one for weekday percentages and one for weekend percentages.
        """
        data['Weekday'] = pd.to_datetime(data['TRANS_DATE']).dt.weekday
        data['IsWeekend'] = data['Weekday'] >= 5  # 5=Saturday, 6=Sunday

        weekend_purchases = data.groupby('CustomerID')['IsWeekend'].sum()
        total_purchases = data.groupby('CustomerID')['IsWeekend'].count()

        weekday_percentage = 100 * (total_purchases - weekend_purchases) / total_purchases
        weekend_percentage = 100 * weekend_purchases / total_purchases

        return weekday_percentage, weekend_percentage

    def synthesize_features(self, monthly_data):
        """
        Synthesizes features for churn prediction from monthly transaction data.

        Parameters:
        monthly_data (list of tuples): List of (month, DataFrame) pairs.

        Returns:
        DataFrame: A combined DataFrame with synthesized features for all customers.
        """
        customer_features = []

        for month, data in monthly_data:
            grouped = data.groupby('CustomerID').agg({
                'MonetaryValue': 'sum',
                'TotalTransactions': 'sum',
                'ItemVariety': 'sum',
                'StoreVariety': 'nunique',
                'TRANS_DATE': ['min', 'max']
            }).reset_index()

            grouped.columns = ['CustomerID', 'Monetary', 'Frequency', 'ItemVariety', 'StoreVariety', 'FirstPurchaseDate', 'LastPurchaseDate']

            # Ensure date columns are in datetime format
            grouped['FirstPurchaseDate'] = pd.to_datetime(grouped['FirstPurchaseDate'])
            grouped['LastPurchaseDate'] = pd.to_datetime(grouped['LastPurchaseDate'])

            # Add derived features
            grouped['SubscriptionDuration'] = (self.reference_date - grouped['FirstPurchaseDate']).dt.days
            grouped['Recency'] = (self.reference_date - grouped['LastPurchaseDate']).dt.days

            # Add month-specific labels
            grouped = grouped.rename(columns={
                'Monetary': f'Monetary-{month}',
                'Frequency': f'Frequency-{month}',
                'ItemVariety': f'ItemVariety-{month}',
                'StoreVariety': f'StoreVariety-{month}'
            })

            customer_features.append(grouped)

        # Combine all monthly features into one DataFrame
        final_features = pd.concat(customer_features, axis=0)
        return final_features

    def assign_churn_labels(self, data, churn_period=6):
        """
        Assigns churn labels to customers based on purchase activity.

        Parameters:
        data (DataFrame): The customer data with 'LastPurchaseDate'.
        churn_period (int): Number of months to consider for churn.

        Returns:
        DataFrame: The data with an added 'Churn' column (1=Churn, 0=Not Churn).
        """
        churn_cutoff = self.reference_date - pd.DateOffset(months=churn_period)
        data['Churn'] = data['LastPurchaseDate'] < churn_cutoff
        data['Churn'] = data['Churn'].astype(int)  # Convert to binary (1=Churn, 0=Not Churn)

        return data