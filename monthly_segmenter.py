import pandas as pd
import os

class MonthlySegmenter:
    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    def split_by_month(self, data):
        """
        Splits data into monthly segments and calculates cumulative metrics.
        """
        monthly_data = []
        current_date = self.start_date

        while current_date <= self.end_date:
            # Filter data for the current month
            next_date = current_date + pd.offsets.MonthEnd(1)
            month_data = data[(data['TRANS_DATE'] >= current_date) & (data['TRANS_DATE'] < next_date)]

            # Calculate cumulative metrics
            month_data = month_data.copy()  # Avoid SettingWithCopyWarning
            month_data['PrevTotalMonetary'] = month_data['MonetaryValue'].cumsum()
            month_data['PrevTotalFrequency'] = month_data['TotalTransactions'].cumsum()
            month_data['PrevItemVariety'] = month_data['ItemVariety'].cumsum()
            # Update the split_by_month method
            month_data['StoreVariety'] = month_data.groupby('CustomerID')['StoreID'].transform('nunique')

            # Save month data for aggregation
            monthly_data.append((current_date.strftime('%Y-%m'), month_data))
            current_date = next_date

        return monthly_data

   
    def save_monthly_data(self, monthly_data, output_dir):
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
       
        # Save files
        for month, data in monthly_data:
            file_name = f"{output_dir}/monthly_data_{month}.csv"
            data.to_csv(file_name, index=False)
            print(f"Saved {file_name}")