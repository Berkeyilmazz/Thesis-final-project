import pandas as pd
import numpy as np


df = pd.read_csv('eve_data.csv')

# Convert 'TRANS_DATE' from string to datetime
df['TRANS_DATE'] = pd.to_datetime(df['TRANS_DATE'])

# Group by CustomerID for individual processing
grouped = df.groupby('CustomerID')

# Initialize an empty list to store the processed customer data
processed_data = []

# Iterate over each customer group
for customer_id, group in grouped:
    # Sort the data for each customer by transaction date
    group = group.sort_values(by='TRANS_DATE')
    
    # Extract the most recent transaction (used for reference date)
    reference_date = group['TRANS_DATE'].max()
    
    # Initialize placeholders for the 4-month period data (frequency, monetary, item variety, store variety)
    frequency_4months = [0, 0, 0, 0]  # Store values for the last 4 months
    monetary_4months = [0, 0, 0, 0]
    item_variety_4months = [0, 0, 0, 0]
    store_variety_4months = [0, 0, 0, 0]
    
    # Iterate over the last 4 months before the reference date and aggregate values
    for i in range(1, 5):
        month_start = reference_date - pd.DateOffset(months=i)
        month_end = month_start + pd.DateOffset(months=1)
        
        # Filter the transactions for this month
        month_data = group[(group['TRANS_DATE'] >= month_start) & (group['TRANS_DATE'] < month_end)]
        
        # Calculate and store aggregated values for this month
        frequency_4months[i-1] = month_data['TRANS_DATE'].count()
        monetary_4months[i-1] = month_data['MonetaryValue'].sum()
        item_variety_4months[i-1] = month_data['ItemVariety'].nunique()  
        store_variety_4months[i-1] = month_data['StoreVariety'].nunique()  
    
    # Calculate subscription duration (difference between the first and last purchase date)
    subscription_duration = (reference_date - group['TRANS_DATE'].min()).days
    
    # Check if the customer made a purchase in the 6 months after the reference date
    purchases_after_reference = group[group['TRANS_DATE'] > reference_date]
    churn = 1 if purchases_after_reference.empty else 0
    
    # Append the aggregated data for this customer to the list
    processed_data.append({
        'CustomerID': customer_id,
        'Frequency-1': frequency_4months[0],
        'Frequency-2': frequency_4months[1],
        'Frequency-3': frequency_4months[2],
        'Frequency-4': frequency_4months[3],
        'Monetary-1': monetary_4months[0],
        'Monetary-2': monetary_4months[1],
        'Monetary-3': monetary_4months[2],
        'Monetary-4': monetary_4months[3],
        'ItemVariety-1': item_variety_4months[0],
        'ItemVariety-2': item_variety_4months[1],
        'ItemVariety-3': item_variety_4months[2],
        'ItemVariety-4': item_variety_4months[3],
        'StoreVariety-1': store_variety_4months[0],
        'StoreVariety-2': store_variety_4months[1],
        'StoreVariety-3': store_variety_4months[2],
        'StoreVariety-4': store_variety_4months[3],
        'SubscriptionDuration': subscription_duration,
        'Churn': churn,
    })

# Convert the processed data into a DataFrame
final_dataset = pd.DataFrame(processed_data)

# Save the final processed dataset
final_dataset.to_csv('4_month_dataset.csv', index=False)

print("Data has been processed and saved.")