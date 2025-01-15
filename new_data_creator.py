import pandas as pd
import numpy as np

# Generate a dummy new data set
num_samples = 100  # Number of rows
new_data = pd.DataFrame({
    'MonetaryValue': np.random.uniform(1000, 60000, num_samples),
    'TotalTransactions': np.random.randint(1, 360, num_samples),
    'ItemVariety': np.random.randint(1, 500, num_samples),
    'StoreVariety': np.random.randint(1, 20, num_samples),
    'PrevTotalMonetary': np.random.uniform(1000, 60000, num_samples),
    'PrevTotalFrequency': np.random.randint(1, 360, num_samples),
    'PrevItemVariety': np.random.randint(1, 500, num_samples),
})

new_data.to_csv('new_data.csv', index=False)
print("Dummy new_data.csv created.")