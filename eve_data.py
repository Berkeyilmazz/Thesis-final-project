import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker and parameters (since cannot upload the original data due to privacy agreements.)
fake = Faker()
num_records = 10000  
# Generate fake data
data = {
    "CustomerID": [fake.uuid4() for _ in range(num_records)],
    "TRANS_DATE": [fake.date_between(start_date='-3y', end_date='today') for _ in range(num_records)],
    "TotalTransactions": [random.randint(1, 400) for _ in range(num_records)],
    "MonetaryValue": [round(random.uniform(10, 70000), 2) for _ in range(num_records)],
    "ItemVariety": [random.randint(1, 600) for _ in range(num_records)],
    "Churn": [random.choice([0, 1]) for _ in range(num_records)]
}

# Convert to DataFrame
eve_data = pd.DataFrame(data)

# Save to CSV
eve_data.to_csv("eve_data.csv", index=False)
print("Eve dataset saved as 'eve_data.csv'.")