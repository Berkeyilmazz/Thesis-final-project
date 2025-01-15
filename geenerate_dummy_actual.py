import pandas as pd
import numpy as np

# Generate dummy data also because of the privacy agreement I made with EveShop.
dummy_actuals = pd.DataFrame({
    'CustomerID': [f'id_{i}' for i in range(1000)],  # since it will be fake data random number will be used for the range for testing
    'ActualChurn': np.random.randint(0, 2, size=1000) # Random churn labels (0 or 1)
})

# Save to CSV
dummy_actuals.to_csv('actual_outcomes.csv', index=False)
print("Dummy actual_outcomes.csv created.")