import pandas as pd
import gc

class DataSanitizer:
    def __init__(self, file_path, chunk_size=100000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def clean_data(self):
        cleaned_data = []
        for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size):
            # 1. Replace commas with periods in numeric columns
            if 'NumericColumn' in chunk.columns:
                chunk['NumericColumn'] = chunk['NumericColumn'].str.replace(',', '.').astype(float)

            # 2. Convert dates to datetime format
            chunk['TRANS_DATE'] = pd.to_datetime(chunk['TRANS_DATE'], format='%Y-%m-%d')

            # 3. Filter invalid rows
            if 'CustomerID' in chunk.columns and 'Price' in chunk.columns:
                chunk = chunk[chunk['CustomerID'].notnull()]  # Filter for valid IDs
                chunk = chunk[chunk['Price'] > 0]  # Filter for positive prices

            # Append cleaned chunk
            cleaned_data.append(chunk)

            # Free memory
            gc.collect()

        # Combine cleaned chunks into a single DataFrame
        final_data = pd.concat(cleaned_data, ignore_index=True)
        return final_data