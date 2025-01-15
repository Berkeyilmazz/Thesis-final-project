import pandas as pd
import matplotlib.pyplot as plt

# Load scores from evaluation CSV file
try:
    scores_df = pd.read_csv('model_scores.csv')  
    print("Scores loaded successfully.")
except FileNotFoundError:
    print("Error: 'eve_model_scores.csv' not found. Please ensure the file exists.")
    exit()

# Check if the file has the required columns
required_columns = ['Model', 'Accuracy', 'ROC-AUC', 'F1-Score']
if not all(column in scores_df.columns for column in required_columns):
    print(f"Error: The file is missing one or more required columns: {required_columns}")
    exit()

# Convert Model names for better visualization
scores_df['Model'] = scores_df['Model'].str.replace('.pkl', '').str.replace('_', ' ').str.title()

# Visualize metrics
metrics = ['Accuracy', 'ROC-AUC', 'F1-Score']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Sort by metric value for better visual order
    sorted_df = scores_df.sort_values(by=metric, ascending=False)
    
    plt.bar(sorted_df['Model'], sorted_df[metric], color='skyblue')
    plt.title(f'Model Comparison: {metric}', fontsize=16)
    plt.ylabel(metric, fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png")  # Save each graph as an image
    plt.show()

print("Visualizations saved as PNG files.")