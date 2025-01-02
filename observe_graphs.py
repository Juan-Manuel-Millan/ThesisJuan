import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

file_name_a0 = "DatosMSEa0Final.csv"
file_name_a1 = "DatosMSEa1Final.csv"

current_directory = os.path.dirname(__file__)
print(current_directory)
# Check if the file exists
# Read the CSV file
df_a1 = pd.read_csv(file_name_a1)
# Read the CSV file
df_a0 = pd.read_csv(file_name_a0)

# Let's assume df_a0 and df_a1 are DataFrames already loaded.
# The unique values of beta and outlier type
betas = [0, 0.2, 0.4, 0.6, 0.8, 1]
markers = ['o', 's', '^', 'v', 'd', 'x']  # Markers: circle, square, triangle, etc.
outlier_types = df_a0['Outlier Name'].unique()  # The unique outlier types

for outlier_type in outlier_types:
    # Filter the data by outlier type
    data_a0 = df_a0[df_a0['Outlier Name'] == outlier_type]
    data_a1 = df_a1[df_a1['Outlier Name'] == outlier_type]
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Removed sharey=True
    fig.suptitle(f'Outlier Type: {outlier_type}', fontsize=16)
    
    # Subplot for df_a0
    for beta, marker in zip(betas, markers):
        subset_a0 = data_a0[data_a0['Beta'] == beta]
        axes[0].plot(subset_a0['Proportion'], subset_a0['MSE_a0'], label=f'Beta={beta}', marker=marker)
    axes[0].set_title('MSE of a0 for different outliers')
    axes[0].set_xlabel('Outlier Proportion')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    
    # Subplot for df_a1
    for beta, marker in zip(betas, markers):
        subset_a1 = data_a1[data_a1['Beta'] == beta]
        axes[1].plot(subset_a1['Proportion'], subset_a1['MSE_a1'], label=f'Beta={beta}', marker=marker)
    axes[1].set_title('MSE of a1 for different outliers')
    axes[1].set_xlabel('Outlier Proportion')
    axes[1].legend()
    
    # Adjust the layout and show
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
