from tabulate import tabulate
import pandas as pd
import os

file_name_a0 = "DatosMSEa0.csv"
file_name_a1 = "DatosMSEa1.csv"
current_directory = os.path.dirname(__file__)
print(current_directory)

# Check if the file exists
# Read the CSV file
dfa1 = pd.read_csv(file_name_a1)
print(dfa1)
# Read the CSV file
dfa0 = pd.read_csv(file_name_a0)
print(dfa0)
dfa1 = dfa1.dropna()
dfa0 = dfa0.dropna()

try:
    dfa0 = dfa0.drop('MSE_a1', axis=1)
except KeyError:    
    print("error")

dfa0.to_csv(file_name_a0, index=False)
dfa1.to_csv(file_name_a1, index=False)
print(tabulate(dfa0, headers='keys', tablefmt='psql'))
print(tabulate(dfa1, headers='keys', tablefmt='psql'))
