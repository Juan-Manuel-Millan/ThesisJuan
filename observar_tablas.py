from tabulate import tabulate
import pandas as pd
import os
nombre_archivoa0 = "DatosMSEa0.csv"
nombre_archivoa1 = "DatosMSEa1.csv"
directorio_actual = os.path.dirname(__file__)
print(directorio_actual)
# Comprobar si el archivo existe
# Leer el archivo CSV
dfa1 = pd.read_csv(nombre_archivoa1)
print(dfa1)
# Leer el archivo CSV
dfa0 = pd.read_csv(nombre_archivoa0)
print(dfa0)
dfa1 = dfa1.dropna()
dfa0 = dfa0.dropna()
try:
    dfa0 = dfa0.drop('MSE_a1', axis=1)
except KeyError:    
    print("error")

dfa0.to_csv(nombre_archivoa0, index=False)
dfa1.to_csv(nombre_archivoa1, index=False)
print(tabulate(dfa0, headers='keys', tablefmt='psql'))
print(tabulate(dfa1, headers='keys', tablefmt='psql'))