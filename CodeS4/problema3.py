import pandas as pd
import numpy as np
import random

# Cargar la matriz de distancias
df_distancias = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="LabDistances", index_col=0)
distancias = df_distancias.to_numpy()
labs = df_distancias.index.tolist()
n_labs = len(labs)

# Calcular la distancia total de una ruta
def calcular_distancia_total(ruta):
    distancia = 0
    for i in range(n_labs - 1):
        distancia += distancias[ruta[i], ruta[i+1]]
    distancia += distancias[ruta[-1], ruta[0]]  # regreso al inicio
    return distancia

# Generar vecino intercambiando dos laboratorios
def generar_vecino(ruta):
    vecino = ruta[:]
    i, j = random.sample(range(n_labs), 2)
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

# Hill climbing principal
def hill_climbing(iteraciones=1000):
    ruta_actual = list(range(n_labs))
    random.shuffle(ruta_actual)
    mejor_distancia = calcular_distancia_total(ruta_actual)

    for _ in range(iteraciones):
        vecino = generar_vecino(ruta_actual)
        distancia_vecino = calcular_distancia_total(vecino)
        if distancia_vecino < mejor_distancia:
            ruta_actual = vecino
            mejor_distancia = distancia_vecino

    return ruta_actual, mejor_distancia

# Ejecutar el algoritmo
mejor_ruta_indices, distancia_minima = hill_climbing()
mejor_ruta_nombres = [labs[i] for i in mejor_ruta_indices]

# Mostrar resultados
print("Ruta optima:")
print(" -> ".join(mejor_ruta_nombres + [mejor_ruta_nombres[0]]))  # circuito cerrado
print(f"Distancia total: {distancia_minima:.2f} metros")
