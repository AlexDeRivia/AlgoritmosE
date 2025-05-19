import pandas as pd
import numpy as np
import random

# Cargar los datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Projects")
costos = df["Cost_Soles"].to_numpy()
beneficios = df["Benefit_Soles"].to_numpy()
presupuesto = 10000
n = len(costos)

# Función de aptitud
def calcular_beneficio(bitstring):
    costo_total = sum(c * b for c, b in zip(costos, bitstring))
    if costo_total > presupuesto:
        return -np.inf  # penalización si excede el presupuesto
    return sum(b * v for b, v in zip(beneficios, bitstring))

# Generar vecino: voltea 1 bit
def generar_vecino(bitstring):
    vecino = bitstring[:]
    i = random.randint(0, n - 1)
    vecino[i] = 1 - vecino[i]
    return vecino

# Hill Climbing
def hill_climbing(iteraciones=1000):
    actual = [random.randint(0, 1) for _ in range(n)]
    mejor_beneficio = calcular_beneficio(actual)

    for _ in range(iteraciones):
        vecino = generar_vecino(actual)
        beneficio_vecino = calcular_beneficio(vecino)
        if beneficio_vecino > mejor_beneficio:
            actual = vecino
            mejor_beneficio = beneficio_vecino

    return actual, mejor_beneficio

# Ejecutar
seleccion, beneficio_total = hill_climbing()
proyectos_seleccionados = df["ProjectID"][np.array(seleccion) == 1].tolist()

# Mostrar resultados
print("Proyectos seleccionados:", proyectos_seleccionados)
print("Beneficio total:", beneficio_total)
print("Costo total:", sum(costos[i] for i in range(n) if seleccion[i] == 1))
