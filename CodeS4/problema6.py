import pandas as pd
import numpy as np
import random

# Cargar datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="ExamQuestions")
dificultades = df["Difficulty"].to_numpy()
tiempos = df["Time_min"].to_numpy()
n = len(df)

# Función de aptitud
def calcular_puntaje(bitstring):
    tiempo_total = sum(t * b for t, b in zip(tiempos, bitstring))
    dificultad_total = sum(d * b for d, b in zip(dificultades, bitstring))
    
    if tiempo_total > 90:
        return -np.inf  # penalización si excede tiempo
    if not (180 <= dificultad_total <= 200):
        return -np.inf  # penalización si no está en rango de dificultad
    return dificultad_total  # objetivo: dificultad más alta dentro del rango permitido

# Vecino: cambiar 1 bit
def generar_vecino(bitstring):
    vecino = bitstring[:]
    i = random.randint(0, n - 1)
    vecino[i] = 1 - vecino[i]
    return vecino

# Hill Climbing
def hill_climbing(iteraciones=1000):
    actual = [random.randint(0, 1) for _ in range(n)]
    mejor_score = calcular_puntaje(actual)

    for _ in range(iteraciones):
        vecino = generar_vecino(actual)
        score_vecino = calcular_puntaje(vecino)
        if score_vecino > mejor_score:
            actual = vecino
            mejor_score = score_vecino

    return actual, mejor_score

# Ejecutar
seleccion, mejor_dificultad = hill_climbing()
preguntas_seleccionadas = df["QuestionID"][np.array(seleccion) == 1].tolist()
tiempo_total = sum(tiempos[i] for i in range(n) if seleccion[i] == 1)

# Mostrar resultados
print("Preguntas seleccionadas:", preguntas_seleccionadas)
print("Dificultad total:", mejor_dificultad)
print("Tiempo total:", tiempo_total)
