import pandas as pd
import numpy as np
import random

# Leer la hoja de disponibilidad
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="MentorAvailability")

# Quitar columna MentorID y trabajar con los slots como matriz binaria
availability = df.drop(columns=["MentorID"]).to_numpy()
n_mentores, n_slots = availability.shape

# Generar una solución inicial válida: elegir un bloque de 2 slots continuos disponibles
def generar_solucion_inicial():
    solucion = []
    for mentor in range(n_mentores):
        horarios_validos = []
        for i in range(n_slots - 1):
            if availability[mentor, i] == 1 and availability[mentor, i + 1] == 1:
                horarios_validos.append(i)
        if horarios_validos:
            solucion.append(random.choice(horarios_validos))
        else:
            solucion.append(None)
    return solucion

# Función de costo: cuenta cuántos choques hay (mentores asignados al mismo bloque)
def calcular_choques(solucion):
    contador = {}
    for bloque in solucion:
        if bloque is not None:
            contador[bloque] = contador.get(bloque, 0) + 1
    return sum(c - 1 for c in contador.values() if c > 1)

# Función de vecindad: cambia el horario de un mentor aleatorio
def generar_vecino(solucion):
    vecino = solucion[:]
    mentor = random.randint(0, n_mentores - 1)
    horarios_validos = []
    for i in range(n_slots - 1):
        if availability[mentor, i] == 1 and availability[mentor, i + 1] == 1:
            horarios_validos.append(i)
    if horarios_validos:
        vecino[mentor] = random.choice(horarios_validos)
    return vecino

# Hill Climbing
def hill_climbing(max_iter=1000):
    actual = generar_solucion_inicial()
    costo_actual = calcular_choques(actual)

    for _ in range(max_iter):
        vecino = generar_vecino(actual)
        costo_vecino = calcular_choques(vecino)

        if costo_vecino < costo_actual:
            actual = vecino
            costo_actual = costo_vecino

        if costo_actual == 0:
            break

    return actual, costo_actual

# Ejecutar algoritmo
mejor_solucion, choques = hill_climbing()

# Resultados
print("Solucion final (bloques asignados por mentor):")
for idx, bloque in enumerate(mejor_solucion):
    print(f"Mentor {df.loc[idx, 'MentorID']}: Slot {bloque} y Slot {bloque+1}" if bloque is not None else f"Mentor {df.loc[idx, 'MentorID']}: No tiene bloque valido")

print(f"\nChoques totales: {choques}")
