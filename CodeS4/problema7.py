import pandas as pd
import numpy as np
import random
from collections import Counter

# Cargar datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Students")
gpas = df["GPA"].to_numpy()
skills = df["Skill"].to_numpy()
n = len(df)
num_equipos = 5
tam_equipo = 4

# Verificamos cantidad válida
assert n == num_equipos * tam_equipo, "Numero de estudiantes incompatible con tamaño de equipos"

# Función para evaluar el equilibrio de una solución
def calcular_aptitud(equipos):
    varianza_gpas = np.var([np.mean([gpas[i] for i in equipo]) for equipo in equipos])
    
    penalizacion_skills = 0
    for equipo in equipos:
        conteo = Counter([skills[i] for i in equipo])
        penalizacion_skills += sum(v - 1 for v in conteo.values() if v > 1)  # penaliza repeticiones
    
    return -(varianza_gpas + penalizacion_skills)  # negativo para maximizar (hill climbing)

# Crear vecino: intercambia dos alumnos de equipos distintos
def generar_vecino(equipos):
    nuevo = [list(equipo) for equipo in equipos]
    eq1, eq2 = random.sample(range(num_equipos), 2)
    i1 = random.randint(0, tam_equipo - 1)
    i2 = random.randint(0, tam_equipo - 1)
    nuevo[eq1][i1], nuevo[eq2][i2] = nuevo[eq2][i2], nuevo[eq1][i1]
    return nuevo

# Hill climbing
def hill_climbing(iteraciones=1000):
    indices = list(range(n))
    random.shuffle(indices)
    equipos = [indices[i*tam_equipo:(i+1)*tam_equipo] for i in range(num_equipos)]
    mejor_aptitud = calcular_aptitud(equipos)

    for _ in range(iteraciones):
        vecino = generar_vecino(equipos)
        aptitud_vecino = calcular_aptitud(vecino)
        if aptitud_vecino > mejor_aptitud:
            equipos = vecino
            mejor_aptitud = aptitud_vecino

    return equipos, -mejor_aptitud

# Ejecutar
equipos_finales, aptitud_final = hill_climbing()

# Mostrar resultados
for idx, equipo in enumerate(equipos_finales, 1):
    miembros = df.iloc[equipo]
    print(f"\nEquipo {idx}")
    print(miembros[["StudentID", "GPA", "Skill"]])

print(f"\nMetrica de desequilibrio (varianza GPA + penalizacion skills): {aptitud_final:.3f}")
