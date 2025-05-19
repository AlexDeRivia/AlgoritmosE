import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURACIONES
# -------------------------------
OFFSET_MIN = -5
OFFSET_MAX = 5
OFFSET_STEP = 0.5
ITERACIONES = 1000

# -------------------------------
# Cargar dataset de calificaciones
# Asegúrate de que el archivo esté en la misma carpeta que este script
# -------------------------------
grades_df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Grades")

# -------------------------------
# Función de aptitud (fitness)
# Maximiza el porcentaje de aprobados penalizando si el promedio > 14
# -------------------------------
def fitness(df, offset):
    adjusted = df[["Parcial1", "Parcial2", "Parcial3"]] + offset
    adjusted = adjusted.clip(lower=0, upper=20)
    student_avg = adjusted.mean(axis=1)
    porcentaje_aprobados = (student_avg >= 11).mean()
    promedio_general = student_avg.mean()
    if promedio_general > 14:
        return porcentaje_aprobados - (promedio_general - 14)
    return porcentaje_aprobados

# -------------------------------
# Hill Climbing aleatorio
# -------------------------------
def hill_climbing_random(df, n_iter=ITERACIONES):
    best_offset = None
    best_score = -np.inf
    history = []

    for _ in range(n_iter):
        # Generar offset aleatorio múltiplo de 0.5 en el rango [-5, 5]
        offset = np.round(np.random.uniform(OFFSET_MIN, OFFSET_MAX) / OFFSET_STEP) * OFFSET_STEP
        score = fitness(df, offset)
        if score > best_score:
            best_score = score
            best_offset = offset
        history.append((offset, score))
    return best_offset, best_score, history

# -------------------------------
# Ejecutar el algoritmo
# -------------------------------
best_offset, best_score, history = hill_climbing_random(grades_df)

# -------------------------------
# Resultados finales
# -------------------------------
adjusted = grades_df[["Parcial1", "Parcial2", "Parcial3"]] + best_offset
adjusted = adjusted.clip(lower=0, upper=20)
student_avg = adjusted.mean(axis=1)

porcentaje_aprobados = (student_avg >= 11).mean() * 100
promedio_general = student_avg.mean()

print(f"Offset optimo encontrado: {best_offset}")
print(f"Porcentaje de aprobados: {porcentaje_aprobados:.2f}%")
print(f"Promedio general de la clase: {promedio_general:.2f}")

# -------------------------------
# Gráfico de la evolución del fitness
# -------------------------------
offsets, scores = zip(*history)
plt.plot(scores, color='green')
plt.title("Evolución del Fitness (Porcentaje de Aprobados)")
plt.xlabel("Iteración")
plt.ylabel("Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
