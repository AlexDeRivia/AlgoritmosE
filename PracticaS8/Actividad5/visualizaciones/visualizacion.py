import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

sns.set(style="whitegrid")

# Ruta base
ruta = "Actividad5/resultados"

# Cargar notas originales
df = pd.read_csv("notas_1u.csv")
notas = df['Nota'].tolist()

# Función para cargar historial y asignaciones
def cargar_datos(nombre):
    with open(os.path.join(ruta, f"historial_{nombre}.pkl"), "rb") as f:
        historial = pickle.load(f)
    with open(os.path.join(ruta, f"asignaciones_{nombre}.pkl"), "rb") as f:
        asignaciones = pickle.load(f)
    return historial, asignaciones

# Representaciones a analizar
nombres = ["real", "binaria", "permutacional"]
datos = {}

for nombre in nombres:
    try:
        historial, asignaciones = cargar_datos(nombre)
        datos[nombre] = {
            "historial": historial,
            "asignaciones": asignaciones
        }
    except FileNotFoundError:
        print(f"[⚠️] No se encontraron archivos para: {nombre}")
        continue

# === GRAFICO 1: Evolución del fitness por generación ===
plt.figure(figsize=(10, 6))
for nombre in datos:
    plt.plot(datos[nombre]["historial"], label=nombre.capitalize())
plt.title("Evolución del Fitness por Generación")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ruta, "grafico_1_fitness.png"))
plt.show()

# === GRAFICO 2: Histograma de notas por examen (A, B, C) usando la representación real ===
asignaciones_real = datos["real"]["asignaciones"]
plt.figure(figsize=(10, 6))

for examen, color in zip(['A', 'B', 'C'], ['#4caf50', '#2196f3', '#ff9800']):
    notas_examen = [notas[i] for i in asignaciones_real[examen]]
    plt.hist(notas_examen, bins=10, alpha=0.6, label=f"Examen {examen}", color=color)

plt.title("Histograma de Notas por Examen (Representación Real)")
plt.xlabel("Nota")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ruta, "grafico_2_histograma_notas.png"))
plt.show()

# === GRAFICO 3: Boxplot de distribución de notas por representación ===
plt.figure(figsize=(10, 6))
data_plot = []
labels = []

for nombre in nombres:
    if nombre not in datos:
        continue
    asignaciones = datos[nombre]["asignaciones"]
    notas_repr = []
    for grupo in ['A', 'B', 'C']:
        notas_grupo = [notas[i] for i in asignaciones[grupo]]
        notas_repr.extend(notas_grupo)
    data_plot.append(notas_repr)
    labels.append(nombre.capitalize())

plt.boxplot(data_plot, labels=labels)
plt.title("Distribución de Notas por Representación")
plt.ylabel("Nota")
plt.tight_layout()
plt.savefig(os.path.join(ruta, "grafico_3_boxplot_distribuciones.png"))
plt.show()
