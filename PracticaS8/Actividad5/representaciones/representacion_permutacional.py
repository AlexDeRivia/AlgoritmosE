import random
import numpy as np
import pandas as pd
import os
import pickle

# Crear carpeta de resultados
os.makedirs('Actividad5/resultados', exist_ok=True)

# Cargar datos
df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

def crear_cromosoma():
    indices = list(range(39))
    random.shuffle(indices)
    return indices

def decodificar_cromosoma(cromosoma):
    return {
        'A': cromosoma[0:13],
        'B': cromosoma[13:26],
        'C': cromosoma[26:39]
    }

def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    
    promedios = {}
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
    
    desv_promedios = np.std(list(promedios.values()))
    
    bonus_diversidad = 0
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        if max(notas_examen) - min(notas_examen) > 5:
            bonus_diversidad += 0.1
    
    fitness = -desv_promedios + bonus_diversidad
    return fitness

def mutacion_intercambio(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    if random.random() < 0.3:
        pos1 = random.randint(0, 38)
        pos2 = random.randint(0, 38)
        cromosoma_mutado[pos1], cromosoma_mutado[pos2] = cromosoma_mutado[pos2], cromosoma_mutado[pos1]
    return cromosoma_mutado

def mutacion_inversion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    if random.random() < 0.2:
        inicio = random.randint(0, 36)
        longitud = random.randint(2, min(5, 39 - inicio))
        segmento = cromosoma_mutado[inicio:inicio + longitud]
        segmento.reverse()
        cromosoma_mutado[inicio:inicio + longitud] = segmento
    return cromosoma_mutado

def algoritmo_genetico(generaciones=50, tam_poblacion=30):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    historial_fitness = []

    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        historial_fitness.append(fitness_scores[0][1])
        nueva_poblacion = []

        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])

        while len(nueva_poblacion) < tam_poblacion:
            padre1 = random.choice(poblacion[:tam_poblacion//3])
            padre2 = random.choice(poblacion[:tam_poblacion//3])
            hijo = mutacion_intercambio(padre1)
            hijo = mutacion_inversion(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if gen % 10 == 0:
            print(f"Generación {gen}: Mejor fitness = {fitness_scores[0][1]:.4f}")

    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma, historial_fitness

# ----------- EJECUCIÓN PRINCIPAL -----------

print("REPRESENTACIÓN PERMUTACIONAL")
print("Problema: Secuenciar alumnos para asignación ordenada a exámenes")
print("Cromosoma: Permutación de 39 índices de alumnos")
print("Decodificación: Posiciones [0-12] -> Examen A, [13-25] -> Examen B, [26-38] -> Examen C\n")

mejor_solucion, historial = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nAsignación final por orden de secuencia:")
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    print(f"\nExamen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Secuencia de alumnos:")
    for i, idx in enumerate(indices):
        print(f"    Posición {i+1}: {alumnos[idx]} (Nota: {notas[idx]})")
        if i >= 4:
            print("    ... (mostrando primeros 5)")
            break

print("\nEstadísticas finales:")
promedios = []
rangos = []
for examen in ['A', 'B', 'C']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))
    rangos.append(max(notas_examen) - min(notas_examen))

print(f"Promedios: A={promedios[0]:.2f}, B={promedios[1]:.2f}, C={promedios[2]:.2f}")
print(f"Rangos de notas: A={rangos[0]:.0f}, B={rangos[1]:.0f}, C={rangos[2]:.0f}")
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")

print("\nEvolución del algoritmo:")
print(f"Fitness inicial: {historial[0]:.4f}")
print(f"Fitness final: {historial[-1]:.4f}")
print(f"Mejora total: {((historial[-1] - historial[0]) / abs(historial[0]) * 100):.1f}%")

# ----------- GUARDAR RESULTADOS -----------

with open('Actividad5/resultados/asignaciones_permutacional.pkl', 'wb') as f:
    pickle.dump(asignaciones_finales, f)

with open('Actividad5/resultados/historial_permutacional.pkl', 'wb') as f:
    pickle.dump(historial, f)
