import random
import numpy as np
import pandas as pd

df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

EXAMENES = ['A', 'B', 'C', 'D']
N_EXAMENES = len(EXAMENES)
MAX_ALUMNOS_POR_EXAMEN = 10

def crear_cromosoma():
    cromosoma = []
    for _ in range(len(alumnos)):
        pesos = [random.random() for _ in range(N_EXAMENES)]
        suma = sum(pesos)
        pesos_norm = [p / suma for p in pesos]
        cromosoma.extend(pesos_norm)
    return cromosoma

def decodificar_cromosoma(cromosoma):
    asignaciones = {e: [] for e in EXAMENES}
    contadores = {e: 0 for e in EXAMENES}
    alumnos_disponibles = list(range(len(alumnos)))

    while alumnos_disponibles:
        mejor_alumno = None
        mejor_examen = None
        mejor_valor = -1

        for alumno in alumnos_disponibles:
            idx = alumno * N_EXAMENES
            for i, examen in enumerate(EXAMENES):
                if contadores[examen] < MAX_ALUMNOS_POR_EXAMEN:
                    valor = cromosoma[idx + i]
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_alumno = alumno
                        mejor_examen = examen

        if mejor_alumno is not None:
            asignaciones[mejor_examen].append(mejor_alumno)
            contadores[mejor_examen] += 1
            alumnos_disponibles.remove(mejor_alumno)

    return asignaciones

def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    promedios = {}
    varianzas = {}

    for examen in EXAMENES:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
        varianzas[examen] = np.var(notas_examen)

    desv_promedios = np.std(list(promedios.values()))
    promedio_varianzas = np.mean(list(varianzas.values()))
    fitness = -desv_promedios - 0.1 * promedio_varianzas
    return fitness

def cruce(padre1, padre2):
    hijo = []
    for i in range(len(alumnos)):
        idx = i * N_EXAMENES
        genes = padre1[idx:idx+N_EXAMENES] if random.random() < 0.5 else padre2[idx:idx+N_EXAMENES]
        genes = [g + random.gauss(0, 0.1) for g in genes]
        genes = [max(0, g) for g in genes]
        suma = sum(genes)
        genes = [g / suma for g in genes] if suma > 0 else [1/N_EXAMENES] * N_EXAMENES
        hijo.extend(genes)
    return hijo

def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    for i in range(len(alumnos)):
        if random.random() < 0.1:
            idx = i * N_EXAMENES
            nuevos_pesos = [random.random() for _ in range(N_EXAMENES)]
            suma = sum(nuevos_pesos)
            cromosoma_mutado[idx:idx+N_EXAMENES] = [p / suma for p in nuevos_pesos]
    return cromosoma_mutado

def algoritmo_genetico(generaciones=200, tam_poblacion=120):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    mejor_global_fitness = float('-inf')
    mejor_global_cromosoma = None

    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        if fitness_scores[0][1] > mejor_global_fitness:
            mejor_global_fitness = fitness_scores[0][1]
            mejor_global_cromosoma = fitness_scores[0][0].copy()

        nueva_poblacion = []
        elite = int(tam_poblacion * 0.1)
        nueva_poblacion.extend([f[0] for f in fitness_scores[:elite]])

        while len(nueva_poblacion) < tam_poblacion:
            padre1 = random.choice(fitness_scores[:tam_poblacion//4])[0]
            padre2 = random.choice(fitness_scores[:tam_poblacion//4])[0]
            hijo = cruce(padre1, padre2)
            hijo = mutacion(hijo)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

        if gen % 30 == 0:
            print(f"Generación {gen}: Mejor fitness = {fitness_scores[0][1]:.4f}")

    return mejor_global_cromosoma

# === EJECUCIÓN PRINCIPAL ===

print("REPRESENTACIÓN REAL - 4 EXÁMENES")
print("Cromosoma: 156 valores reales (39 alumnos x 4 pesos normalizados)")
print("Gen: [0.2, 0.3, 0.1, 0.4] representa afinidad a A, B, C, D\n")

mejor_solucion = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución optimizada:")
for examen in EXAMENES:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    varianza = np.var(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos")
    print(f"  Promedio: {promedio:.2f}, Varianza: {varianza:.2f}")
    print(f"  Rango de notas: [{min(notas_examen):.0f} - {max(notas_examen):.0f}]")

print("\nAnálisis de equilibrio:")
promedios = []
for examen in EXAMENES:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))

print("Promedios por examen:", ", ".join([f"{e}={p:.2f}" for e, p in zip(EXAMENES, promedios)]))
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")
print(f"Diferencia máxima entre promedios: {max(promedios) - min(promedios):.2f}")
