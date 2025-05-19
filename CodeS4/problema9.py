import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from deap import base, creator, tools
import random
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Emails")
X = df[[f"Feature{i}" for i in range(1, 6)]].to_numpy()
y = df["Spam"].to_numpy()

# Separar en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("weight_or_threshold", random.uniform, -2.0, 2.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.weight_or_threshold, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluación: F1-score
def evaluate(ind):
    weights = np.array(ind[:5])
    threshold = ind[5]
    scores = X_val @ weights
    preds = (scores > threshold).astype(int)
    return (f1_score(y_val, preds),)

toolbox.register("evaluate", evaluate)

# Mutación gaussiana
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=1.0)
toolbox.register("select", tools.selBest)

# Hill climbing local sobre un individuo
def hill_climb(ind, steps=10):
    best = ind[:]
    best_fitness = toolbox.evaluate(best)[0]
    for _ in range(steps):
        neighbor = creator.Individual(best[:])
        toolbox.mutate(neighbor)
        fit = toolbox.evaluate(neighbor)[0]
        if fit > best_fitness:
            best = neighbor
            best_fitness = fit
    return best

# Evolución con hill climbing por individuo
def run_evolution(pop_size=20, gens=40):
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    f1_curve = []

    for gen in range(gens):
        offspring = []
        for ind in pop:
            mutated = creator.Individual(ind[:])
            toolbox.mutate(mutated)
            mutated = hill_climb(mutated)  # Hill climbing local
            mutated.fitness.values = toolbox.evaluate(mutated)
            offspring.append(mutated)

        pop = tools.selBest(offspring + pop, k=pop_size)
        best_f1 = max(ind.fitness.values[0] for ind in pop)
        f1_curve.append(best_f1)

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_ind.fitness.values[0], f1_curve

# Ejecutar
mejor_ind, mejor_f1, f1_curve = run_evolution()

# Resultados
pesos = mejor_ind[:5]
umbral = mejor_ind[5]
print(f" Mejores pesos: {np.round(pesos, 3)}")
print(f"Umbral óptimo: {umbral:.3f}")
print(f"F1-score alcanzado: {mejor_f1:.3f}")

# Curva F1 vs generación
plt.plot(f1_curve)
plt.xlabel("Generación")
plt.ylabel("F1-score")
plt.title("Convergencia F1-score (Spam Filter)")
plt.grid(True)
plt.show()
