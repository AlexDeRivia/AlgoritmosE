import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="HousePrices")
X = df[["Rooms", "Area_m2"]].to_numpy()
y = df["Price_Soles"].to_numpy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_alpha", random.uniform, 0.01, 100.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_alpha, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_model(individual):
    alpha = individual[0]
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return (rmse,)

toolbox.register("evaluate", eval_model)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=1.0)
toolbox.register("select", tools.selBest)  # greedy selection

# Hill climbing + población
def run_hill_climbing(pop_size=20, n_gen=50):
    pop = toolbox.population(n=pop_size)
    fits = list(map(toolbox.evaluate, pop))
    rmse_curve = []

    for gen in range(n_gen):
        offspring = []
        for ind in pop:
            clone = toolbox.clone(ind)
            toolbox.mutate(clone)
            del clone.fitness.values
            clone.fitness.values = toolbox.evaluate(clone)

            # Greedy: elige el mejor entre actual y mutado
            if clone.fitness.values[0] < ind.fitness.values[0]:
                offspring.append(clone)
            else:
                offspring.append(ind)

        pop = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        rmse_curve.append(min(fits))

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_ind.fitness.values[0], rmse_curve

# Ejecutar optimización
mejor_alpha, mejor_rmse, curva = run_hill_climbing()

print(f"Mejor alpha encontrado: {mejor_alpha[0]:.4f}")
print(f"RMSE del modelo: {mejor_rmse:.2f}")

# Curva de convergencia
plt.plot(curva)
plt.xlabel("Generación")
plt.ylabel("RMSE mínimo")
plt.title("Curva de convergencia - Hill Climbing con DEAP")
plt.grid(True)
plt.show()
