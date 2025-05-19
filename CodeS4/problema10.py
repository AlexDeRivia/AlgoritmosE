import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from deap import base, creator, tools
import matplotlib.pyplot as plt

# === Preparar datos ===
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Enrollments")
X = df[["Credits", "Prev_GPA", "Extracurricular_hours"]].to_numpy()
y = LabelEncoder().fit_transform(df["Category"])
y_cat = to_categorical(y)

X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# === DEAP setup ===
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Genotipo: [n_layers, n1, n2, n3, learning_rate]
def create_individual():
    n_layers = random.randint(1, 3)
    neurons = [random.randint(4, 32) for _ in range(3)]
    lr = random.uniform(0.0005, 0.05)
    return creator.Individual([n_layers] + neurons + [lr])

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función para construir y evaluar la red
def evaluate(ind):
    n_layers, n1, n2, n3, lr = ind
    model = Sequential()
    model.add(Dense(n1, activation="relu", input_shape=(3,)))
    if n_layers >= 2:
        model.add(Dense(n2, activation="relu"))
    if n_layers == 3:
        model.add(Dense(n3, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, verbose=0, batch_size=16)
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    acc = accuracy_score(y_true, y_pred_classes)
    return (acc,)

toolbox.register("evaluate", evaluate)

# Mutación suave
def mutate_ind(ind):
    ind[0] = min(3, max(1, ind[0] + random.choice([-1, 0, 1])))
    for i in range(1, 4):
        ind[i] = min(64, max(4, ind[i] + random.choice([-2, -1, 0, 1, 2])))
    ind[4] = max(0.0001, ind[4] + random.gauss(0, 0.002))
    return ind

toolbox.register("mutate", mutate_ind)
toolbox.register("select", tools.selBest)

# Hill climbing local
def hill_climb(ind, steps=5):
    best = creator.Individual(ind[:])
    best.fitness.values = toolbox.evaluate(best)
    for _ in range(steps):
        neighbor = creator.Individual(best[:])
        mutate_ind(neighbor)
        neighbor.fitness.values = toolbox.evaluate(neighbor)
        if neighbor.fitness.values[0] > best.fitness.values[0]:
            best = neighbor
    return best

# Evolución
def run_evolution(generations=10, pop_size=10):
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    acc_curve = []

    for gen in range(generations):
        offspring = []
        for ind in pop:
            clone = creator.Individual(ind[:])
            mutate_ind(clone)
            improved = hill_climb(clone)
            improved.fitness.values = toolbox.evaluate(improved)
            offspring.append(improved)
        pop = tools.selBest(pop + offspring, k=pop_size)
        acc_curve.append(max(ind.fitness.values[0] for ind in pop))

    best_ind = tools.selBest(pop, 1)[0]
    return best_ind, best_ind.fitness.values[0], acc_curve

# Ejecutar
mejor, acc, curva = run_evolution()

print(" Arquitectura óptima:")
print(f"Número de capas ocultas: {mejor[0]}")
print(f"Neuronas por capa: {mejor[1:4]}")
print(f"Learning rate: {mejor[4]:.5f}")
print(f" Accuracy alcanzado: {acc:.3f}")

# Gráfica
plt.plot(curva)
plt.xlabel("Generación")
plt.ylabel("Accuracy")
plt.title("Curva de convergencia - Red neuronal evolutiva")
plt.grid(True)
plt.show()
