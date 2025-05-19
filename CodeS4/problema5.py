import pandas as pd
import numpy as np
import random

# Cargar datos
df = pd.read_excel("hill_climbing_datasets.xlsx", sheet_name="Tesistas")
disponibilidad = df.drop(columns=["TesistaID"]).to_numpy()
n_tesistas, n_franjas = disponibilidad.shape
n_salas = 6

# Representar una solución como: lista de (sala, franja) asignada a cada tesista
def generar_solucion_inicial():
    solucion = []
    for t in range(n_tesistas):
        franjas_disp = [i for i in range(n_franjas) if disponibilidad[t, i] == 1]
        if not franjas_disp:
            solucion.append((None, None))
        else:
            franja = random.choice(franjas_disp)
            sala = random.randint(0, n_salas - 1)
            solucion.append((sala, franja))
    return solucion

# Costo = solapamientos + penalización por franjas dispersas por sala
def calcular_costo(solucion):
    # Matriz [sala][franja] de ocupación
    ocupacion = np.zeros((n_salas, n_franjas), dtype=int)
    for sala, franja in solucion:
        if sala is not None and franja is not None:
            ocupacion[sala, franja] += 1
    
    # Solapamientos = más de una defensa en misma sala/franja
    solapamientos = np.sum(ocupacion > 1)

    # Penalización por huecos: defensas no continuas por sala
    huecos = 0
    exceso = 0
    for s in range(n_salas):
        franjas_ocupadas = [i for i in range(n_franjas) if ocupacion[s, i] > 0]
        if franjas_ocupadas:
            rango = max(franjas_ocupadas) - min(franjas_ocupadas) + 1
            huecos += rango - len(franjas_ocupadas)
            if len(franjas_ocupadas) > 4:
                exceso += len(franjas_ocupadas) - 4  # Penalizar más de 4 franjas

    return solapamientos + huecos + 2 * exceso  # exceso penaliza doble

# Generar vecino: cambiar sala o franja de un tesista
def generar_vecino(solucion):
    vecino = solucion[:]
    t = random.randint(0, n_tesistas - 1)
    franjas_disp = [i for i in range(n_franjas) if disponibilidad[t, i] == 1]
    if franjas_disp:
        franja = random.choice(franjas_disp)
        sala = random.randint(0, n_salas - 1)
        vecino[t] = (sala, franja)
    return vecino

# Hill climbing
def hill_climbing(max_iter=2000):
    actual = generar_solucion_inicial()
    costo_actual = calcular_costo(actual)

    for _ in range(max_iter):
        vecino = generar_vecino(actual)
        costo_vecino = calcular_costo(vecino)
        if costo_vecino < costo_actual:
            actual = vecino
            costo_actual = costo_vecino
            if costo_actual == 0:
                break

    return actual, costo_actual

# Ejecutar algoritmo
mejor_solucion, costo_final = hill_climbing()

# Mostrar resultados
print("Calendario final:")
for i, (sala, franja) in enumerate(mejor_solucion):
    tesista = df.loc[i, "TesistaID"]
    if sala is not None:
        print(f"{tesista} -> Sala {sala+1}, Franja F{franja+1}")
    else:
        print(f"{tesista} -> No asignado")

print(f"\nCosto final (solapamientos + huecos + exceso): {costo_final}")
