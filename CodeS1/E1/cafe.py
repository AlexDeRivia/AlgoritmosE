import numpy as np 

# Paso 1: Creamos un array con los precios de las 4 cafeterías
precios = np.array([2.50, 3.00, 1.75, 2.20])

# Paso 2: Calculamos cuántos cafés puede comprar Jorge con S/ 10 en cada cafetería
# Usamos np.floor para redondear hacia abajo (solo puede comprar cafés completos)
max_cafes = np.floor(10 / precios)

# Paso 3: Obtenemos la mayor cantidad de cafés que se pueden comprar y su índice
mayor_cantidad = int(max_cafes.max())            # Convertimos a entero
indice_mayor = max_cafes.argmax()                # Índice de la cafetería con más cafés

# Paso 4: Obtenemos el precio mínimo y su índice
precio_minimo = precios.min()
indice_minimo = precios.argmin()

# Lista con nombres de cafeterías según el orden del array
cafeterias = ['A', 'B', 'C', 'D']

# Paso 5: Imprimimos el resultado final
print(f"Con S/10 puedo comprar como maximo {mayor_cantidad} cafes en la cafeteria {cafeterias[indice_mayor]} (precio mínimo S/{precio_minimo:.2f} en cafeteria {cafeterias[indice_minimo]}).")
