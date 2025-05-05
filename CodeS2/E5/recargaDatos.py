import numpy as np

# Datos
gb = np.array([1, 2, 5, 10])         # GB por paquete
precios = np.array([5, 9, 20, 35])   # Precios por paquete

# 1. Calcular el costo por GB
costo_por_gb = precios / gb

# Mostrar paquetes y sus costos por GB
print("Paquetes y su costo por GB:")
for i in range(len(gb)):
    print(f"- Paquete de {gb[i]} GB por S/ {precios[i]}: S/ {costo_por_gb[i]:.2f} por GB")

# 2. Encontrar el más económico por GB
indice_minimo = np.argmin(costo_por_gb)
tamaño = gb[indice_minimo]
precio = precios[indice_minimo]
costo = costo_por_gb[indice_minimo]

print("\nPaquete mas economico por GB:")
print(f"- Tamano del paquete: {tamaño} GB")
print(f"- Precio: S/ {precio}")
print(f"- Costo por GB: S/ {costo:.2f}")
