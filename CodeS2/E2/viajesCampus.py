import numpy as np

# Presupuesto de Carlos para transporte
presupuesto = 15.0

# Precios por viaje de cada medio de transporte: bus, combi, tren
precios = np.array([2.50, 3.00, 1.80])
medios = np.array(["bus", "combi", "tren"])

# Calculamos la cantidad de viajes posibles con cada medio (redondeando hacia abajo)
viajes = np.floor(presupuesto / precios)

# array.max() devuelve el número máximo de viajes posibles
# array.argmax() devuelve el índice del medio de transporte con más viajes
max_viajes = viajes.max()
mejor_opcion = viajes.argmax()

#salida
print("Cantidad de viajes por medio de transporte:")
for medio, cantidad in zip(medios, viajes.astype(int)):
    print(f"- {medio}: {cantidad} viajes")

print(f"\nCarlos puede hacer mas viajes usando el {medios[mejor_opcion]} con {int(max_viajes)} viajes.")
