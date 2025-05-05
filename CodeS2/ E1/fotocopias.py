import numpy as np

# Presupuesto
presupuesto = 8.0

# Precios por página en cada copistería
precios = np.array([0.10, 0.12, 0.08])
copisterias = np.array(["Copisteria 1", "Copisteria 2", "Copisteria 3"])

# Cálculo de la cantidad de páginas que puede fotocopiar en cada copistería
# np.floor divide el presupuesto entre los precios y redondea hacia abajo (parte entera)
paginas = np.floor(presupuesto / precios)

# np.argmax devuelve el índice del valor máximo en el array (la copistería con más páginas)
mejor_opcion = np.argmax(paginas)

# Resultados
print("Cantidad de paginas por copisteria:")
for nombre, cantidad in zip(copisterias, paginas.astype(int)):
    print(f"- {nombre}: {cantidad} paginas")

print(f"\nMartina puede fotocopiar mas en {copisterias[mejor_opcion]} con {int(paginas[mejor_opcion])} paginas.")
