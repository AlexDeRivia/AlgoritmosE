import pandas as pd  

# Paso 1: Crear el diccionario con los datos y construir el DataFrame
datos = {
    'Estudiante': ['Ana', 'Luis', 'Maria', 'Juan', 'Carla'],
    'Horas_usadas': [3, 5, 2, 4, 1]
}

df = pd.DataFrame(datos)  # Convertimos el diccionario en un DataFrame

# Paso 2: Añadir una columna 'Costo_total' multiplicando horas * S/2.00 por hora
df['Costo_total'] = df['Horas_usadas'] * 2.0

# Paso 3: Mostrar el DataFrame completo
print("DataFrame de uso de laboratorio:\n")
print(df.head())  # Muestra las primeras filas (en este caso, todo el DataFrame)

# Paso 4: Calcular estadísticas descriptivas de la columna 'Costo_total'
estadisticas = df['Costo_total'].describe()

# Paso 5: Filtrar estudiantes con gasto mayor a S/6.00
mayores_6 = df[df['Costo_total'] > 6.0]

# Paso 6: Imprimir resumen final
promedio = estadisticas['mean']
estudiantes_mayores_6 = mayores_6['Estudiante'].tolist()

print(f"\nEl gasto promedio fue de S/ {promedio:.2f}; los estudiantes que gastaron mas de S/6.00 son:")
print(estudiantes_mayores_6)
