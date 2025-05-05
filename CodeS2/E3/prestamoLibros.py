import pandas as pd

# creacion de DataFrame
data = {
    'Estudiante': ['Rosa', 'David', 'Elena', 'Mario', 'Paula'],
    'Dias_prestamo': [7, 10, 5, 12, 3]
}
df = pd.DataFrame(data)

# Mostrar el DataFrame
print("Tabla de prestamos:")
print(df)

# Calcular resumen estadístico de los días de préstamo
resumen = df['Dias_prestamo'].describe()
print("\nResumen estadistico de los dias de prestamo:")
print(resumen)

# Filtrar estudiantes que retuvieron el libro más de 8 días
filtro_mayores_8 = df[df['Dias_prestamo'] > 8]
print("\nEstudiantes que retuvieron el libro mas de 8 dias:")
print(filtro_mayores_8)
