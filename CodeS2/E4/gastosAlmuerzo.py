import pandas as pd

# Lista de gastos de Ana de lunes a viernes
gastos = [4.0, 3.5, 5.0, 4.2, 3.8]
dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes']

# 1. creacion del DataFrame con la columna 'Gasto' y los días como índice
df = pd.DataFrame({'Gasto': gastos}, index=dias)

# 2. Calcular gasto total y promedio (media)
gasto_total = df['Gasto'].sum()
gasto_medio = df['Gasto'].mean()

# 3. Filtrar los días en que gastó más que el promedio
dias_mayor_que_promedio = df[df['Gasto'] > gasto_medio]

# Mostrar resultados
print("Gastos diarios:")
print(df)

print(f"\nGasto total de la semana: S/ {gasto_total:.2f}")
print(f"Gasto promedio diario: S/ {gasto_medio:.2f}")

print("\nDias en que Ana gasto mss que el promedio:")
print(dias_mayor_que_promedio)
