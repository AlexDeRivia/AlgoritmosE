from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

app = Flask(__name__)

# Variables globales
media_global = minimo_global = maximo_global = 0
cuartiles_global = {}
categorias_global = {}

histograma_global = ''
categorias_img_global = ''
boxplot_img_global = ''
acumulado_img_global = ''

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Lectura de CSV
    df = pd.read_csv(filepath)

    # Estadísticas básicas
    media = df['Nota'].mean()
    minimo = df['Nota'].min()
    maximo = df['Nota'].max()
    cuartiles = df['Nota'].quantile([0.25, 0.5, 0.75]).to_dict()

    # Clasificación por categorías
    df['Categoria'] = pd.cut(df['Nota'], bins=[0, 13.99, 17.99, 20], labels=['C', 'B', 'A'])
    categorias = df['Categoria'].value_counts().sort_index()

    # Gráficos
    os.makedirs('static', exist_ok=True)

    # Histograma
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Nota'], bins=10, kde=True, color='skyblue')
    plt.title('Distribución de Notas')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    hist_path = os.path.join('static', 'histograma.png')
    plt.savefig(hist_path)
    plt.close()

    # Categorías A/B/C
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Categoria', data=df, order=['A', 'B', 'C'], palette='pastel')
    plt.title('Cantidad de Estudiantes por Categoría')
    cat_path = os.path.join('static', 'categorias.png')
    plt.savefig(cat_path)
    plt.close()

    # Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Nota', data=df, color='lightgreen')
    plt.title('Boxplot de Notas')
    boxplot_path = os.path.join('static', 'boxplot.png')
    plt.savefig(boxplot_path)
    plt.close()

    # Histograma apilado por categoría
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='Nota', hue='Categoria', multiple='stack', palette='Set2')
    plt.title('Distribución Acumulada por Categoría')
    stacked_path = os.path.join('static', 'acumulado.png')
    plt.savefig(stacked_path)
    plt.close()

    # Guardar en variables globales para el PDF
    global media_global, minimo_global, maximo_global, cuartiles_global
    global categorias_global, histograma_global, categorias_img_global, boxplot_img_global, acumulado_img_global

    media_global = media
    minimo_global = minimo
    maximo_global = maximo
    cuartiles_global = cuartiles
    categorias_global = categorias.to_dict()
    histograma_global = hist_path
    categorias_img_global = cat_path
    boxplot_img_global = boxplot_path
    acumulado_img_global = stacked_path

    return render_template('analisis.html',
                           media=media,
                           minimo=minimo,
                           maximo=maximo,
                           cuartiles=cuartiles,
                           categorias=categorias.to_dict(),
                           histograma=hist_path,
                           categorias_img=cat_path,
                           boxplot_img=boxplot_path,
                           acumulado_img=stacked_path)


@app.route('/reporte')
def generar_reporte():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Reporte de Análisis de Notas', ln=True, align='C')

    # Estadísticas
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Media: {round(media_global, 2)}", ln=True)
    pdf.cell(0, 10, f"Mínimo: {minimo_global}", ln=True)
    pdf.cell(0, 10, f"Máximo: {maximo_global}", ln=True)
    pdf.cell(0, 10, f"Cuartil 25%: {round(cuartiles_global[0.25], 2)}", ln=True)
    pdf.cell(0, 10, f"Mediana (50%): {round(cuartiles_global[0.5], 2)}", ln=True)
    pdf.cell(0, 10, f"Cuartil 75%: {round(cuartiles_global[0.75], 2)}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10, "Interpretación: La media indica el rendimiento promedio de los estudiantes. "
                          "Los cuartiles ayudan a entender la dispersión y el rango intercuartil de las notas.")

    # Categorías
    pdf.ln(5)
    for cat, count in categorias_global.items():
        pdf.cell(0, 10, f"Categoría {cat}: {count} estudiantes", ln=True)
    pdf.multi_cell(0, 10, "Interpretación: Esta distribución permite al docente identificar cuántos alumnos "
                          "están en cada rango de desempeño (A: excelente, B: aceptable, C: deficiente).")

    # Gráficos con interpretación
    for title, path, explanation in [
        ("Distribución de Notas", histograma_global, "Nos muestra cómo se concentran las notas."),
        ("Categorías A/B/C", categorias_img_global, "Visualiza cuántos estudiantes hay en cada grupo."),
        ("Boxplot", boxplot_img_global, "Identifica mediana, cuartiles y posibles valores atípicos."),
        ("Histograma Acumulado", acumulado_img_global, "Muestra las notas separadas por categorías acumuladas."),
    ]:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(path, w=170)
        pdf.ln(2)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, "Interpretación: " + explanation)

    # Guardar PDF
    pdf_path = os.path.join('static', 'reporte.pdf')
    pdf.output(pdf_path)

    return redirect(url_for('static', filename='reporte.pdf'))


if __name__ == '__main__':
    app.run(debug=True)
