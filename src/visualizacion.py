import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ==============================================================================
# Función para crear histogramas
# ==============================================================================

def crear_histograma_multiples(data, columnas, titulo):
    """
    Entradas:
        - data: DataFrame de pandas que contiene los datos a graficar.
        - columnas: Lista de nombres de columnas a graficar.
        - titulo: Titulo del grafico.
    Salidas:
        - Ninguna.
    Descripcion:
        Esta funcion crea un histograma para cada columna en la lista de columnas
        y los guarda en un archivo PNG.
    """

    # Crear un directorio para guardar los histogramas
    if not os.path.exists('../outputs/figuras'):
        os.makedirs('../outputs/figuras')

    # Crear un histograma para cada columna (subplots)
    fig, axes = plt.subplots(nrows=int(len(columnas)/2), ncols=2, figsize=(14, 2*len(columnas)))
    fig.suptitle(titulo, fontsize=16)
    axes = axes.flatten()
    for i, columna in enumerate(columnas):
        sns.histplot(data[columna], ax=axes[i], kde=True)
        axes[i].set_title(f'Histograma de {columna}')
        axes[i].set_xlabel(columnas[i])
        axes[i].set_ylabel('Frecuencia')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../outputs/figuras/histogramas.png')
    plt.show()

# ==============================================================================
# Función para crear gráficos de densidad (KDE)
# ==============================================================================

def crear_kde_multiples(data, columnas, titulo):
    """
    Entradas:
        - data: DataFrame de pandas que contiene los datos a graficar.
        - columnas: Lista de nombres de columnas a graficar.
        - titulo: Titulo del grafico.
    Salidas:
        - Ninguna.
    Descripcion:
        Esta funcion crea un KDE para cada columna en la lista de columnas
        y los guarda en un archivo PNG.
    """

    # Crear un directorio para guardar los KDEs
    if not os.path.exists('../outputs/figuras'):
        os.makedirs('../outputs/figuras')

    # Crear un KDE para cada columna (subplots)
    fig, axes = plt.subplots(nrows=int(len(columnas)/2), ncols=2, figsize=(14, 2*len(columnas)))
    fig.suptitle(titulo, fontsize=16)
    axes = axes.flatten()
    for i, columna in enumerate(columnas):
        sns.kdeplot(data[columna], ax=axes[i], fill=True)
        axes[i].set_title(f'KDE de {columna}')
        axes[i].set_xlabel(columnas[i])
        axes[i].set_ylabel('Frecuencia')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../outputs/figuras/kdes.png')
    plt.show()

# ==============================================================================
# Función para crear gráficos de correlación
# ==============================================================================

def crear_grafico_correlacion(data, cols, agrupar_por, titulo):
    """
    Entradas:
        - data: Es la información que se va a graficar (dataframe).
        - cols: Lista de columnas que se van a graficar (dataframe).
        - agrupar_por: Columna que se va a agrupar (dataframe).
        - titulo: Título del gráfico (string).

    Salida:
        - Un gráfico de correlación que muestra la relación entre las dos variables.
        - El gráfico se guarda como un archivo PNG en la ruta '../reporte/figuras/'
          con un nombre de archivo generado a partir del nombre de la columna del eje y.
        - El gráfico también se muestra en pantalla.
    """
    # Ordenando el dataframe por país y luego por año
    df_ordenado = data.sort_values(by=[agrupar_por])

    # Agrupando por país y año y calculando la media (esto ya lo tenías bien)
    df_agrupado = df_ordenado.groupby([agrupar_por]).mean().reset_index()

    corr_matrix = df_agrupado[cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix,
                annot=True,            # Mostrar valores
                cmap='coolwarm',       # Colormap
                linewidths=0.5,        # Ancho de bordes
                fmt=".2f",             # Formato decimal
                annot_kws={"size": 10},
                vmin=-1, 
                vmax=1)       # Rango de colores = 1

    plt.title(titulo)
    plt.savefig('../outputs/figuras/correlacion.png')
    # Mostrar el gráfico
    plt.show()

# ==============================================================================
# Función para crear un gráfico interactivo de lineas
# ==============================================================================
def crear_grafico_lineas_interactivo(data, col_car1, col_car2, lista_col_datos, etiqueta_eje_x, titulo_base):
    """
    Entradas:
        - data: Es un dataframe que contiene los datos de interés.
        - col_car1: Nombre de la columna característica que será el selector (por ejemplo: 'Country').
        - col_car2: Nombre de la columna característica del eje X (por ejemplo: 'Year').
        - lista_col_datos: Lista de nombres de columnas que se graficarán en el eje Y. (por ejemplo: ['Total Energy Consumption (TWh)', 'Per Capita Energy Use (kWh)']).
        - etiqueta_eje_x: Etiqueta para el eje X del gráfico.
        - titulo_base: Título base del gráfico.
    Salidas:
        - Gráfico interactivo de líneas que muestra el comportamiento de las variables,
          con botones para seleccionar los países.
    """
    # Crear un directorio para guardar los gráficos
    if not os.path.exists('../outputs/html'):
        os.makedirs('../outputs/html')

    # Ordenar y agrupar los datos
    df_ordenado = data.sort_values(by=[col_car1,col_car2])
    df_agrupado = df_ordenado.groupby([col_car1, col_car2]).mean().reset_index()

    # Inicializando la figura
    fig = go.Figure()

    # Añadir trazas para cada país para CADA variable, inicialmente ocultas
    for i, col_data in enumerate(lista_col_datos):
        lista_paises = df_agrupado[col_car1].unique().tolist()
        for pais in lista_paises:
            pais_data = df_agrupado[df_agrupado[col_car1] == pais]
            fig.add_trace(go.Scatter(
                x=pais_data[col_car2],
                y=pais_data[col_data],
                mode='lines+markers',
                name=pais,
                visible=(col_data == lista_col_datos[0]) # Solo la primera variable es visible al inicio
            ))

    # Crear botones para seleccionar la variable a graficar
    botones = []
    num_variables = len(lista_col_datos)
    num_paises = len(df_agrupado[col_car1].unique())

    for i, col_data in enumerate(lista_col_datos):
        visibles = [False] * (num_variables * num_paises)
        for j in range(num_paises):
            # Calcular el índice de la traza: (índice_variable * num_paises) + índice_pais
            visibles[i * num_paises + j] = True

        # Determinar la etiqueta del eje Y dinámicamente o puedes pasarla como parte de lista_col_datos
        # Por simplicidad, aquí usaremos el nombre de la columna como etiqueta
        etiqueta_eje_y = col_data

        botones.append(dict(
            label=col_data.replace("_", " ").title(), # Nombre más amigable para el botón
            method="update",
            args=[{"visible": visibles},
                  {"title": f"{titulo_base} - {col_data.replace('_', ' ').title()}",
                   "yaxis_title": etiqueta_eje_y}] # Actualiza el título del eje Y
        ))

    # Actualizar el layout del gráfico
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=botones,
            direction="down",
            x=1.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )],
        title={
            'text': f"{titulo_base} - {lista_col_datos[0].replace('_', ' ').title()}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=etiqueta_eje_x,
        yaxis_title=lista_col_datos[0], # Etiqueta inicial del eje Y
        hovermode="x unified",
        height=600
    )
    # Mostrar el gráfico
    fig.show()
    # Guardar el gráfico como un archivo HTML
    pio.write_html(fig, file='../outputs/html/grafico_lineas_interactivo.html', auto_open=True)

def crear_box_plot_interactivo(data, nombre_col1, lista_col_datos, etiqueta_eje_x, titulo_base):
    """
    Entradas:
        - data: Es la información que se va a graficar (dataframe).
        - nombre_col1: Nombre de la columna del DataFrame que se utilizará para el eje x
                       (variable categórica para los boxplots, ej: 'Country').
        - lista_col_datos: Lista de nombres de las columnas numéricas cuyos valores
                           se distribuirán en los boxplots (ej: ['Consumo_Total', 'Consumo_Per_Capita']).
        - titulo_base: Título base del gráfico (string).
        - etiqueta_eje_x: Etiqueta para el eje x del gráfico (string).

    Salida:
        - Un gráfico interactivo de boxplot que muestra la distribución de la variable numérica
          seleccionada para cada categoría de nombre_col1, con un menú desplegable para
          elegir la variable.
        - El gráfico se guarda como un archivo HTML y se muestra en pantalla.
    """
    fig = go.Figure()

    # Obtener las categorías únicas de la columna col_car1 (eje X)
    categorias_x = data[nombre_col1].unique().tolist()
    categorias_x.sort() # Opcional: ordenar las categorías para consistencia

    # Iterar sobre cada variable numérica en la lista_col_datos
    for i, col_data in enumerate(lista_col_datos):
        # Para cada variable, añadir boxplots para cada categoría en nombre_col1
        for categoria in categorias_x:
            # Filtrar los datos para la categoría actual y la columna de datos actual
            data_filtrada = data[data[nombre_col1] == categoria]

            fig.add_trace(go.Box(
                y=data_filtrada[col_data],
                x=data_filtrada[nombre_col1],
                name=categoria, # Esto es para la leyenda si fuera necesario, aunque en boxplots no siempre se usa así
                boxpoints='outliers', # Muestra los outliers
                jitter=0.3, # Distribuye los puntos ligeramente para evitar solapamiento
                pointpos=-1.8, # Posición de los puntos con respecto a la caja
                marker_color='rgba(0,128,128,0.7)', # Color de los marcadores
                line_color='rgb(8,81,156)', # Color de la línea de la caja
                fillcolor='rgba(8,81,156,0.2)', # Color de relleno de la caja
                hovertemplate=f"<b>{nombre_col1}</b>: %{{x}}<br>" +
                              f"<b>{col_data.replace('_', ' ').title()}</b>: %{{y}}<br>" +
                              "<extra></extra>", # Elimina el "trace 0" del hover
                visible=(col_data == lista_col_datos[0]) # Solo la primera variable es visible al inicio
            ))

    # Crear botones para seleccionar la variable a graficar
    botones = []
    num_variables = len(lista_col_datos)
    num_categorias = len(categorias_x) # Número de boxplots por variable

    for i, col_data in enumerate(lista_col_datos):
        visibles = [False] * (num_variables * num_categorias)
        # Calcula los índices de las trazas que corresponden a la variable actual
        for j in range(num_categorias):
            visibles[i * num_categorias + j] = True

        etiqueta_eje_y = col_data.replace("_", " ").title() # Formato amigable para el eje Y

        botones.append(dict(
            label=etiqueta_eje_y,
            method="update",
            args=[{"visible": visibles},
                  {"title": f"{titulo_base} - Distribución de {etiqueta_eje_y} por {etiqueta_eje_x}",
                   "yaxis_title": etiqueta_eje_y}]
        ))

    # Actualizar el layout del gráfico
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=botones,
            direction="down",
            x=1.1,
            y=1.15,
            xanchor="left",
            yanchor="top"
        )],
        title={
            'text': f"{titulo_base} - Distribución de {lista_col_datos[0].replace('_', ' ').title()} por {etiqueta_eje_x}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=etiqueta_eje_x,
        yaxis_title=lista_col_datos[0].replace('_', ' ').title(), # Etiqueta inicial del eje Y
        height=600,
        showlegend=False # Generalmente no necesaria para boxplots con un solo grupo de datos a la vez
    )

    # Mostrar el gráfico
    fig.show()
    # Guardar el gráfico como un archivo HTML
    pio.write_html(fig, file='../outputs/html/grafico_lineas_interactivo.html', auto_open=True)

    