import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Importar visualización personalizada
import src.visualizacion as vis

# ==============================================================================
# Función para mensualizar datos (meustrear 12 puntos mensuales)
# ==============================================================================

def mensualizar_datos(df, columna_valor, window=12):
    """
    A partir de múltiples registros por año, genera una serie mensual sintética por país-año usando media rodante
    y luego muestrea 12 puntos de forma equidistante.

    Entradas:
        - df: DataFrame original con columnas 'Country', 'Year', y columna de valor.
        - columna_valor: nombre de la columna que contiene el valor a suavizar.
        - window: tamaño de la ventana rodante (12 para meses).

    Salida:
        - df_expandido: DataFrame con columna 'ds' mensual y columna 'y' para Prophet.
    """
    df_expandido = []

    for (country, year), grupo in df.groupby(['Country', 'Year']):
        # Asegurarse de tener suficientes registros para calcular al menos una media móvil
        if len(grupo) < window:
            continue

        # Media rodante sobre los valores
        valores_suavizados = grupo[columna_valor].rolling(window=window, min_periods=window).mean().dropna().reset_index(drop=True)

        # Si después de la media móvil no hay suficientes puntos para el muestreo, saltar
        if len(valores_suavizados) < window:
            continue

        # Si valores_suavizados tiene más de 12 elementos, seleccionamos 12 de forma equidistante.
        if len(valores_suavizados) > 12:
            indices = (np.arange(12) * (len(valores_suavizados) - 1) / 11).astype(int)
            valores_finales_y = valores_suavizados.iloc[indices].reset_index(drop=True)
        else: # Si ya tiene 12 o menos (pero ya filtramos los <12)
            valores_finales_y = valores_suavizados

        # Crear fechas mensuales para el año
        fechas = pd.date_range(start=f'{year}-01-01', periods=12, freq='MS') # Siempre 12 fechas

        # Asegurarse de que valores_finales_y tenga la misma longitud que fechas (12)
        if len(valores_finales_y) != 12:
            
            print(f"Advertencia: El muestreo para {country}-{year} no resultó en 12 valores. Se reajustará.")
            
            if len(valores_finales_y) > 12:
                valores_finales_y = valores_finales_y.iloc[:12]
            elif len(valores_finales_y) < 12:
                valores_finales_y = pd.concat([valores_finales_y,
                                                pd.Series([valores_finales_y.iloc[-1]] * (12 - len(valores_finales_y)))],
                                               ignore_index=True)

        df_temp = pd.DataFrame({
            'ds': fechas,
            'y': valores_finales_y,
            'Country': country
        })

        df_expandido.append(df_temp)

    return pd.concat(df_expandido, ignore_index=True)

# ==============================================================================
# Función para entrenar modelos Prophet y generar pronósticos
# ==============================================================================

def entrenar_modelos_prophet(df_prophet, nombre_variable, unidad, prefijo,
                             horizonte_tiempo = 5, 
                             ruta_salida='../outputs/prophet_forecasts',
                             ruta_modelos='../models/prophet_models'):
    """
    Entrena modelos Prophet por país, guarda gráficos y serializa los modelos entrenados.

    Parámetros:
    - df_prophet: DataFrame con columnas ['ds', 'y', 'Country']
    - nombre_variable: Nombre de la variable a pronosticar (ej. 'Consumo Total de Energía')
    - unidad: Unidad de la variable (ej. 'TWh')
    - prefijo: Prefijo para los nombres de los archivos guardados (ej. 'total_energy')
    - horizonte_tiempo: int, número de años a predecir
    - ruta_salida: str, carpeta para guardar los gráficos
    - ruta_modelos: str, carpeta para guardar los modelos Prophet

    Retorna:
    - prophet_models: dict, modelos Prophet entrenados por país
    - predicciones: dict, predicciones generadas por cada modelo
    """

    os.makedirs(ruta_salida, exist_ok=True)
    os.makedirs(ruta_modelos, exist_ok=True)

    prophet_models = {}
    predicciones = {}

    for country in df_prophet['Country'].unique():
        print(f"Entrenando Prophet para {country}...")

        df_country = df_prophet[df_prophet['Country'] == country].copy()

        m = Prophet(
            seasonality_mode='additive',
            changepoint_prior_scale=0.75,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            n_changepoints=30
        )

        m.fit(df_country)
        prophet_models[country] = m

        future = m.make_future_dataframe(periods=horizonte_tiempo, freq='Y')
        pred = m.predict(future)
        predicciones[country] = pred

        # Gráfico de predicción
        fig = m.plot(pred, include_legend=True)
        add_changepoints_to_plot(fig.gca(), m, pred)
        plt.title(f'Pronóstico de {nombre_variable} - {country}')
        plt.xlabel('Año')
        plt.ylabel(f'{nombre_variable} {unidad}')
        plt.savefig(f'{ruta_salida}/prophet_forecast_{prefijo}_{country.replace(" ", "_")}.png', dpi=300)
        plt.show()
        plt.close(fig)

        # Gráfico de componentes
        fig2 = m.plot_components(pred)
        plt.suptitle(f'Componentes del Pronóstico - {country}', fontsize=14)
        plt.savefig(f'{ruta_salida}/prophet_components_{prefijo}_{country.replace(" ", "_")}.png', dpi=300)
        plt.show()
        plt.close(fig2)

        # Guardar el modelo entrenado
        with open(f'{ruta_modelos}/prophet_model_{prefijo}_{country.replace(" ", "_")}.joblib', 'wb') as f:
            joblib.dump(m, f)

    print("Modelos Prophet entrenados, visualizaciones y archivos .joblib guardados.")
    return prophet_models, predicciones

# ==============================================================================
# Función de remuestreo de datos mensuales para XGBoost
# ==============================================================================

def remuestrear_datos_mensuales(df, columnas_caracteristicas, window=12):
    """
    Genera una serie mensual sintética por país-año usando media rodante y muestreo equidistante para múltiples columnas.

    Parámetros:
        - df: DataFrame original con columnas 'Country', 'Year', y las columnas de características.
        - columnas_caracteristicas: Lista de nombres de columnas a suavizar (ej. ['Total Energy Consumption (TWh)', 'Carbon Emissions (Million Tons)']).
        - window: Tamaño de la ventana rodante (12 para simular meses).

    Retorna:
        - df_mensual: DataFrame con columna 'ds' (fechas mensuales), las columnas suavizadas, y 'Country'.
    """
    df_mensual = []

    for (country, year), grupo in df.groupby(['Country', 'Year']):
        # Asegurarse de que haya suficientes registros para la ventana
        if len(grupo) < window:
            continue

        # Crear un diccionario para almacenar los valores suavizados de cada columna
        valores_suavizados = {}
        for col in columnas_caracteristicas:
            # Aplicar media rodante a cada columna
            suavizados = grupo[col].rolling(window=window, min_periods=window).mean().dropna().reset_index(drop=True)
            valores_suavizados[col] = suavizados

        # Verificar si hay suficientes datos después de la media móvil
        # Usamos la primera columna para comprobar, asumiendo que todas tienen la misma longitud después de suavizar
        if len(valores_suavizados[columnas_caracteristicas[0]]) < window:
            continue

        # Muestrear 12 puntos equidistantes para cada columna
        datos_mensuales = {}
        for col in columnas_caracteristicas:
            if len(valores_suavizados[col]) > 12:
                indices = np.linspace(0, len(valores_suavizados[col]) - 1, 12).astype(int)
                valores_finales = valores_suavizados[col].iloc[indices].reset_index(drop=True)
            else:
                valores_finales = valores_suavizados[col]

            # Asegurarse de que siempre haya 12 valores
            if len(valores_finales) != 12:
                if len(valores_finales) > 12:
                    valores_finales = valores_finales.iloc[:12]
                else:
                    valores_finales = pd.concat([valores_finales,
                                                 pd.Series([valores_finales.iloc[-1]] * (12 - len(valores_finales)))],
                                                ignore_index=True)
            datos_mensuales[col] = valores_finales

        # Crear fechas mensuales
        fechas = pd.date_range(start=f'{year}-01-01', periods=12, freq='MS')

        # Construir el DataFrame temporal para este país-año
        df_temp = pd.DataFrame({
            'ds': fechas,
            'Country': country
        })
        for col in columnas_caracteristicas:
            df_temp[col] = datos_mensuales[col]

        df_mensual.append(df_temp)

    # Combinar todos los DataFrames y asegurar que no esté vacío
    if not df_mensual:
        raise ValueError("No se generaron datos mensuales. Verifica que haya suficientes datos por país-año.")
    
    return pd.concat(df_mensual, ignore_index=True)

# ==============================================================================
# Función para entrenar modelo XGBoost de regresión
# ==============================================================================

def entrenar_modelo_xgboost_regresion(X_train, y_train, X_test, y_test, 
                                      ruta_modelo='../models/xgboost_model/best_xgboost_regressor.joblib',
                                      random_state=42):
    """
    Entrena un modelo de regresión XGBoost con GridSearchCV si no existe previamente.

    Parámetros:
    - X_train: DataFrame con características de entrenamiento
    - y_train: Series o array con objetivo de entrenamiento
    - X_test: DataFrame con características de prueba
    - y_test: Series o array con objetivo de prueba
    - ruta_modelo: str, ruta para guardar/cargar el modelo entrenado
    - random_state: int, semilla para reproducibilidad

    Retorna:
    - best_xgb_model: modelo XGBoost entrenado y validado
    """
    if os.path.exists(ruta_modelo):
        print("Cargando modelo previamente entrenado...")
        best_xgb_model = joblib.load(ruta_modelo)
    else:
        print("Entrenando modelo (esto puede tardar)...")
        model_xgb = xgb.XGBRegressor(random_state=random_state, objective='reg:squarederror')

        # Definir la grilla de hiperparámetros
        param_grid_xgb = {
            'n_estimators': [200, 300, 400, 500],  # Aumentar el rango para más árboles, permitiendo capturar más patrones
            'max_depth': [6, 8, 10, 12],          # Incluir profundidades más altas para manejar variabilidad
            'learning_rate': [0.05, 0.1, 0.15],   # Rango intermedio para un aprendizaje más estable y rápido
            'subsample': [0.7, 0.8, 0.9],         # Incluir valores más bajos para regularización
            'colsample_bytree': [0.7, 0.8, 0.9]   # Incluir valores más bajos para reducir sobreajuste
        }

        # Validación cruzada tipo series temporales
        cv_xgb = TimeSeriesSplit(n_splits=5)

        # GridSearchCV
        grid_search_xgb = GridSearchCV(model_xgb, 
                                       param_grid_xgb, 
                                       cv=cv_xgb, 
                                       scoring='neg_mean_squared_error', 
                                       n_jobs=-1, 
                                       verbose=1)

        # Entrenar el mejor modelo
        grid_search_xgb.fit(X_train, y_train)
        best_xgb_model = grid_search_xgb.best_estimator_

        # Guardar el modelo entrenado
        os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
        joblib.dump(best_xgb_model, ruta_modelo)
        print("Modelo guardado correctamente.")

    # Evaluación del modelo
    y_pred = best_xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluación del modelo:")
    print(f"   - MSE en Test: {mse:.2f}")
    print(f"   - R² en Test: {r2:.2f}")

    print("Parámetros óptimos encontrados:")
    print(best_xgb_model.get_params())

    # Visualización de resultados
    plt.figure(figsize=(12, 8))
    plt.plot(y_test.index, y_test, label='Real', marker='o')
    plt.plot(y_test.index, y_pred, label='Predicción', marker='x')
    plt.title('Predicciones vs. Valores Reales')
    plt.xlabel('Índice')
    plt.ylabel('Emisiones de Carbono (Millones de Toneladas)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../outputs/xgb/predicciones_xgboost.png')
    plt.show()
    plt.close()

    return best_xgb_model

# ==============================================================================
# Función para entrenar modelo Random Forest de clasificación
# ==============================================================================

def entrenar_modelo_clasificacion_base(X_train, y_train, X_test, y_test, 
                                       ruta_modelo='../models/forest_model/initial_random_forest_classifier.joblib'):
    """
    Entrena un modelo base RandomForestClassifier y evalúa su desempeño.

    Parámetros:
        - X_train, y_train: datos de entrenamiento
        - X_test, y_test: datos de prueba
        - ruta_modelo: ruta para guardar el modelo entrenado

    Retorna:
        - model_cls: modelo entrenado o cargado
    """

    if os.path.exists(ruta_modelo):
        print("Cargando modelo base previamente entrenado...")
        model_cls = joblib.load(ruta_modelo)
    else:
        print("Entrenando modelo base RandomForestClassifier...")

        model_cls = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        model_cls.fit(X_train, y_train)

        # Guardar modelo
        os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
        joblib.dump(model_cls, ruta_modelo)
        print(f"Modelo guardado en: {ruta_modelo}")

    # --- Evaluación del modelo ---
    y_pred = model_cls.predict(X_test)
    y_proba = model_cls.predict_proba(X_test)[:, 1]

    print("\nEvaluación del modelo base:")
    print(f"  - Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
    print(f"  - Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"  - Recall:    {recall_score(y_test, y_pred):.2f}")
    print(f"  - F1-Score:  {f1_score(y_test, y_pred):.2f}")
    print(f"  - ROC AUC:   {roc_auc_score(y_test, y_proba):.2f}")

    # --- Visualización: Matriz de Confusión ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Baja adopción", "Alta adopción"],
                yticklabels=["Baja adopción", "Alta adopción"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - Clasificación Base")
    plt.tight_layout()
    plt.savefig("../outputs/forest/matriz_confusion_clasificacion_base.png", dpi=300)
    plt.show()
    plt.close()

    return model_cls

# ==============================================================================
# Función para entrenar modelo Random Forest de clasificación optimizado
# ==============================================================================

def entrenar_modelo_clasificacion_tuned(X_train, y_train, X_test, y_test,
                                        ruta_modelo='../models/forest_model/best_random_forest_classifier.joblib',
                                        tscv=None):
    """
    Entrena y optimiza un modelo Random Forest para clasificación usando GridSearchCV.

    Parámetros:
    - X_train, y_train: datos de entrenamiento
    - X_test, y_test: datos de prueba
    - ruta_modelo: ruta para guardar el modelo entrenado
    - tscv: estrategia de validación cruzada (TimeSeriesSplit o int)

    Retorna:
    - best_cls_model: modelo optimizado entrenado o cargado
    """

    if os.path.exists(ruta_modelo):
        print("Cargando modelo previamente optimizado...")
        best_cls_model = joblib.load(ruta_modelo)
    else:
        print("Iniciando búsqueda de hiperparámetros...")

        # Espacio de búsqueda enfocado
        param_grid_cls_tuned = {
            'n_estimators': [120, 170, 200],
            'max_depth': [11, 14, 16],
            'min_samples_split': [4, 8],
            'criterion': ['entropy']
        }

        # Instancia GridSearchCV
        grid_search_cls = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            param_grid=param_grid_cls_tuned,
            cv=tscv if tscv else 5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )

        # Entrenamiento
        grid_search_cls.fit(X_train, y_train)
        best_cls_model = grid_search_cls.best_estimator_

        print(f"\nMejores parámetros encontrados: {grid_search_cls.best_params_}")
        print(f"Mejor ROC AUC en validación cruzada: {grid_search_cls.best_score_:.2f}")

        # Guardar modelo
        os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
        joblib.dump(best_cls_model, ruta_modelo)
        print(f"\nModelo optimizado guardado en: {ruta_modelo}")

    # Evaluación y visualización (ya sea cargado o entrenado)
    y_pred = best_cls_model.predict(X_test)
    y_proba = best_cls_model.predict_proba(X_test)[:, 1]

    print("\nEvaluación en el conjunto de prueba:")
    print(f"  - Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
    print(f"  - Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"  - Recall:    {recall_score(y_test, y_pred):.2f}")
    print(f"  - F1-Score:  {f1_score(y_test, y_pred):.2f}")
    print(f"  - ROC AUC:   {roc_auc_score(y_test, y_proba):.2f}")

    # Visualización de la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["Baja adopción", "Alta adopción"],
                yticklabels=["Baja adopción", "Alta adopción"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - Modelo Clasificación Optimizado")
    plt.tight_layout()
    plt.savefig("../outputs/forest/matriz_confusion_clasificacion_tuned.png", dpi=300)
    plt.show()

    return best_cls_model