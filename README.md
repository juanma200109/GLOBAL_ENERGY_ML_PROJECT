# Consumo de Energía Global (2000-2024)

## 1. Descripción del Proyecto

Este proyecto analiza un conjunto de datos que abarca dos décadas (2000-2024) de consumo energético global, desglosado por país y sector (industrial y doméstico). Su propósito es examinar las tendencias históricas del consumo de energía, evaluar la adopción de fuentes renovables frente a la dependencia de combustibles fósiles, y explorar la relación entre estos patrones y las emisiones de $\text{CO}_2$. Además, se investigan disparidades regionales y se buscan correlaciones con factores económicos, geográficos y políticas energéticas, proporcionando una visión integral de la transición energética mundial.

## 2. Motivación y Objetivos
**Motivación**
- **Relevancia de la Energía:** La energía es un pilar fundamental para el desarrollo socioeconómico, industrial y la mejora de la calidad de vida, impulsando avances tecnológicos y el bienestar general.

**Transición Energética:** La urgencia de diversificar la matriz energética hacia fuentes sostenibles (como solar y eólica) surge del impacto ambiental de los combustibles fósiles, su agotamiento progresivo y la necesidad de mitigar el cambio climático.


**Objetivos**
- Analizar la evolución del consumo energético global y sectorial (industrial y doméstico) entre 2000 y 2024, identificando tendencias y cambios significativos.
- Evaluar la penetración de las energías renovables en el mix energético de los principales países consumidores, midiendo el ritmo de adopción.
- Explorar cómo variables económicas, geográficas y políticas influyen en la transición hacia fuentes renovables.
- Cuantificar la dependencia de combustibles fósiles y su interacción con el incremento de energías renovables, destacando contrastes entre países.
- Determinar la correlación entre el consumo energético y las emisiones de $\text{CO}_2$ a nivel nacional, con un enfoque en patrones sectoriales y regionales.

## 3. Estructura del Repositorio

```text
├── data/
│   ├── original/
│   │   └── global_energy_consumption.csv
│   └── procesados/
│       └── global_energy_consumption_clean.csv
│       └── global_energy_consumption_features.csv
├── notebooks/
│   └── 01_exploracion_datos.ipynb
│   └── 02_preprocesamiento_datos.ipynb
│   └── 03_ingenieria_caracteristicas.ipynb
│   └── 04_entrenamiento_modelos.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocesamiento.py
│   ├── ingenieria_caracteristicas.py
│   ├── modelado.py
│   └── visualizacion.py
├── .gitignore
├── LICENCE
├── README.md
├── requirements.txt

```

## 4. Fuentes de Datos
- **Global Energy Consumption (2000-2024):** Dataset original que contiene métricas de consumo energético, uso per cápita, participación de renovables, dependencia fósil y emisiones de $\text{CO}_2$ por país y año.

- **dataset:** [Consumo de Energía Global (2000-2024)](https://www.kaggle.com/datasets/atharvasoundankar/global-energy-consumption-2000-2024 ) 

## 5. Visualización y Reportes
Los resultados y visualizaciones clave se encuentran en los siguientes notebooks:

- `01_exploracion_datos.ipynb:` Análisis exploratorio de datos (EDA) con histogramas, gráficos KDE, correlaciones y series temporales por país.
- `02_preprocesamiento_datos.ipynb:` Limpieza, estandarización de variables numéricas y codificación de variables categóricas (One-Hot Encoding para Country).
- `03_ingenieria_caracteristicas.ipynb:` Creación de nuevas características como tasas de crecimiento anual y ratios energéticos.
- `04_entrenamiento_modelos.ipynb:` Entrenamiento y evaluación de modelos predictivos (Prophet para series temporales, XGBoost para regresión, Random Forest para clasificación).

Los gráficos incluyen distribuciones, tendencias temporales, mapas de calor de correlación y matrices de confusión, generados con Matplotlib, Seaborn y herramientas específicas de cada modelo.

## 6. Licencia

Este proyecto está bajo la licencia **Apache-2.0 license**

## 7. Autor

- Juan Manuel Martínez Estrada
- manuel.martinez1@utp.edu.co
- [LinkedIn](https://www.linkedin.com/in/juan-manuel-martinez-estrada-8b17b2292/)

