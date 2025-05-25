import pandas as pd
import numpy as np

# ==============================================================================
# Función para calcular tasas de crecimiento anual
# ==============================================================================

def calcular_tasas_crecimiento_anual(df, columnas, col_grupo='Country', col_tiempo='Year'):
    """
    Calcula tasas de crecimiento anual para variables numéricas, ordenando previamente el dataset.

    Entradas:
        - df: DataFrame, base de datos.
        - columnas: list, columnas numéricas para calcular la tasa de crecimiento.
        - col_grupo: str, columna para agrupar (por defecto 'Country').
        - col_tiempo: str, columna de tiempo para ordenar cronológicamente (por defecto 'Year').

    Salida:
        - df: DataFrame con las columnas nuevas de tasas anuales y nulos imputados con 0.
    """
    try:
        # Ordenar por grupo y tiempo
        df = df.sort_values(by=[col_grupo, col_tiempo])

        # Calcular tasas de crecimiento anual por grupo
        for col in columnas:
            nombre_col = f"Tasa Anual {col}"
            df[nombre_col] = df.groupby(col_grupo)[col].pct_change() * 100

        # Imputar valores faltantes con 0
        columnas_tasa = [f"Tasa Anual {col}" for col in columnas]
        df[columnas_tasa] = df[columnas_tasa].fillna(0)

        return df

    except KeyError as e:
        print(f"Error: Columna no encontrada en el DataFrame. {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    
    return df

# ==============================================================================
# Función para calcular ratios energéticos
# ==============================================================================

def calcular_ratios_energeticos(df):
    """
    Calcula métricas derivadas o ratios energéticos a partir de columnas clave del DataFrame.

    Entradas:
        - df: DataFrame con columnas energéticas necesarias:
            - 'Renewable Energy Share (%)'
            - 'Fossil Fuel Dependency (%)'
            - 'Total Energy Consumption (TWh)'
            - 'Carbon Emissions (Million Tons)'
            - 'Energy Price Index (USD/kWh)'

    Salidas:
        - df: DataFrame original con 4 nuevas columnas:
            - 'renewable_fossil_ratio'
            - 'energy_efficiency'
            - 'fossil_total_ratio'
            - 'price_emissions_ratio'
    """
    try:
        df['renewable_fossil_ratio'] = df['Renewable Energy Share (%)'] / df['Fossil Fuel Dependency (%)']

        df['energy_efficiency'] = df['Total Energy Consumption (TWh)'] / df['Carbon Emissions (Million Tons)']

        df['fossil_total_ratio'] = df['Fossil Fuel Dependency (%)'] / (
            df['Fossil Fuel Dependency (%)'] + df['Renewable Energy Share (%)']
        )

        df['price_emissions_ratio'] = df['Energy Price Index (USD/kWh)'] / df['Carbon Emissions (Million Tons)']

        return df

    except KeyError as e:
        print(f"Error: Columna faltante en el DataFrame -> {e}")
    except ZeroDivisionError:
        print("Error: División por cero detectada en alguna fila.")
    except Exception as e:
        print(f"Error inesperado: {e}")

    return df