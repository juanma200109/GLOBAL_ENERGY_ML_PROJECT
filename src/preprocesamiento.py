import pandas as pd

def cargar_datos(ruta_archivo):
    """
    Carga un archivo CSV y devuelve un DataFrame de pandas.

    Entradas:
        - ruta_archivo: str, ruta del archivo CSV a cargar
    Salidas:
        - datos: DataFrame, datos cargados desde el archivo CSV    
    """
    try:
        # Cargar datos desde archivo CSV
        datos = pd.read_csv(ruta_archivo)
        return datos
    except FileNotFoundError:
        print("Error: El archivo no existe")
    except pd.errors.EmptyDataError:
        print("Error: El archivo está vacío")
    except pd.errors.ParserError:
        print("Error: Error al analizar el archivo")
    except Exception as e:
        print(f"Error inesperado: {e}")
    return None

def estandarizacion(df_base, columnas, scaler):
    """
    Estandariza las columnas de un DataFrame utilizando un objeto scaler (ej. StandardScaler).

    Entradas:
        - df_base: DataFrame, base de datos a estandarizar
        - columnas: list, columnas a estandarizar
        - scaler: objeto de escalado (ej. StandardScaler)
    Salidas:
        - df_escalados: DataFrame, datos estandarizados
    """

    try:
        # Evitando duplicados en el DataFrame
        df_base = df_base.loc[:, ~df_base.columns.duplicated()]

        # Escalado de datos
        datos_escalados = scaler.fit_transform(df_base)
        df_escalados = pd.DataFrame(datos_escalados, columns=columnas)
        return df_escalados
    except ValueError as e:
        print("Ya se escalaron estas columnas o hay un conflicto de dimensiones:", e)

def combinar_df(df, df_escalados, columnas):
    """
    Combina el DataFrame original con el DataFrame escalado, asegurando que sólo 
    queden las columnas escaladas en lugar de las originales.

    Entradas:
        - df: DataFrame, base de datos original
        - df_escalados: DataFrame, datos estandarizados
        - columnas: list, columnas a estandarizar
    Salidas:
        - df: DataFrame, base de datos combinada
    """ 
    try:
        # Verificar que todas las columnas originales existen
        faltantes = [c for c in columnas if c not in df.columns]
        if faltantes:
            raise KeyError(f"No se encuentran en df las columnas originales: {faltantes}")

        # Verificar que df_escalados tiene el mismo número de filas
        if len(df_escalados) != len(df):
            raise ValueError(
                f"df_escalados tiene {len(df_escalados)} filas, pero df tiene {len(df)} filas"
            )
        
        # Verificar que df_escalados tiene exactamente las columnas esperadas
        esperadas = set(columnas)
        escaladas = set(df_escalados.columns)
        if escaladas != esperadas:
            raise ValueError(
                f"Las columnas de df_escalados {sorted(escaladas)} no coinciden con las esperadas {sorted(esperadas)}"
            )

        # Eliminar las columnas originales y concatenar las escaladas
        df_sin_orig = df.drop(columns=columnas)
        df_comb = pd.concat([df_sin_orig, df_escalados], axis=1)

        # Asegurar que no queden duplicados
        df_comb = df_comb.loc[:, ~df_comb.columns.duplicated()]

        return df_comb

    except KeyError as e:
        print(f"[Error de columnas]: {e}")
        raise

    except ValueError as e:
        print(f"[Error de dimensiones]: {e}")
        raise

    except Exception as e:
        print(f"[Error inesperado]: {e}")
        raise


def code_categori(df, column, encoder):
    """
    Codifica una columna categórica de df con OneHotEncoder y reemplaza la columna original
    por sus dummies.

    Entradas:
        - df: DataFrame original.
        - column: str, nombre de la columna categórica a codificar.
        - encoder: instancia de sklearn.preprocessing.OneHotEncoder, configurada.
    Salidas:
        - df_out: DataFrame con la columna original eliminada y las dummies añadidas.
    """
    try:
        # 1) Verificar existencia de la columna
        if column not in df.columns:
            raise KeyError(f"La columna '{column}' no existe en el DataFrame.")
        
        # 2) Aplicar encoder
        datos_enc = encoder.fit_transform(df[[column]])
        cols_enc = encoder.get_feature_names_out([column])
        
        # 3) Comprobar dimensiones
        if datos_enc.shape[0] != len(df):
            raise ValueError(
                f"El encoder devolvió {datos_enc.shape[0]} filas, pero df tiene {len(df)} filas."
            )
        
        # 4) Crear DataFrame de dummies
        df_dummies = pd.DataFrame(datos_enc, columns=cols_enc, index=df.index)
        
        # 5) Eliminar la columna original y concatenar
        # df_sin = df.drop(columns=[column])
        df_out = pd.concat([df, df_dummies], axis=1)
        
        # 6) Eliminar duplicados por si acaso
        df_out = df_out.loc[:, ~df_out.columns.duplicated()]
        
        return df_out

    except KeyError as e:
        print(f"[Error de columna]: {e}")
        raise

    except ValueError as e:
        print(f"[Error de dimensión]: {e}")
        raise

    except Exception as e:
        print(f"[Error inesperado]: {e}")
        raise

def guardar_dataframe(df, ruta_archivo):
    """
    Guarda un DataFrame en un archivo CSV.

    Entradas:
        - df: DataFrame a guardar.
        - ruta_archivo: str, ruta del archivo CSV donde se guardará el DataFrame.
    Salidas:
        - None
    """
    try:
        df.to_csv(ruta_archivo, index=False)
        print(f"DataFrame guardado en {ruta_archivo}")
    except Exception as e:
        print(f"Error al guardar el DataFrame: {e}")