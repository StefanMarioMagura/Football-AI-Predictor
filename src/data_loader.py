# src/data_loader.py

import pandas as pd
import os

def load_all_league_data(data_folder='../data', league_prefix='E0'):
    """
    Carga todos los archivos CSV de una liga específica de una carpeta dada
    y los concatena en un único DataFrame.

    Args:
        data_folder (str): La ruta a la carpeta donde se encuentran los archivos CSV.
        league_prefix (str): El prefijo de los archivos de la liga (ej. 'E0' para Premier League).

    Returns:
        pd.DataFrame: Un DataFrame consolidado con los datos de la liga.
    """
    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith(league_prefix) and f.endswith('.csv')]
    
    df_list = []
    for file in all_files:
        try:
            # Intentar leer con distintas codificaciones si hay problemas
            df = pd.read_csv(file, encoding='latin1') 
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='utf-8') # Otra opción de codificación

        # Añadir la temporada al DataFrame (opcional, para referencia)
        filename = os.path.basename(file)
        if '(' in filename and ')' in filename:
            season_str = filename.split('(')[1].split(')')[0]
            df['Season'] = season_str
        else:
            # Para el archivo "E0.csv" que es la temporada actual sin paréntesis
            df['Season'] = 'Current' 

        df_list.append(df)
    
    # Concatenar todos los DataFrames
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Limpieza inicial de columnas: algunas temporadas pueden tener columnas extra o diferentes
    # Nos quedamos solo con las columnas que sabemos que son útiles y están en la mayoría
    
    # Columnas esperadas y su mapeo potencial
    expected_cols = {
        'Div': 'League', 'Date': 'Date', 'HomeTeam': 'HomeTeam', 'AwayTeam': 'AwayTeam',
        'FTHG': 'FullTimeHomeGoals', 'FTAG': 'FullTimeAwayGoals', 'FTR': 'FullTimeResult',
        'HTHG': 'HalfTimeHomeGoals', 'HTAG': 'HalfTimeAwayGoals', 'HTR': 'HalfTimeResult',
        'Referee': 'Referee', 'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget', 'HC': 'HomeCorners',
        'AC': 'AwayCorners', 'HF': 'HomeFouls', 'AF': 'AwayFouls',
        'HY': 'HomeYellowCards', 'AY': 'AwayYellowCards', 'HR': 'HomeRedCards',
        'AR': 'AwayRedCards'
        # Podrías añadir más si las descargas de otras temporadas las incluyen
        # Por ejemplo, 'BbMxH', 'BbMxD', 'BbMxA' para las cuotas de las casas de apuestas (aunque no las usaremos directamente para la predicción)
    }

    # Renombrar columnas para consistencia
    full_df.rename(columns=expected_cols, inplace=True)
    
    # Filtrar solo las columnas que nos interesan y existen en el DataFrame
    # Esto manejará el caso de que algunas temporadas no tengan todas las columnas
    filtered_cols = [col for col in expected_cols.values() if col in full_df.columns]
    full_df = full_df[filtered_cols]

    # Convertir 'Date' a formato de fecha
    # Football-Data.org usa a veces formato dd/mm/yy y a veces dd/mm/yyyy
    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
    
    # Eliminar filas con fechas nulas o resultados nulos (partidos incompletos/errores)
    full_df.dropna(subset=['Date', 'FullTimeResult'], inplace=True)
    
    # Ordenar por fecha para asegurar el orden cronológico, importante para la ingeniería de características
    full_df = full_df.sort_values(by='Date').reset_index(drop=True)
    
    print(f"Cargados {len(full_df)} partidos de la liga {league_prefix}.")
    print(f"Columnas disponibles: {full_df.columns.tolist()}")
    
    return full_df

