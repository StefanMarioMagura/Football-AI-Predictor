# src/predictor.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Añadir la ruta de src al path para poder importar data_loader y feature_engineer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_loader import load_all_league_data
from feature_engineer import calculate_team_stats

# Definir las columnas a excluir aquí también, para que predictor.py sea autocontenido
# Esta lista debe ser la misma que la usada en model_trainer.py y main.py
features_to_exclude = [
    'HomeTeam', 'AwayTeam', 'Date', 'FullTimeResult',
    'League',
    'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HalfTimeResult', 'Referee',
]

def load_model_and_encoder(model_path='../models/xgboost_football_predictor.joblib',
                           encoder_path='../models/label_encoder.joblib'):
    """Carga el modelo entrenado y el LabelEncoder."""
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        print(f"Modelo cargado desde: {model_path}")
        print(f"LabelEncoder cargado desde: {encoder_path}") # Corregido 'le_path' a 'encoder_path'
        return model, le
    except FileNotFoundError:
        print("Error: Modelo o LabelEncoder no encontrados. Asegúrate de haberlos entrenado y guardado.")
        return None, None

def make_prediction_for_match(home_team, away_team, current_data_df, trained_model, label_encoder):
    """
    Genera un pronóstico para un partido futuro basándose en las estadísticas actuales.

    Args:
        home_team (str): Nombre del equipo local.
        away_team (str): Nombre del equipo visitante.
        current_data_df (pd.DataFrame): DataFrame que contiene todos los datos de partidos
                                        hasta el momento del partido a predecir.
                                        Es CRUCIAL que este DataFrame contenga los partidos
                                        anteriores para calcular las estadísticas de forma.
        trained_model: El modelo de ML entrenado.
        label_encoder: El LabelEncoder usado para codificar las etiquetas.

    Returns:
        dict: Un diccionario con las probabilidades de H, D, A.
    """

    # Asegúrate de que el DataFrame de datos actuales esté ordenado por fecha
    current_data_df = current_data_df.sort_values(by='Date').reset_index(drop=True)

    # Crea un DataFrame temporal para calcular las características del partido.
    # Añadimos el partido actual con resultados dummy (0,0,D) temporalmente
    # Solo para que calculate_team_stats pueda iterar sobre él y calcular las stats PREVIAS
    # Las stats reales del partido no se usarán para la predicción, solo para la actualización posterior
    dummy_match = pd.DataFrame([{
        'HomeTeam': home_team, 'AwayTeam': away_team, 'Date': pd.to_datetime('2099-01-01'), # Fecha futura para asegurar que va al final
        'FullTimeHomeGoals': 0, 'FullTimeAwayGoals': 0, 'FullTimeResult': 'D', # Dummy results
        'HomeShotsTarget': 0, 'AwayShotsTarget': 0, 'HomeCorners': 0, 'AwayCorners': 0,
        'HomeFouls': 0, 'AwayFouls': 0, 'HomeYellowCards': 0, 'AwayYellowCards': 0,
        'HomeRedCards': 0, 'AwayRedCards': 0
    }])

    # Unir los datos históricos con el partido "futuro" para calcular las características
    # Es crucial que el partido futuro sea el ÚLTIMO en el DataFrame para que las estadísticas
    # de los equipos se calculen correctamente hasta ese momento.
    combined_df = pd.concat([current_data_df, dummy_match], ignore_index=True)

    # Calcular las características para todos los partidos, incluyendo el "futuro"
    # El bucle de calculate_team_stats se encargará de calcular las características
    # del partido dummy basándose en los datos anteriores.
    features_df = calculate_team_stats(combined_df.copy())

    # Extraer las características del último partido (el partido futuro) como una Serie.
    # Usamos .iloc[-1] para obtener la última fila como una Serie.
    # Luego, eliminamos las columnas que no son características predictivas usando 'drop' con 'labels'.
    # La lista 'features_to_exclude' debe estar disponible (la hemos definido al principio del archivo).
    match_features_series = features_df.iloc[-1].drop(labels=[col for col in features_to_exclude if col in features_df.columns], errors='ignore')

    # Para asegurar que 'expected_feature_cols' esté disponible aquí.
    # Es la lista de columnas que el modelo espera en la entrada (las mismas que X_train).
    # Se puede obtener de las columnas de 'features_df' después de quitar las excluidas.
    expected_feature_cols = [col for col in features_df.columns if col not in features_to_exclude]

    # Asegurarse de que las columnas estén en el orden correcto y rellenar NaN si hay alguna característica faltante.
    # Convertimos la Serie 'match_features_series' a un DataFrame de una sola fila,
    # reindexando para asegurar el orden de las columnas y rellenando los valores NaN con 0.
    aligned_X_predict = pd.DataFrame([match_features_series.reindex(expected_feature_cols).fillna(0)])

    # Asegurarse de que los tipos de datos coincidan (convertir a numérico)
    for col in aligned_X_predict.columns:
        aligned_X_predict[col] = pd.to_numeric(aligned_X_predict[col], errors='coerce').fillna(0)

    # Hacer la predicción de probabilidades
    probabilities = trained_model.predict_proba(aligned_X_predict)[0] # [0] para obtener el array de probabilidades

    # Decodificar los resultados para hacerlos legibles
    decoded_results = label_encoder.inverse_transform(trained_model.classes_) # Obtener el orden de las clases

    prediction_results = {decoded_results[i]: probabilities[i] for i in range(len(probabilities))}

    # Ordenar por probabilidad (opcional)
    sorted_predictions = sorted(prediction_results.items(), key=lambda item: item[1], reverse=True)

    print(f"\nPredicción para {home_team} vs {away_team}:")
    for result, prob in sorted_predictions:
        print(f"  {result}: {prob:.2%}")

    return prediction_results

