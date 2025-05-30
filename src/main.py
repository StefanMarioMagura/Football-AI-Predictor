# src/main.py

import os
import pandas as pd # Aunque pandas se usa en los módulos, a veces es útil aquí para manipulación si se necesita.

# Importar funciones de nuestros módulos
from data_loader import load_all_league_data
from feature_engineer import calculate_team_stats
from model_trainer import train_and_evaluate_model
from predictor import load_model_and_encoder, make_prediction_for_match

# Lista de características a excluir, necesaria para la preparación de datos en el backtesting
# Esta lista debe ser la misma que la usada en model_trainer.py para consistencia.
features_to_exclude = [
    'HomeTeam', 'AwayTeam', 'Date', 'FullTimeResult', 
    'League', 
    'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HalfTimeResult', 'Referee',
]

if __name__ == "__main__":
    print("--- Iniciando el programa de Pronósticos de Fútbol con IA ---")

    # 1. Asegurar la existencia de las carpetas necesarias
    print("\n1. Verificando la estructura de carpetas...")
    data_folder = './data'
    models_folder = '../models'
    if not os.path.exists(data_folder):
        print(f"Advertencia: La carpeta '{data_folder}' no existe. Asegúrate de crearla y descargar los CSVs de Football-Data.org.")
        exit() # Salir si no hay datos para procesar
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        print(f"Carpeta '{models_folder}' creada.")
    print("Estructura de carpetas verificada.")

    # 2. Cargar los datos históricos de los partidos
    # Aquí cargamos los CSVs que descargaste de Football-Data.org
    print("\n2. Cargando datos históricos de la Premier League (E0)...")
    df_raw = load_all_league_data(data_folder=data_folder, league_prefix='E0')
    if df_raw.empty:
        print("Error: No se cargaron datos. Revisa tus archivos CSV en la carpeta 'data'.")
        exit()
    print("Datos brutos cargados exitosamente.")

    # 3. Ingeniería de Características: Transformar datos brutos en información útil
    # Esto es donde calculamos promedios, formas, etc., de los equipos antes de cada partido.
    print("\n3. Realizando Ingeniería de Características (esto puede tardar)...")
    df_features = calculate_team_stats(df_raw.copy()) 
    # Eliminamos las primeras filas que tienen NaN debido a la falta de historial para calcular las características iniciales
    df_features.dropna(subset=[col for col in df_features.columns if col not in ['HomeTeam', 'AwayTeam', 'Date', 'FullTimeResult']], inplace=True)
    if df_features.empty:
        print("Error: El DataFrame de características está vacío después de la limpieza. Puede que tus datos iniciales sean muy pocos.")
        exit()
    print(f"Características calculadas para {len(df_features)} partidos.")

    # 4. Entrenamiento o Carga del Modelo de Machine Learning
    # Intentamos cargar un modelo ya entrenado. Si no existe, lo entrenamos desde cero.
    print("\n4. Cargando o Entrenando el Modelo de Predicción...")
    trained_model, label_encoder = load_model_and_encoder(
        model_path=os.path.join(models_folder, 'xgboost_football_predictor.joblib'),
        encoder_path=os.path.join(models_folder, 'label_encoder.joblib')
    )
    
    if trained_model is None: # Si el modelo no se cargó, lo entrenamos
        print("Modelo no encontrado. Procediendo a entrenar un nuevo modelo.")
        trained_model, label_encoder = train_and_evaluate_model(df_features)
        if trained_model is None: 
            print("Error: No se pudo entrenar el modelo. Saliendo.")
            exit()
    else:
        print("Modelo y LabelEncoder cargados exitosamente.")

    # --- 5. Backtesting (Evaluación del rendimiento histórico del modelo) ---
    # Esto simula cómo se habría comportado tu modelo en el pasado.
    print("\n--- 5. Realizando Backtesting del Modelo ---")
    
    # Definimos una fecha de corte para el backtesting
    # Por ejemplo, usamos los datos hasta el 1 de enero de 2024 para entrenar
    # y los datos a partir de esa fecha para probar.
    # Ajusta esta fecha según la cantidad de datos que tengas y quieras simular.
    cutoff_date = pd.to_datetime('2024-01-01') 
    
    # Dividimos el DataFrame de características en conjuntos de entrenamiento y prueba para el backtesting
    df_features_train_backtest = df_features[df_features['Date'] < cutoff_date].copy()
    df_features_test_backtest = df_features[df_features['Date'] >= cutoff_date].copy()
    
    if df_features_test_backtest.empty:
        print(f"Advertencia: No hay partidos para el backtesting después de la fecha de corte {cutoff_date}. Ajusta la fecha.")
    else:
        print(f"Partidos para entrenar en backtest: {len(df_features_train_backtest)}")
        print(f"Partidos para predecir en backtest: {len(df_features_test_backtest)}")

        # Es crucial reentrenar el modelo con los datos disponibles ANTES del punto de corte
        # Esto simula un escenario real donde el modelo solo conoce el pasado.
        print(f"Re-entrenando modelo para backtesting con datos hasta {cutoff_date.strftime('%Y-%m-%d')}...")
        # Aquí pasamos directamente el dataframe de características ya calculadas para este subconjunto
        model_for_backtest, le_for_backtest = train_and_evaluate_model(df_features_train_backtest)

        if model_for_backtest is None:
            print("No se pudo re-entrenar el modelo para backtesting. Saltando el backtesting.")
        else:
            # Preparamos las características para el conjunto de prueba del backtesting
            X_test_backtest = df_features_test_backtest.drop(columns=[col for col in features_to_exclude if col in df_features_test_backtest.columns], errors='ignore')
            for col in X_test_backtest.columns:
                X_test_backtest[col] = pd.to_numeric(X_test_backtest[col], errors='coerce')
            X_test_backtest = X_test_backtest.fillna(0)
            
            y_test_backtest_encoded = le_for_backtest.transform(df_features_test_backtest['FullTimeResult'])

            # Realizamos las predicciones en el conjunto de prueba del backtesting
            y_pred_backtest = model_for_backtest.predict(X_test_backtest)
            # y_proba_backtest = model_for_backtest.predict_proba(X_test_backtest) # Si quisieras las probabilidades

            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            accuracy_backtest = accuracy_score(y_test_backtest_encoded, y_pred_backtest)
            print(f"\nPrecisión (Accuracy) en el Backtesting (partidos después de {cutoff_date.strftime('%Y-%m-%d')}): {accuracy_backtest:.4f}")
            print("\nReporte de Clasificación del Backtesting:")
            print(classification_report(y_test_backtest_encoded, y_pred_backtest, target_names=le_for_backtest.classes_))
            print("\nMatriz de Confusión del Backtesting:")
            print(confusion_matrix(y_test_backtest_encoded, y_pred_backtest))

    # --- 6. Ejemplo de Predicción para un Partido Futuro ---
    # Aquí demostramos cómo predecir un partido que aún no ha sucedido.
    # Para esto, necesitamos todos los datos históricos disponibles hasta "hoy".
    print("\n--- 6. Haciendo una Predicción para un Partido Futuro (Ejemplo) ---")
    
    # IMPORTANTE: Asegúrate de que estos nombres de equipo coincidan EXACTAMENTE con los datos
    # Por ejemplo, 'Man Utd' o 'Manchester Utd', dependiendo de cómo estén en tus CSVs.
    # Puedes revisar df_raw['HomeTeam'].unique() para ver los nombres exactos.
    home_team_future = 'Man Utd' 
    away_team_future = 'Liverpool'

    # Llamamos a la función de predicción. Le pasamos el df_raw COMPLETO
    # para que las características de Man Utd y Liverpool se calculen basándose
    # en todo el historial disponible hasta el momento.
    make_prediction_for_match(home_team_future, away_team_future, df_raw.copy(), trained_model, label_encoder)

    print("\n--- Proceso de Pronósticos de Fútbol completado. ---")