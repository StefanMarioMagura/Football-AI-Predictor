# src/model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Para guardar el modelo
import os # Necesitamos os para manejar rutas de archivos

def train_and_evaluate_model(df_features):
    """
    Entrena un modelo XGBoost para predecir el resultado del partido (1, X, 2).

    Args:
        df_features (pd.DataFrame): DataFrame con las características y la etiqueta.

    Returns:
        tuple: (modelo entrenado, LabelEncoder usado)
    """

    # 1. Preparación de X (features) e y (target)
    features_to_exclude = [
        'HomeTeam', 'AwayTeam', 'Date', 'FullTimeResult',
        'League',
        'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HalfTimeResult', 'Referee',
    ]

    X = df_features.drop(columns=[col for col in features_to_exclude if col in df_features.columns], errors='ignore')

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    y = df_features['FullTimeResult']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Clases codificadas: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, shuffle=True)

    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)} partidos")
    print(f"Tamaño del conjunto de prueba: {len(X_test)} partidos")

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    print("\nEntrenando el modelo XGBoost...")
    model.fit(X_train, y_train)
    print("Modelo entrenado.")

    print("\nEvaluando el modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # --- INICIO DE LA SECCIÓN CRÍTICA DE GUARDADO ---

    # Obtener la ruta del directorio actual (donde está model_trainer.py, es decir, 'src')
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir un nivel para llegar a la raíz del proyecto (C:\Users\Stefan\Desktop\futbol-ia)
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, os.pardir))
    # Construir la ruta completa a la carpeta 'models'
    models_dir = os.path.join(project_root_dir, 'models')

    # Asegurarse de que la carpeta 'models' exista, creándola si es necesario
    # Esto creará la carpeta models en la raíz del proyecto (futbol-ia)
    try:
        os.makedirs(models_dir, exist_ok=True)
        print(f"Verificando/Creando carpeta de modelos en: {models_dir}")
    except OSError as e:
        print(f"Error al crear la carpeta de modelos {models_dir}: {e}")
        # Considerar si se debe detener la ejecución aquí o continuar (podría fallar el guardado)
        return None, None # Si no se puede crear la carpeta, el modelo no se puede guardar

    # Rutas completas para los archivos del modelo y encoder
    model_full_path = os.path.join(models_dir, 'xgboost_football_predictor.joblib')
    le_full_path = os.path.join(models_dir, 'label_encoder.joblib')

    try:
        joblib.dump(model, model_full_path)
        joblib.dump(le, le_full_path)
        print(f"\nModelo guardado en: {model_full_path}")
        print(f"LabelEncoder guardado en: {le_full_path}")
    except Exception as e:
        print(f"Error al guardar el modelo o LabelEncoder en {models_dir}: {e}")
        # Si el guardado falla, retornamos None, None para indicar que no se pudo guardar
        return None, None

    # --- FIN DE LA SECCIÓN CRÍTICA DE GUARDADO ---

    return model, le