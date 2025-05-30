# app.py
import streamlit as st
import pandas as pd
import os
import sys

# Asegúrate de que Python pueda encontrar tus módulos (data_loader, feature_engineer, predictor)
# Ajusta la ruta si tu estructura es diferente. Aquí asumimos que app.py está en la raíz y src contiene los módulos.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importa tus funciones y el modelo/encoder
from data_loader import load_all_league_data
from predictor import load_model_and_encoder, make_prediction_for_match
# from model_trainer import train_and_evaluate_model # Solo si necesitas re-entrenar desde la app

# --- Configuración de la Interfaz ---
st.set_page_config(page_title="Pronóstico de Fútbol con IA", layout="centered")
st.title("⚽ Pronóstico de Fútbol con IA")
st.markdown("---")

# --- Cargar Datos y Modelo al inicio (solo una vez) ---
@st.cache_data # Carga los datos una sola vez para mejorar el rendimiento
def load_data():
    st.write("Cargando datos históricos de la Premier League...")
    df_raw = load_all_league_data(data_folder='data') # O './data' si app.py está en la raíz
    st.write(f"Datos cargados: {len(df_raw)} partidos.")
    return df_raw

@st.cache_resource # Carga el modelo y encoder una sola vez
def load_model():
    st.write("Cargando modelo y LabelEncoder...")
    model, encoder = load_model_and_encoder(model_path='models/xgboost_football_predictor.joblib',
                                              encoder_path='models/label_encoder.joblib')
    if model is None or encoder is None:
        st.error("Error al cargar el modelo o el LabelEncoder. Asegúrate de haber ejecutado main.py al menos una vez para entrenarlos y guardarlos.")
    return model, encoder

df_raw = load_data()
model, label_encoder = load_model()

if model and label_encoder and not df_raw.empty:
    # --- Selección de Equipos ---
    st.header("Realizar una Predicción")

    # Obtener la lista de todos los equipos únicos del DataFrame
    all_teams = pd.Series(df_raw['HomeTeam'].unique().tolist() + df_raw['AwayTeam'].unique().tolist()).unique()
    all_teams_sorted = sorted(all_teams) # Ordenar alfabéticamente

    col1, col2 = st.columns(2)
    with col1:
        home_team_future = st.selectbox("Selecciona Equipo Local:", all_teams_sorted)
    with col2:
        # Elige un índice diferente para que no sean el mismo por defecto
        # Se asegura de que el índice no exceda la longitud de la lista
        default_away_index = (all_teams_sorted.index(home_team_future) + 1) % len(all_teams_sorted) \
                             if home_team_future in all_teams_sorted else 0
        away_team_future = st.selectbox("Selecciona Equipo Visitante:", all_teams_sorted, index=default_away_index)

    if home_team_future == away_team_future:
        st.warning("¡El equipo local y el visitante no pueden ser el mismo!")
    else:
        if st.button("Predecir Resultado"):
            with st.spinner("Calculando predicción..."):
                prediction_results = make_prediction_for_match(
                    home_team_future,
                    away_team_future,
                    df_raw.copy(), # Pasa una copia de los datos brutos
                    model,
                    label_encoder
                )
                st.success("¡Predicción realizada!")
                st.write("---")
                st.subheader(f"Probabilidades para {home_team_future} vs {away_team_future}:")
                
                # Iterar sobre los resultados ordenados para mostrar las probabilidades
                for result_code, prob in sorted(prediction_results.items(), key=lambda item: item[1], reverse=True):
                    display_label = "" # Variable para almacenar la etiqueta a mostrar
                    
                    # *** INICIO DE LA MODIFICACIÓN ***
                    if result_code == 'H':
                        display_label = f"{home_team_future} Gana"
                    elif result_code == 'D':
                        display_label = "Empate"
                    elif result_code == 'A':
                        display_label = f"{away_team_future} Gana"
                    # *** FIN DE LA MODIFICACIÓN ***
                    
                    st.metric(label=display_label, value=f"{prob:.2%}") # Mostrar la probabilidad con la etiqueta descriptiva
elif not model or not label_encoder:
    st.error("El modelo no se pudo cargar. Asegúrate de ejecutar `python src/main.py` para entrenarlo y guardarlo.")
elif df_raw.empty:
    st.error("No se pudieron cargar los datos históricos. Asegúrate de que los CSVs estén en la carpeta 'data'.")

st.markdown("---")
st.caption("Proyecto de Pronósticos de Fútbol con IA")