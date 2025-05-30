# src/feature_engineer.py

import pandas as pd
import numpy as np

def calculate_team_stats(df):
    """
    Calcula estadísticas acumulativas y de forma para cada equipo antes de cada partido.
    Se asume que el DataFrame ya está ordenado por fecha.

    Args:
        df (pd.DataFrame): DataFrame con los datos brutos de los partidos, ordenado por fecha.

    Returns:
        pd.DataFrame: DataFrame con las nuevas características añadidas.
    """
    
    # Inicializar DataFrames para almacenar estadísticas por equipo
    # Usaremos diccionarios para almacenar las estadísticas por equipo para un acceso rápido
    team_stats = {} 
    
    # Columnas para las nuevas características
    features = []

    # Iterar sobre cada partido en el DataFrame
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Inicializar estadísticas para equipos si no existen
        if home_team not in team_stats:
            team_stats[home_team] = {
                'MatchesPlayed': 0, 'GoalsScored': 0, 'GoalsConceded': 0, 'ShotsTarget': 0,
                'Corners': 0, 'Fouls': 0, 'YellowCards': 0, 'RedCards': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0,
                'Last5GoalsScored': [], 'Last5GoalsConceded': [], 'Last5Results': [],
                'HomeGoalsScored': 0, 'HomeGoalsConceded': 0, 'HomeWins': 0, 'HomeDraws': 0, 'HomeLosses': 0,
                'AwayGoalsScored': 0, 'AwayGoalsConceded': 0, 'AwayWins': 0, 'AwayDraws': 0, 'AwayLosses': 0,
            }
        if away_team not in team_stats:
            team_stats[away_team] = {
                'MatchesPlayed': 0, 'GoalsScored': 0, 'GoalsConceded': 0, 'ShotsTarget': 0,
                'Corners': 0, 'Fouls': 0, 'YellowCards': 0, 'RedCards': 0, 'Wins': 0, 'Draws': 0, 'Losses': 0,
                'Last5GoalsScored': [], 'Last5GoalsConceded': [], 'Last5Results': [],
                'HomeGoalsScored': 0, 'HomeGoalsConceded': 0, 'HomeWins': 0, 'HomeDraws': 0, 'HomeLosses': 0,
                'AwayGoalsScored': 0, 'AwayGoalsConceded': 0, 'AwayWins': 0, 'AwayDraws': 0, 'AwayLosses': 0,
            }

        # --- Características del equipo local antes del partido ---
        home_features = {
            f'Home_AvgGoalsScored_Prev': team_stats[home_team]['GoalsScored'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_AvgGoalsConceded_Prev': team_stats[home_team]['GoalsConceded'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_AvgShotsTarget_Prev': team_stats[home_team]['ShotsTarget'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_AvgCorners_Prev': team_stats[home_team]['Corners'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_WinRatio_Prev': team_stats[home_team]['Wins'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_DrawRatio_Prev': team_stats[home_team]['Draws'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            f'Home_LossRatio_Prev': team_stats[home_team]['Losses'] / team_stats[home_team]['MatchesPlayed'] if team_stats[home_team]['MatchesPlayed'] > 0 else 0,
            
            # Forma reciente (últimos 5 partidos)
            f'Home_Form_GoalsScored_Last5': sum(team_stats[home_team]['Last5GoalsScored']) / len(team_stats[home_team]['Last5GoalsScored']) if team_stats[home_team]['Last5GoalsScored'] else 0,
            f'Home_Form_GoalsConceded_Last5': sum(team_stats[home_team]['Last5GoalsConceded']) / len(team_stats[home_team]['Last5GoalsConceded']) if team_stats[home_team]['Last5GoalsConceded'] else 0,
            f'Home_Form_Wins_Last5': team_stats[home_team]['Last5Results'].count('W') / len(team_stats[home_team]['Last5Results']) if team_stats[home_team]['Last5Results'] else 0,
            f'Home_Form_Draws_Last5': team_stats[home_team]['Last5Results'].count('D') / len(team_stats[home_team]['Last5Results']) if team_stats[home_team]['Last5Results'] else 0,
            f'Home_Form_Losses_Last5': team_stats[home_team]['Last5Results'].count('L') / len(team_stats[home_team]['Last5Results']) if team_stats[home_team]['Last5Results'] else 0,

            # Estadísticas específicas de jugar en casa
            f'Home_HomeWinRatio_Prev': team_stats[home_team]['HomeWins'] / (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) if (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) > 0 else 0,
            f'Home_HomeGoalsScored_Prev': team_stats[home_team]['HomeGoalsScored'] / (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) if (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) > 0 else 0,
            f'Home_HomeGoalsConceded_Prev': team_stats[home_team]['HomeGoalsConceded'] / (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) if (team_stats[home_team]['HomeWins'] + team_stats[home_team]['HomeDraws'] + team_stats[home_team]['HomeLosses']) > 0 else 0,
        }
        
        # --- Características del equipo visitante antes del partido ---
        away_features = {
            f'Away_AvgGoalsScored_Prev': team_stats[away_team]['GoalsScored'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_AvgGoalsConceded_Prev': team_stats[away_team]['GoalsConceded'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_AvgShotsTarget_Prev': team_stats[away_team]['ShotsTarget'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_AvgCorners_Prev': team_stats[away_team]['Corners'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_WinRatio_Prev': team_stats[away_team]['Wins'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_DrawRatio_Prev': team_stats[away_team]['Draws'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,
            f'Away_LossRatio_Prev': team_stats[away_team]['Losses'] / team_stats[away_team]['MatchesPlayed'] if team_stats[away_team]['MatchesPlayed'] > 0 else 0,

            # Forma reciente (últimos 5 partidos)
            f'Away_Form_GoalsScored_Last5': sum(team_stats[away_team]['Last5GoalsScored']) / len(team_stats[away_team]['Last5GoalsScored']) if team_stats[away_team]['Last5GoalsScored'] else 0,
            f'Away_Form_GoalsConceded_Last5': sum(team_stats[away_team]['Last5GoalsConceded']) / len(team_stats[away_team]['Last5GoalsConceded']) if team_stats[away_team]['Last5GoalsConceded'] else 0,
            f'Away_Form_Wins_Last5': team_stats[away_team]['Last5Results'].count('W') / len(team_stats[away_team]['Last5Results']) if team_stats[away_team]['Last5Results'] else 0,
            f'Away_Form_Draws_Last5': team_stats[away_team]['Last5Results'].count('D') / len(team_stats[away_team]['Last5Results']) if team_stats[away_team]['Last5Results'] else 0,
            f'Away_Form_Losses_Last5': team_stats[away_team]['Last5Results'].count('L') / len(team_stats[away_team]['Last5Results']) if team_stats[away_team]['Last5Results'] else 0,

            # Estadísticas específicas de jugar fuera de casa
            f'Away_AwayWinRatio_Prev': team_stats[away_team]['AwayWins'] / (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) if (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) > 0 else 0,
            f'Away_AwayGoalsScored_Prev': team_stats[away_team]['AwayGoalsScored'] / (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) if (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) > 0 else 0,
            f'Away_AwayGoalsConceded_Prev': team_stats[away_team]['AwayGoalsConceded'] / (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) if (team_stats[away_team]['AwayWins'] + team_stats[away_team]['AwayDraws'] + team_stats[away_team]['AwayLosses']) > 0 else 0,

        }

        # --- Crear características relativas entre equipos ---
        relative_features = {
            'GoalDifference_Prev': home_features['Home_AvgGoalsScored_Prev'] - away_features['Away_AvgGoalsConceded_Prev'],
            'ShotsTargetDifference_Prev': home_features['Home_AvgShotsTarget_Prev'] - away_features['Away_AvgShotsTarget_Prev'],
            'FormDifference_GoalsScored_Last5': home_features['Home_Form_GoalsScored_Last5'] - away_features['Away_Form_GoalsScored_Last5'],
            'FormDifference_GoalsConceded_Last5': home_features['Home_Form_GoalsConceded_Last5'] - away_features['Away_Form_GoalsConceded_Last5'],
            'FormDifference_Wins_Last5': home_features['Home_Form_Wins_Last5'] - away_features['Away_Form_Wins_Last5'],
        }

        # Combina todas las características para este partido
        current_features = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': row['Date'],
            'FullTimeResult': row['FullTimeResult'], # Mantener el resultado para el target
            **home_features,
            **away_features,
            **relative_features
        }
        features.append(current_features)

        # --- Actualizar las estadísticas de los equipos DESPUÉS del partido ---
        # Equipo Local
        team_stats[home_team]['MatchesPlayed'] += 1
        team_stats[home_team]['GoalsScored'] += row['FullTimeHomeGoals']
        team_stats[home_team]['GoalsConceded'] += row['FullTimeAwayGoals']
        team_stats[home_team]['ShotsTarget'] += row['HomeShotsTarget'] if 'HomeShotsTarget' in row else 0 # Añadir check por si la columna no existe
        team_stats[home_team]['Corners'] += row['HomeCorners'] if 'HomeCorners' in row else 0
        team_stats[home_team]['Fouls'] += row['HomeFouls'] if 'HomeFouls' in row else 0
        team_stats[home_team]['YellowCards'] += row['HomeYellowCards'] if 'HomeYellowCards' in row else 0
        team_stats[home_team]['RedCards'] += row['HomeRedCards'] if 'HomeRedCards' in row else 0

        # Actualizar resultados para forma reciente (últimos 5 partidos)
        team_stats[home_team]['Last5GoalsScored'].append(row['FullTimeHomeGoals'])
        team_stats[home_team]['Last5GoalsConceded'].append(row['FullTimeAwayGoals'])
        if row['FullTimeResult'] == 'H': # Home win
            team_stats[home_team]['Wins'] += 1
            team_stats[home_team]['HomeWins'] += 1
            team_stats[home_team]['Last5Results'].append('W')
        elif row['FullTimeResult'] == 'D': # Draw
            team_stats[home_team]['Draws'] += 1
            team_stats[home_team]['HomeDraws'] += 1
            team_stats[home_team]['Last5Results'].append('D')
        else: # Away win (Home Loss)
            team_stats[home_team]['Losses'] += 1
            team_stats[home_team]['HomeLosses'] += 1
            team_stats[home_team]['Last5Results'].append('L')

        # Mantener solo los últimos 5 resultados para la forma
        if len(team_stats[home_team]['Last5GoalsScored']) > 5:
            team_stats[home_team]['Last5GoalsScored'].pop(0)
            team_stats[home_team]['Last5GoalsConceded'].pop(0)
            team_stats[home_team]['Last5Results'].pop(0)

        # Equipo Visitante
        team_stats[away_team]['MatchesPlayed'] += 1
        team_stats[away_team]['GoalsScored'] += row['FullTimeAwayGoals']
        team_stats[away_team]['GoalsConceded'] += row['FullTimeHomeGoals']
        team_stats[away_team]['ShotsTarget'] += row['AwayShotsTarget'] if 'AwayShotsTarget' in row else 0
        team_stats[away_team]['Corners'] += row['AwayCorners'] if 'AwayCorners' in row else 0
        team_stats[away_team]['Fouls'] += row['AwayFouls'] if 'AwayFouls' in row else 0
        team_stats[away_team]['YellowCards'] += row['AwayYellowCards'] if 'AwayYellowCards' in row else 0
        team_stats[away_team]['RedCards'] += row['AwayRedCards'] if 'AwayRedCards' in row else 0

        # Actualizar resultados para forma reciente (últimos 5 partidos)
        team_stats[away_team]['Last5GoalsScored'].append(row['FullTimeAwayGoals'])
        team_stats[away_team]['Last5GoalsConceded'].append(row['FullTimeHomeGoals'])
        if row['FullTimeResult'] == 'A': # Away win
            team_stats[away_team]['Wins'] += 1
            team_stats[away_team]['AwayWins'] += 1
            team_stats[away_team]['Last5Results'].append('W')
        elif row['FullTimeResult'] == 'D': # Draw
            team_stats[away_team]['Draws'] += 1
            team_stats[away_team]['AwayDraws'] += 1
            team_stats[away_team]['Last5Results'].append('D')
        else: # Home win (Away Loss)
            team_stats[away_team]['Losses'] += 1
            team_stats[away_team]['AwayLosses'] += 1
            team_stats[away_team]['Last5Results'].append('L')

        # Mantener solo los últimos 5 resultados para la forma
        if len(team_stats[away_team]['Last5GoalsScored']) > 5:
            team_stats[away_team]['Last5GoalsScored'].pop(0)
            team_stats[away_team]['Last5GoalsConceded'].pop(0)
            team_stats[away_team]['Last5Results'].pop(0)
            
    return pd.DataFrame(features)


