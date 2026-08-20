"""
========================================
MLB Predictor — Configuración Global
========================================
Parámetros centralizados para endpoints, rutas y constantes del modelo.
"""

import os
from datetime import datetime, timezone

# --- API MLB (Stats API) ---
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1  # MLB principal
MLB_LEAGUE_IDS = "103,104"  # Americana y Nacional

# --- Rutas del Repositorio ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_PREDICTIONS = os.path.join(BASE_DIR, "data", "predictions")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Asegurar que existan las carpetas
for carpeta in [DATA_RAW, DATA_PROCESSED, DATA_PREDICTIONS, MODELS_DIR]:
    os.makedirs(carpeta, exist_ok=True)

# --- Fechas ---
HOY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TEMPORADA_ACTUAL = datetime.now(timezone.utc).year

# --- Modelo ---
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.05
RANDOM_STATE = 42

# --- Features del Modelo ---
FEATURES_MODELO = [
    "era_pitcher_home", "whip_pitcher_home", "ops_team_home",
    "era_pitcher_away", "whip_pitcher_away", "ops_team_away",
    "win_pct_home", "win_pct_away", "run_diff_home", "run_diff_away",
    "rest_days_home", "rest_days_away", "is_home_advantage"
]
