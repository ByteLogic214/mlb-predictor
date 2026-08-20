"""
========================================
MLB Predictor — Ingeniería de Características
========================================
Construye el vector de features para cada partido combinando:
estadísticas de pitchers, equipos, standings y forma reciente.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from src.fetcher import (
    obtener_estadisticas_equipo,
    obtener_estadisticas_lanzador,
    obtener_standings,
    obtener_historial_juegos
)
from src.utils import logger


def _safe_float(val, default=0.0):
    """Convierte un valor a float de forma segura."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Convierte un valor a int de forma segura."""
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def construir_features_juego(juego: Dict, standings_df: pd.DataFrame) -> Dict:
    """
    Para un juego dado, consulta todas las estadísticas necesarias
    y devuelve un diccionario con el vector de features.
    """
    home_id = juego["home_team_id"]
    away_id = juego["away_team_id"]
    home_pitcher = juego.get("home_pitcher_id")
    away_pitcher = juego.get("away_pitcher_id")

    # --- Estadísticas de Equipos ---
    home_team_stats = obtener_estadisticas_equipo(home_id)
    away_team_stats = obtener_estadisticas_equipo(away_id)

    # --- Estadísticas de Lanzadores ---
    home_pitcher_stats = obtener_estadisticas_lanzador(home_pitcher) if home_pitcher else {}
    away_pitcher_stats = obtener_estadisticas_lanzador(away_pitcher) if away_pitcher else {}

    # --- Standings (ya cacheado) ---
    home_stand = standings_df[standings_df["team_id"] == home_id]
    away_stand = standings_df[standings_df["team_id"] == away_id]

    home_win_pct = home_stand["win_pct"].values[0] if not home_stand.empty else 0.5
    away_win_pct = away_stand["win_pct"].values[0] if not away_stand.empty else 0.5
    home_run_diff = home_stand["run_diff"].values[0] if not home_stand.empty else 0
    away_run_diff = away_stand["run_diff"].values[0] if not away_stand.empty else 0

    # --- Forma Reciente (últimos 10 juegos) ---
    home_historial = obtener_historial_juegos(home_id, dias=10)
    away_historial = obtener_historial_juegos(away_id, dias=10)

    home_forma = sum(1 for g in home_historial if g.get("home_score", 0) > g.get("away_score", 0)) / max(len(home_historial), 1)
    away_forma = sum(1 for g in away_historial if g.get("away_score", 0) > g.get("home_score", 0)) / max(len(away_historial), 1)

    # --- Días de descanso del lanzador (simulado por disponibilidad en API) ---
    # Como la API gratuita no expone descanso directo, usamos un proxy:
    # si no hay stats del pitcher, asumimos descanso mínimo (poco fiable, pero API limitada)
    rest_days_home = 4 if home_pitcher_stats else 0
    rest_days_away = 4 if away_pitcher_stats else 0

    # --- Ensamblar Features ---
    features = {
        "game_pk": juego["game_pk"],
        "fecha": juego["fecha"],
        "home_team_id": home_id,
        "home_team_name": juego["home_team_name"],
        "away_team_id": away_id,
        "away_team_name": juego["away_team_name"],
        "home_pitcher_id": home_pitcher,
        "home_pitcher_name": juego.get("home_pitcher_name"),
        "away_pitcher_id": away_pitcher,
        "away_pitcher_name": juego.get("away_pitcher_name"),

        # Pitcher stats
        "era_pitcher_home": _safe_float(home_pitcher_stats.get("era"), 4.50),
        "whip_pitcher_home": _safe_float(home_pitcher_stats.get("whip"), 1.30),
        "era_pitcher_away": _safe_float(away_pitcher_stats.get("era"), 4.50),
        "whip_pitcher_away": _safe_float(away_pitcher_stats.get("whip"), 1.30),

        # Team offense (OPS como proxy de poder ofensivo)
        "ops_team_home": _safe_float(home_team_stats.get("hitting", {}).get("ops"), 0.720),
        "ops_team_away": _safe_float(away_team_stats.get("hitting", {}).get("ops"), 0.720),

        # Standings
        "win_pct_home": _safe_float(home_win_pct, 0.500),
        "win_pct_away": _safe_float(away_win_pct, 0.500),
        "run_diff_home": _safe_int(home_run_diff, 0),
        "run_diff_away": _safe_int(away_run_diff, 0),

        # Forma reciente
        "forma_home": _safe_float(home_forma, 0.5),
        "forma_away": _safe_float(away_forma, 0.5),

        # Descanso (proxy)
        "rest_days_home": rest_days_home,
        "rest_days_away": rest_days_away,

        # Ventaja de localía (feature binaria)
        "is_home_advantage": 1,
    }

    logger.info(f"Features construidas para: {juego['away_team_name']} @ {juego['home_team_name']}")
    return features


def construir_dataset(juegos: List[Dict]) -> pd.DataFrame:
    """
    Construye un DataFrame completo con features para una lista de juegos.
    """
    standings_df = obtener_standings()

    registros = []
    for juego in juegos:
        try:
            feat = construir_features_juego(juego, standings_df)
            registros.append(feat)
        except Exception as e:
            logger.error(f"Error procesando juego {juego.get('game_pk')}: {e}")

    df = pd.DataFrame(registros)
    logger.info(f"Dataset construido: {len(df)} filas, {len(df.columns)} columnas")
    return df
