"""
========================================
MLB Predictor — Ingeniería de Características
========================================
Construye features con diferenciales relativos (era_diff, ops_diff, etc.)
para evitar que el modelo dependa de valores absolutos descontextualizados.
Agrega validación de coherencia local/visitante.
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
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def construir_features_juego(juego: Dict, standings_df: pd.DataFrame) -> Dict:
    """
    Construye el vector de features para un juego.
    CORRECCIÓN: Valida que home_team_id y away_team_id no sean None.
    """
    home_id = juego.get("home_team_id")
    away_id = juego.get("away_team_id")
    home_pitcher = juego.get("home_pitcher_id")
    away_pitcher = juego.get("away_pitcher_id")

    if not home_id or not away_id:
        raise ValueError(f"IDs de equipo inválidos en juego {juego.get('game_pk')}")

    # --- Estadísticas de Equipos ---
    home_team_stats = obtener_estadisticas_equipo(home_id)
    away_team_stats = obtener_estadisticas_equipo(away_id)

    # --- Estadísticas de Lanzadores ---
    home_pitcher_stats = obtener_estadisticas_lanzador(home_pitcher) if home_pitcher else {}
    away_pitcher_stats = obtener_estadisticas_lanzador(away_pitcher) if away_pitcher else {}

    # --- Standings ---
    home_stand = standings_df[standings_df["team_id"] == home_id]
    away_stand = standings_df[standings_df["team_id"] == away_id]

    home_win_pct = home_stand["win_pct"].values[0] if not home_stand.empty else 0.500
    away_win_pct = away_stand["win_pct"].values[0] if not away_stand.empty else 0.500
    home_run_diff = home_stand["run_diff"].values[0] if not home_stand.empty else 0
    away_run_diff = away_stand["run_diff"].values[0] if not away_stand.empty else 0

    # --- Forma Reciente ---
    home_historial = obtener_historial_juegos(home_id, dias=10)
    away_historial = obtener_historial_juegos(away_id, dias=10)

    home_forma = sum(
        1 for g in home_historial
        if g.get("home_score", 0) > g.get("away_score", 0)
    ) / max(len(home_historial), 1)

    away_forma = sum(
        1 for g in away_historial
        if g.get("away_score", 0) > g.get("home_score", 0)
    ) / max(len(away_historial), 1)

    # --- Descanso del lanzador (proxy por disponibilidad de stats) ---
    rest_days_home = 4 if home_pitcher_stats else 0
    rest_days_away = 4 if away_pitcher_stats else 0

    # --- Valores base con defaults realistas ---
    era_home = _safe_float(home_pitcher_stats.get("era"), 4.20)
    whip_home = _safe_float(home_pitcher_stats.get("whip"), 1.30)
    era_away = _safe_float(away_pitcher_stats.get("era"), 4.20)
    whip_away = _safe_float(away_pitcher_stats.get("whip"), 1.30)

    ops_home = _safe_float(home_team_stats.get("hitting", {}).get("ops"), 0.720)
    ops_away = _safe_float(away_team_stats.get("hitting", {}).get("ops"), 0.720)

    # --- DIFERENCIALES (features clave para evitar sobreajuste) ---
    era_diff = era_away - era_home          # + = ventaja para local (away ERA peor)
    ops_diff = ops_home - ops_away          # + = ventaja para local
    win_pct_diff = home_win_pct - away_win_pct
    run_diff_net = home_run_diff - away_run_diff

    features = {
        "game_pk": juego["game_pk"],
        "fecha": juego["fecha"],
        "home_team_id": home_id,
        "home_team_name": juego.get("home_team_name", "Unknown"),
        "away_team_id": away_id,
        "away_team_name": juego.get("away_team_name", "Unknown"),
        "home_pitcher_id": home_pitcher,
        "home_pitcher_name": juego.get("home_pitcher_name"),
        "away_pitcher_id": away_pitcher,
        "away_pitcher_name": juego.get("away_pitcher_name"),

        # Stats individuales
        "era_pitcher_home": era_home,
        "whip_pitcher_home": whip_home,
        "ops_team_home": ops_home,
        "era_pitcher_away": era_away,
        "whip_pitcher_away": whip_away,
        "ops_team_away": ops_away,

        # Standings
        "win_pct_home": _safe_float(home_win_pct, 0.500),
        "win_pct_away": _safe_float(away_win_pct, 0.500),
        "run_diff_home": _safe_int(home_run_diff, 0),
        "run_diff_away": _safe_int(away_run_diff, 0),

        # Forma reciente
        "forma_home": _safe_float(home_forma, 0.5),
        "forma_away": _safe_float(away_forma, 0.5),

        # Descanso
        "rest_days_home": rest_days_home,
        "rest_days_away": rest_days_away,

        # Diferenciales (nuevos — evitan saturación)
        "era_diff": era_diff,
        "ops_diff": ops_diff,
        "win_pct_diff": win_pct_diff,
        "run_diff_net": run_diff_net,
    }

    logger.info(f"Features OK: {features['away_team_name']} @ {features['home_team_name']}")
    return features


def construir_dataset(juegos: List[Dict]) -> pd.DataFrame:
    """Construye DataFrame de features con manejo robusto de errores."""
    standings_df = obtener_standings()

    registros = []
    for juego in juegos:
        try:
            feat = construir_features_juego(juego, standings_df)
            registros.append(feat)
        except Exception as e:
            logger.error(f"Error en juego {juego.get('game_pk')}: {e}")

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    logger.info(f"Dataset: {len(df)} filas × {len(df.columns)} cols")
    return df
