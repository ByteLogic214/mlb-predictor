"""
========================================
MLB Predictor — Ingenería de Features
========================================
Procesamiento de datos con Bayesian Shrinkage para lanzadores.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from src.config import (
    FEATURES_MODELO, LEAGUE_ERA_BASELINE, LEAGUE_WHIP_BASELINE, BAYESIAN_IP_WEIGHT
)
from src.fetcher import (
    obtener_estadisticas_lanzador,
    obtener_estadisticas_equipo,
    obtener_standings,
    obtener_historial_juegos
)
from src.utils import logger


def bayesian_adjusted_stat(stat_val: float, ip: float, baseline: float, weight: float = BAYESIAN_IP_WEIGHT) -> float:
    """Aplica suavizado bayesiano empujando las métricas de muestras pequeñas hacia la media de la liga."""
    if pd.isna(ip) or ip <= 0 or pd.isna(stat_val):
        return baseline
    return float((stat_val * ip + baseline * weight) / (ip + weight))


def construir_dataset(juegos: List[Dict]) -> pd.DataFrame:
    """Construye las features para una lista de juegos de la MLB."""
    if not juegos:
        return pd.DataFrame()

    standings = obtener_standings()
    standings_dict = {}
    if not standings.empty:
        for _, row in standings.iterrows():
            standings_dict[row["team_id"]] = row.to_dict()

    dataset = []

    for juego in juegos:
        try:
            home_id = juego.get("home_team_id")
            away_id = juego.get("away_team_id")

            if not home_id or not away_id:
                continue

            # Stats Equipos
            home_team_stats = obtener_estadisticas_equipo(home_id)
            away_team_stats = obtener_estadisticas_equipo(away_id)

            ops_home = float(home_team_stats.get("hitting", {}).get("ops", 0.735))
            ops_away = float(away_team_stats.get("hitting", {}).get("ops", 0.725))

            # Pitchers Abridores
            home_p_id = juego.get("home_pitcher_id")
            away_p_id = juego.get("away_pitcher_id")

            home_p_stats = obtener_estadisticas_lanzador(home_p_id) if home_p_id else {}
            away_p_stats = obtener_estadisticas_lanzador(away_p_id) if away_p_id else {}

            ip_home = float(home_p_stats.get("inningsPitched", 0))
            era_raw_home = float(home_p_stats.get("era", LEAGUE_ERA_BASELINE))
            whip_raw_home = float(home_p_stats.get("whip", LEAGUE_WHIP_BASELINE))

            ip_away = float(away_p_stats.get("inningsPitched", 0))
            era_raw_away = float(away_p_stats.get("era", LEAGUE_ERA_BASELINE))
            whip_raw_away = float(away_p_stats.get("whip", LEAGUE_WHIP_BASELINE))

            # Suavizado Bayesiano para prevenir distorsiones por muestra pequeña
            era_home = bayesian_adjusted_stat(era_raw_home, ip_home, LEAGUE_ERA_BASELINE)
            whip_home = bayesian_adjusted_stat(whip_raw_home, ip_home, LEAGUE_WHIP_BASELINE)

            era_away = bayesian_adjusted_stat(era_raw_away, ip_away, LEAGUE_ERA_BASELINE)
            whip_away = bayesian_adjusted_stat(whip_raw_away, ip_away, LEAGUE_WHIP_BASELINE)

            # Standings Metrics
            home_stand = standings_dict.get(home_id, {})
            away_stand = standings_dict.get(away_id, {})

            win_pct_home = float(home_stand.get("win_pct", 0.500))
            win_pct_away = float(away_stand.get("win_pct", 0.500))

            run_diff_home = float(home_stand.get("run_diff", 0))
            run_diff_away = float(away_stand.get("run_diff", 0))

            # Forma Reciente (Últimos 10 juegos)
            hist_home = obtener_historial_juegos(home_id, dias=10)
            hist_away = obtener_historial_juegos(away_id, dias=10)

            wins_h = sum(1 for g in hist_home if g.get("home_score", 0) > g.get("away_score", 0))
            forma_home = wins_h / len(hist_home) if hist_home else 0.500

            wins_a = sum(1 for g in hist_away if g.get("away_score", 0) > g.get("home_score", 0))
            forma_away = wins_a / len(hist_away) if hist_away else 0.500

            # Diferenciales
            era_diff = era_away - era_home
            ops_diff = ops_home - ops_away
            win_pct_diff = win_pct_home - win_pct_away
            run_diff_net = run_diff_home - run_diff_away

            dataset.append({
                "game_pk": juego.get("game_pk"),
                "fecha": juego.get("fecha"),
                "home_team_name": juego.get("home_team_name"),
                "away_team_name": juego.get("away_team_name"),
                "home_pitcher_name": juego.get("home_pitcher_name", "TBD"),
                "away_pitcher_name": juego.get("away_pitcher_name", "TBD"),
                "era_pitcher_home": era_home,
                "whip_pitcher_home": whip_home,
                "ops_team_home": ops_home,
                "era_pitcher_away": era_away,
                "whip_pitcher_away": whip_away,
                "ops_team_away": ops_away,
                "win_pct_home": win_pct_home,
                "win_pct_away": win_pct_away,
                "run_diff_home": run_diff_home,
                "run_diff_away": run_diff_away,
                "forma_home": forma_home,
                "forma_away": forma_away,
                "rest_days_home": 4,
                "rest_days_away": 4,
                "era_diff": era_diff,
                "ops_diff": ops_diff,
                "win_pct_diff": win_pct_diff,
                "run_diff_net": run_diff_net
            })
        except Exception as e:
            logger.warning(f"Error procesando juego {juego.get('game_pk')}: {e}")
            continue

    return pd.DataFrame(dataset)
