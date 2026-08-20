"""
========================================
MLB Predictor — Ingeniería de Features
========================================
Procesamiento de datos con Bayesian Shrinkage y variables dinámicas de serie/bullpen.
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


def obtener_contexto_serie(home_id: int, away_id: int, fecha: str) -> dict:
    """Detecta si el juego pertenece a una serie en curso y quién ganó el partido anterior."""
    hist_home = obtener_historial_juegos(home_id, dias=5)
    
    partidos_serie = [
        g for g in hist_home 
        if (g.get("home_team_id") == away_id or g.get("away_team_id") == away_id)
        and g.get("fecha") < fecha
    ]
    
    if not partidos_serie:
        return {"game_num_series": 1, "prev_game_winner_home": 0}
    
    partidos_serie = sorted(partidos_serie, key=lambda x: x["fecha"], reverse=True)
    ultimo_juego = partidos_serie[0]
    
    gano_local_previo = 1 if ultimo_juego.get("home_score", 0) > ultimo_juego.get("away_score", 0) else 0
    
    return {
        "game_num_series": len(partidos_serie) + 1,
        "prev_game_winner_home": gano_local_previo
    }


def calcular_uso_bullpen(team_id: int) -> float:
    """Calcula el índice de desgaste del bullpen en los últimos 3 días."""
    juegos_recientes = obtener_historial_juegos(team_id, dias=3)
    if not juegos_recientes:
        return 0.0
    
    carga = 0.0
    for g in juegos_recientes:
        diff = abs(g.get("home_score", 0) - g.get("away_score", 0))
        if diff <= 3:
            carga += 1.5
        else:
            carga += 0.5
            
    return carga


def construir_dataset(juegos: List[Dict]) -> pd.DataFrame:
    """Construye las features completas para una lista de juegos de la MLB."""
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
            fecha_juego = juego.get("fecha")

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

            # Suavizado Bayesiano
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

            # Forma Reciente
            hist_home = obtener_historial_juegos(home_id, dias=10)
            hist_away = obtener_historial_juegos(away_id, dias=10)

            wins_h = sum(1 for g in hist_home if g.get("home_score", 0) > g.get("away_score", 0))
            forma_home = wins_h / len(hist_home) if hist_home else 0.500

            wins_a = sum(1 for g in hist_away if g.get("away_score", 0) > g.get("home_score", 0))
            forma_away = wins_a / len(hist_away) if hist_away else 0.500

            # Variables Dinámicas
            ctx_serie = obtener_contexto_serie(home_id, away_id, fecha_juego)
            bullpen_home = calcular_uso_bullpen(home_id)
            bullpen_away = calcular_uso_bullpen(away_id)

            # Diferenciales
            era_diff = era_away - era_home
            ops_diff = ops_home - ops_away
            win_pct_diff = win_pct_home - win_pct_away
            run_diff_net = run_diff_home - run_diff_away
            bullpen_load_diff = bullpen_away - bullpen_home

            dataset.append({
                "game_pk": juego.get("game_pk"),
                "fecha": fecha_juego,
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
                "run_diff_net": run_diff_net,
                "game_num_series": ctx_serie["game_num_series"],
                "prev_game_winner_home": ctx_serie["prev_game_winner_home"],
                "bullpen_load_diff": bullpen_load_diff
            })
        except Exception as e:
            logger.warning(f"Error procesando juego {juego.get('game_pk')}: {e}")
            continue

    return pd.DataFrame(dataset)
