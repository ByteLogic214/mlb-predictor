"""
========================================
MLB Predictor — Ingesta de Datos (Fetcher)
========================================
Consulta robusta a la API gratuita de MLB (statsapi.mlb.com).
Obtiene: calendario diario, lanzadores abridores, estadísticas de equipos y pitchers.
"""

import requests
import time
import pandas as pd
from typing import List, Dict, Optional
from src.config import MLB_API_BASE, MLB_SPORT_ID, MLB_LEAGUE_IDS, TEMPORADA_ACTUAL, DATA_RAW
from src.utils import logger, guardar_json

HEADERS = {
    "User-Agent": "MLB-Predictor/1.0 (GitHub-Actions; research)",
    "Accept": "application/json"
}


def _get(url: str, params: Optional[dict] = None, max_reintentos: int = 3) -> Optional[dict]:
    """
    Wrapper robusto para GET con reintentos exponenciales y manejo de errores.
    """
    for intento in range(max_reintentos):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Intento {intento + 1}/{max_reintentos} fallido: {url} | Error: {e}")
            time.sleep(2 ** intento)
    logger.error(f"Fallo definitivo al consultar: {url}")
    return None


def obtener_calendario_diario(fecha: str, temporada: int = TEMPORADA_ACTUAL) -> List[Dict]:
    """
    Obtiene el calendario de juegos para una fecha específica.
    Incluye los lanzadores abridores probables (probablePitcher).
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "date": fecha,
        "sportId": MLB_SPORT_ID,
        "hydrate": "probablePitcher",
        "fields": "dates,games,gamePk,gameDate,teams,away,home,team,id,name,probablePitcher,id,fullName,venue,name"
    }

    data = _get(url, params)
    if not data or "dates" not in data or not data["dates"]:
        logger.warning(f"No hay juegos programados para {fecha}")
        return []

    juegos = []
    for fecha_data in data["dates"]:
        for juego in fecha_data.get("games", []):
            registro = {
                "game_pk": juego.get("gamePk"),
                "fecha": fecha,
                "temporada": temporada,
                "venue": juego.get("venue", {}).get("name", "Desconocido"),
                "away_team_id": juego.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                "away_team_name": juego.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                "home_team_id": juego.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                "home_team_name": juego.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                "away_pitcher_id": juego.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id"),
                "away_pitcher_name": juego.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName"),
                "home_pitcher_id": juego.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id"),
                "home_pitcher_name": juego.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName"),
            }
            juegos.append(registro)

    logger.info(f"{len(juegos)} juegos encontrados para {fecha}")
    guardar_json({"juegos": juegos}, f"{DATA_RAW}/calendario_{fecha}.json")
    return juegos


def obtener_estadisticas_equipo(team_id: int, temporada: int = TEMPORADA_ACTUAL) -> Dict:
    """
    Obtiene estadísticas ofensivas (hitting) y de pitcheo (pitching) de un equipo.
    """
    stats = {"hitting": {}, "pitching": {}}

    for grupo in ["hitting", "pitching"]:
        url = f"{MLB_API_BASE}/stats"
        params = {
            "stats": "season",
            "group": grupo,
            "season": temporada,
            "sportIds": MLB_SPORT_ID,
            "teamId": team_id,
            "fields": "stats,splits,stat,avg,ops,era,whip,wins,losses,runs,runsScored"
        }
        data = _get(url, params)
        if data and "stats" in data and data["stats"]:
            splits = data["stats"][0].get("splits", [])
            if splits:
                stats[grupo] = splits[0].get("stat", {})

    # Guardar backup
    guardar_json(stats, f"{DATA_RAW}/team_{team_id}_stats.json")
    return stats


def obtener_estadisticas_lanzador(person_id: int, temporada: int = TEMPORADA_ACTUAL) -> Dict:
    """
    Obtiene estadísticas de pitcheo de un lanzador específico.
    """
    if not person_id:
        return {}

    url = f"{MLB_API_BASE}/people/{person_id}/stats"
    params = {
        "stats": "season",
        "group": "pitching",
        "season": temporada,
        "sportIds": MLB_SPORT_ID,
        "fields": "stats,splits,stat,era,whip,wins,losses,strikeouts,inningsPitched,earnedRuns"
    }

    data = _get(url, params)
    if not data or "stats" not in data or not data["stats"]:
        return {}

    splits = data["stats"][0].get("splits", [])
    if not splits:
        return {}

    stats = splits[0].get("stat", {})
    guardar_json(stats, f"{DATA_RAW}/pitcher_{person_id}_stats.json")
    return stats


def obtener_standings(temporada: int = TEMPORADA_ACTUAL) -> pd.DataFrame:
    """
    Obtiene las posiciones actuales para calcular % victorias y diferencial de carreras.
    """
    url = f"{MLB_API_BASE}/standings"
    params = {
        "leagueId": MLB_LEAGUE_IDS,
        "season": temporada,
        "sportIds": MLB_SPORT_ID,
        "fields": "records,teamRecords,team,id,name,wins,losses,winningPercentage,runsScored,runsAllowed"
    }

    data = _get(url, params)
    if not data or "records" not in data:
        return pd.DataFrame()

    filas = []
    for division in data.get("records", []):
        for equipo in division.get("teamRecords", []):
            team = equipo.get("team", {})
            filas.append({
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "wins": equipo.get("wins", 0),
                "losses": equipo.get("losses", 0),
                "win_pct": float(equipo.get("winningPercentage", 0)),
                "runs_scored": equipo.get("runsScored", 0),
                "runs_allowed": equipo.get("runsAllowed", 0),
                "run_diff": equipo.get("runsScored", 0) - equipo.get("runsAllowed", 0)
            })

    df = pd.DataFrame(filas)
    guardar_json(filas, f"{DATA_RAW}/standings_{temporada}.json")
    return df


def obtener_historial_juegos(team_id: int, dias: int = 10, temporada: int = TEMPORADA_ACTUAL) -> List[Dict]:
    """
    Obtiene los últimos N juegos finalizados de un equipo para calcular forma reciente.
    """
    from datetime import datetime, timedelta
    fecha_fin = datetime.now().strftime("%Y-%m-%d")
    fecha_ini = (datetime.now() - timedelta(days=dias + 15)).strftime("%Y-%m-%d")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "teamId": team_id,
        "startDate": fecha_ini,
        "endDate": fecha_fin,
        "sportId": MLB_SPORT_ID,
        "gameTypes": "R",  # Solo temporada regular
        "fields": "dates,games,gamePk,gameDate,status,abstractGameState,teams,away,home,score"
    }

    data = _get(url, params)
    if not data or "dates" not in data:
        return []

    juegos = []
    for fecha_data in data.get("dates", []):
        for juego in fecha_data.get("games", []):
            if juego.get("status", {}).get("abstractGameState") == "Final":
                juegos.append({
                    "game_pk": juego.get("gamePk"),
                    "fecha": juego.get("gameDate"),
                    "home_score": juego.get("teams", {}).get("home", {}).get("score", 0),
                    "away_score": juego.get("teams", {}).get("away", {}).get("score", 0),
                })

    # Ordenar por fecha descendente y tomar los últimos N
    juegos = sorted(juegos, key=lambda x: x["fecha"], reverse=True)[:dias]
    return juegos
