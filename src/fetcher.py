"""
========================================
MLB Predictor — Ingesta de Datos (Fetcher)
========================================
Consulta robusta a la API oficial de MLB.
Incluye extracción de IDs de equipos en el historial para soportar 
métricas dinámicas de serie y fatiga de bullpen.
"""

import requests
import time
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from src.config import MLB_API_BASE, MLB_SPORT_ID, MLB_LEAGUE_IDS, TEMPORADA_ACTUAL, DATA_RAW
from src.utils import logger, guardar_json

HEADERS = {
    "User-Agent": "MLB-Predictor/1.0 (GitHub-Actions; research)",
    "Accept": "application/json"
}


def _get(url: str, params: Optional[dict] = None, max_reintentos: int = 3) -> Optional[dict]:
    """Wrapper robusto para GET con reintentos exponenciales."""
    for intento in range(max_reintentos):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Reintento {intento + 1}/{max_reintentos}: {url} | {e}")
            time.sleep(2 ** intento)
    logger.error(f"Fallo definitivo: {url}")
    return None


def obtener_calendario_diario(fecha: str, temporada: int = TEMPORADA_ACTUAL, solo_futuros: bool = True) -> List[Dict]:
    """
    Obtiene juegos para una fecha dada.
    Si solo_futuros=True, extrae probablePitcher para partidos programados.
    Si solo_futuros=False, omite hydrate para eficientar ingesta histórica.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "date": fecha,
        "sportId": MLB_SPORT_ID,
        "gameTypes": "R",
        "fields": "dates,games,gamePk,gameDate,status,abstractGameState,teams,away,home,team,id,name,venue,name"
    }
    if solo_futuros:
        params["hydrate"] = "probablePitcher"
        params["fields"] = "dates,games,gamePk,gameDate,status,abstractGameState,teams,away,home,team,id,name,probablePitcher,id,fullName,venue,name"

    data = _get(url, params)
    if not data or "dates" not in data or not data["dates"]:
        return []

    juegos = []
    for fecha_data in data["dates"]:
        for juego in fecha_data.get("games", []):
            estado = juego.get("status", {}).get("abstractGameState", "")
            if estado not in ["Final", "Live", "Preview", "Scheduled", "Pre-Game"]:
                continue

            registro = {
                "game_pk": juego.get("gamePk"),
                "fecha": fecha,
                "temporada": temporada,
                "estado": estado,
                "venue": juego.get("venue", {}).get("name", "Desconocido"),
                "away_team_id": juego.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                "away_team_name": juego.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                "home_team_id": juego.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                "home_team_name": juego.get("teams", {}).get("home", {}).get("team", {}).get("name"),
            }

            if solo_futuros:
                registro["away_pitcher_id"] = juego.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("id")
                registro["away_pitcher_name"] = juego.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName")
                registro["home_pitcher_id"] = juego.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("id")
                registro["home_pitcher_name"] = juego.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName")
            else:
                registro["away_pitcher_id"] = None
                registro["home_pitcher_id"] = None
                registro["away_pitcher_name"] = None
                registro["home_pitcher_name"] = None

            juegos.append(registro)

    logger.info(f"{len(juegos)} juegos en {fecha}")
    return juegos


def obtener_pitchers_reales(game_pk: int) -> Dict:
    """
    Obtiene los IDs de los lanzadores abridores reales desde el boxscore oficial.
    """
    url = f"{MLB_API_BASE}/game/{game_pk}/boxscore"
    data = _get(url)
    if not data:
        return {}

    try:
        away_pitcher = data.get("teams", {}).get("away", {}).get("pitchers", [None])[0]
        home_pitcher = data.get("teams", {}).get("home", {}).get("pitchers", [None])[0]
        
        away_players = data.get("teams", {}).get("away", {}).get("players", {})
        home_players = data.get("teams", {}).get("home", {}).get("players", {})

        away_name = None
        home_name = None
        if away_pitcher and f"ID{away_pitcher}" in away_players:
            away_name = away_players[f"ID{away_pitcher}"].get("person", {}).get("fullName")
        if home_pitcher and f"ID{home_pitcher}" in home_players:
            home_name = home_players[f"ID{home_pitcher}"].get("person", {}).get("fullName")

        return {
            "away_pitcher_id": away_pitcher,
            "away_pitcher_name": away_name,
            "home_pitcher_id": home_pitcher,
            "home_pitcher_name": home_name
        }
    except Exception as e:
        logger.warning(f"Error obteniendo pitchers reales para {game_pk}: {e}")
        return {}


def obtener_resultado_juego(game_pk: int) -> Optional[int]:
    """
    Retorna 1 si ganó LOCAL, 0 si ganó VISITANTE, None si el dato es inconsistente.
    """
    url = f"{MLB_API_BASE}/game/{game_pk}/linescore"
    data = _get(url)
    if not data:
        return None

    try:
        teams = data.get("teams", {})
        home_runs = teams.get("home", {}).get("runs", 0)
        away_runs = teams.get("away", {}).get("runs", 0)

        if home_runs is None or away_runs is None:
            return None
        if home_runs > away_runs:
            return 1
        elif away_runs > home_runs:
            return 0
        else:
            return None
    except Exception as e:
        logger.warning(f"Error parseando resultado de {game_pk}: {e}")
        return None


def obtener_estadisticas_equipo(team_id: int, temporada: int = TEMPORADA_ACTUAL) -> Dict:
    """Estadísticas acumuladas de bateo y picheo de un equipo."""
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

    return stats


def obtener_estadisticas_lanzador(person_id: int, temporada: int = TEMPORADA_ACTUAL) -> Dict:
    """Estadísticas individuales de un lanzador."""
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

    return splits[0].get("stat", {})


def obtener_standings(temporada: int = TEMPORADA_ACTUAL) -> pd.DataFrame:
    """Tabla de posiciones actual con porcentaje de victorias y diferencial de carreras."""
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

    return pd.DataFrame(filas)


def obtener_historial_juegos(team_id: int, dias: int = 10, temporada: int = TEMPORADA_ACTUAL) -> List[Dict]:
    """
    Extrae los últimos N juegos finalizados de un equipo.
    Incluye home_team_id y away_team_id para la detección de H2H y patrones de serie.
    """
    hoy = datetime.now()
    fecha_fin = hoy.strftime("%Y-%m-%d")
    fecha_ini = (hoy - timedelta(days=dias + 20)).strftime("%Y-%m-%d")

    url = f"{MLB_API_BASE}/schedule"
    params = {
        "teamId": team_id,
        "startDate": fecha_ini,
        "endDate": fecha_fin,
        "sportId": MLB_SPORT_ID,
        "gameTypes": "R",
        "fields": "dates,games,gamePk,gameDate,status,abstractGameState,teams,away,home,team,id,score"
    }

    data = _get(url, params)
    if not data or "dates" not in data:
        return []

    juegos = []
    for fecha_data in data.get("dates", []):
        for juego in fecha_data.get("games", []):
            if juego.get("status", {}).get("abstractGameState") == "Final":
                teams = juego.get("teams", {})
                juegos.append({
                    "game_pk": juego.get("gamePk"),
                    "fecha": juego.get("gameDate")[:10],
                    "home_team_id": teams.get("home", {}).get("team", {}).get("id"),
                    "away_team_id": teams.get("away", {}).get("team", {}).get("id"),
                    "home_score": teams.get("home", {}).get("score", 0),
                    "away_score": teams.get("away", {}).get("score", 0),
                })

    juegos = sorted(juegos, key=lambda x: x["fecha"], reverse=True)[:dias]
    return juegos
