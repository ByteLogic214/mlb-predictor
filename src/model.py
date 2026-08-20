"""
========================================
MLB Predictor — Modelo y Predicción
========================================
Entrena un clasificador XGBoost con features de rendimiento ajustadas.
Genera predicciones estructuradas en JSON y Markdown.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from src.config import (
    FEATURES_MODELO, MODELS_DIR, DATA_PREDICTIONS, DATA_PROCESSED, TEMPORADA_ACTUAL
)
from src.fetcher import obtener_calendario_diario, obtener_historial_juegos
from src.features import construir_dataset
from src.utils import logger, guardar_json


RUTA_MODELO = os.path.join(MODELS_DIR, "xgboost_mlb.pkl")
RUTA_HISTORIAL = os.path.join(DATA_PROCESSED, "historial_entrenamiento.csv")


def _entrenar_con_historial(temporada: int = TEMPORADA_ACTUAL) -> xgb.XGBClassifier:
    """
    Entrena el modelo usando juegos ya finalizados de la temporada.
    Como no tenemos un dataset histórico local, construimos uno dinámicamente
    consultando juegos pasados de la API.
    """
    logger.info("Construyendo dataset histórico para entrenamiento...")

    # Obtener juegos de los últimos 30 días para entrenar
    from datetime import datetime
    hoy = datetime.now()
    juegos_historicos = []

    for i in range(1, 31):
        fecha_pasada = (hoy - timedelta(days=i)).strftime("%Y-%m-%d")
        juegos_dia = obtener_calendario_diario(fecha_pasada, temporada)
        if juegos_dia:
            juegos_historicos.extend(juegos_dia)

    if not juegos_historicos:
        logger.warning("No se encontraron juegos históricos. Entrenando con datos sintéticos mínimos.")
        return _entrenar_minimo()

    # Construir features
    df = construir_dataset(juegos_historicos)
    if df.empty:
        return _entrenar_minimo()

    # Para cada juego, necesitamos la etiqueta (ganó local o visitante)
    # Consultamos el resultado real de cada juego
    etiquetas = []
    for _, row in df.iterrows():
        resultado = _obtener_resultado_juego(row["game_pk"])
        if resultado is not None:
            etiquetas.append(resultado)
        else:
            etiquetas.append(None)

    df["target"] = etiquetas
    df = df.dropna(subset=["target"])

    if len(df) < 10:
        logger.warning("Muestras insuficientes. Usando modelo mínimo.")
        return _entrenar_minimo()

    X = df[FEATURES_MODELO]
    y = df["target"].astype(int)

    # División temporal (no aleatoria para series de tiempo)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    modelo = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    modelo.fit(X_train, y_train)

    if len(y_test) > 0:
        preds = modelo.predict(X_test)
        acc = accuracy_score(y_test, preds)
        loss = log_loss(y_test, modelo.predict_proba(X_test))
        logger.info(f"Modelo entrenado — Accuracy: {acc:.3f} | LogLoss: {loss:.3f}")

    # Guardar modelo y datos
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    df.to_csv(RUTA_HISTORIAL, index=False)
    logger.info(f"Modelo guardado en {RUTA_MODELO}")

    return modelo


def _obtener_resultado_juego(game_pk: int) -> int:
    """
    Consulta el resultado final de un juego. Retorna 1 si ganó el local, 0 si ganó el visitante.
    """
    import requests
    from src.config import MLB_API_BASE
    from src.fetcher import HEADERS

    url = f"{MLB_API_BASE}/game/{game_pk}/feed/live"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        data = resp.json()
        estado = data.get("gameData", {}).get("status", {}).get("abstractGameState")
        if estado != "Final":
            return None

        home_score = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("home", {}).get("runs", 0)
        away_score = data.get("liveData", {}).get("linescore", {}).get("teams", {}).get("away", {}).get("runs", 0)
        return 1 if home_score > away_score else 0
    except Exception:
        return None


def _entrenar_minimo() -> xgb.XGBClassifier:
    """
    Modelo de respaldo con datos sintéticos mínimos para evitar fallo total.
    """
    logger.warning("Entrenando modelo mínimo de respaldo...")
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "era_pitcher_home": np.random.normal(4.0, 1.0, n),
        "whip_pitcher_home": np.random.normal(1.25, 0.15, n),
        "ops_team_home": np.random.normal(0.730, 0.040, n),
        "era_pitcher_away": np.random.normal(4.2, 1.0, n),
        "whip_pitcher_away": np.random.normal(1.28, 0.15, n),
        "ops_team_away": np.random.normal(0.720, 0.040, n),
        "win_pct_home": np.random.normal(0.52, 0.08, n),
        "win_pct_away": np.random.normal(0.48, 0.08, n),
        "run_diff_home": np.random.normal(10, 30, n),
        "run_diff_away": np.random.normal(-5, 30, n),
        "rest_days_home": np.random.choice([3, 4, 5, 6], n),
        "rest_days_away": np.random.choice([3, 4, 5, 6], n),
        "is_home_advantage": np.ones(n)
    })
    y = (X["win_pct_home"] + X["ops_team_home"] * 0.5 - X["era_pitcher_home"] * 0.1 > 0.5).astype(int)

    modelo = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        objective="binary:logistic", use_label_encoder=False
    )
    modelo.fit(X, y)

    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    return modelo


def cargar_o_entrenar_modelo() -> xgb.XGBClassifier:
    """
    Carga el modelo existente o entrena uno nuevo si no existe.
    """
    if os.path.exists(RUTA_MODELO):
        logger.info("Cargando modelo existente...")
        with open(RUTA_MODELO, "rb") as f:
            return pickle.load(f)
    else:
        logger.info("No se encontró modelo. Iniciando entrenamiento...")
        return _entrenar_con_historial()


def predecir_juegos(df_features: pd.DataFrame, modelo: xgb.XGBClassifier) -> List[Dict]:
    """
    Genera predicciones para un DataFrame de features.
    Retorna lista de diccionarios con probabilidades y picks.
    """
    if df_features.empty:
        return []

    X = df_features[FEATURES_MODELO]
    probs = modelo.predict_proba(X)[:, 1]  # Probabilidad de victoria del local
    preds = modelo.predict(X)

    resultados = []
    for i, (_, row) in enumerate(df_features.iterrows()):
        prob_home = round(float(probs[i]), 3)
        prob_away = round(1.0 - prob_home, 3)
        pick = "HOME" if preds[i] == 1 else "AWAY"
        confianza = "ALTA" if max(prob_home, prob_away) > 0.65 else "MEDIA" if max(prob_home, prob_away) > 0.55 else "BAJA"

        resultados.append({
            "game_pk": int(row["game_pk"]),
            "fecha": row["fecha"],
            "matchup": f"{row['away_team_name']} @ {row['home_team_name']}",
            "prediccion": pick,
            "prob_home_win": prob_home,
            "prob_away_win": prob_away,
            "confianza": confianza,
            "home_pitcher": row.get("home_pitcher_name"),
            "away_pitcher": row.get("away_pitcher_name"),
            "features_destacadas": {
                "era_home": row.get("era_pitcher_home"),
                "era_away": row.get("era_pitcher_away"),
                "ops_home": row.get("ops_team_home"),
                "ops_away": row.get("ops_team_away"),
                "win_pct_home": row.get("win_pct_home"),
                "win_pct_away": row.get("win_pct_away")
            }
        })

    return resultados


def generar_reporte(resultados: List[Dict], fecha: str) -> str:
    """
    Genera un reporte Markdown legible con las predicciones del día.
    """
    lineas = [
        f"# ⚾ Predicciones MLB — {fecha}",
        "",
        "> Generado automáticamente por MLB Predictor vía GitHub Actions",
        f"> Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
        "## 📊 Resumen del Día",
        f"**Total de juegos analizados:** {len(resultados)}",
        f"**Picks de alta confianza (>65%):** {sum(1 for r in resultados if r['confianza'] == 'ALTA')}",
        "",
        "---",
        "",
        "## 🎯 Predicciones Detalladas",
        "",
        "| Matchup | Pick | Prob. Local | Prob. Visitante | Confianza | Lanzador Local | Lanzador Visitante |",
        "|---------|------|-------------|-----------------|-----------|----------------|--------------------|",
    ]

    for r in resultados:
        lineas.append(
            f"| {r['matchup']} | **{r['prediccion']}** | {r['prob_home_win']:.1%} | {r['prob_away_win']:.1%} | {r['confianza']} | {r.get('home_pitcher', 'TBD')} | {r.get('away_pitcher', 'TBD')} |"
        )

    lineas.extend([
        "",
        "---",
        "",
        "## 🧠 Metodología",
        "",
        "El modelo utiliza **XGBoost** entrenado con las siguientes variables:",
        "",
        "- **ERA** y **WHIP** del lanzador abridor",
        "- **OPS** (On-base Plus Slugging) del equipo",
        "- **Win %** y **diferencial de carreras** del standings",
        "- **Forma reciente** (últimos 10 juegos)",
        "- **Ventaja de localía**",
        "",
        "Fuente de datos: [MLB Stats API](https://statsapi.mlb.com) (API oficial y gratuita de MLB).",
        "",
        "---",
        "",
        "*Este reporte se actualiza automáticamente todos los días.*"
    ])

    return "\n".join(lineas)


def ejecutar_pipeline(fecha: str = None):
    """
    Pipeline completo: ingesta → features → predicción → reporte.
    """
    if fecha is None:
        fecha = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"🚀 Iniciando pipeline para {fecha}")

    # 1. Ingesta
    juegos = obtener_calendario_diario(fecha)
    if not juegos:
        logger.info("No hay juegos para predecir hoy.")
        return

    # 2. Features
    df = construir_dataset(juegos)

    # 3. Modelo
    modelo = cargar_o_entrenar_modelo()

    # 4. Predicción
    resultados = predecir_juegos(df, modelo)

    # 5. Guardar JSON
    ruta_json = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.json")
    guardar_json({"fecha": fecha, "predicciones": resultados}, ruta_json)

    # 6. Generar y guardar Markdown
    reporte_md = generar_reporte(resultados, fecha)
    ruta_md = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(reporte_md)
    logger.info(f"Reporte Markdown guardado: {ruta_md}")

    # 7. Guardar también como README diario (sobrescribe el último)
    ruta_readme = os.path.join(DATA_PREDICTIONS, "README.md")
    with open(ruta_readme, "w", encoding="utf-8") as f:
        f.write(reporte_md)
    logger.info("Pipeline completado exitosamente ✅")
