"""
========================================
MLB Predictor — Modelo y Predicción (CORREGIDO)
========================================
- Alineación fija de etiquetas (target asignado fila por fila, nunca por lista desplazada)
- XGBoost regularizado (max_depth=4, reg_alpha, reg_lambda)
- Clipping de probabilidades [0.20, 0.80] (reality check del béisbol)
- Calibración logística post-entrenamiento
- Eliminación de feature constante is_home_advantage
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from src.config import (
    FEATURES_MODELO, MODELS_DIR, DATA_PREDICTIONS, DATA_PROCESSED,
    TEMPORADA_ACTUAL, PROB_MIN, PROB_MAX,
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE_BYTREE,
    REG_ALPHA, REG_LAMBDA, RANDOM_STATE
)
from src.fetcher import (
    obtener_calendario_diario,
    obtener_resultado_juego,
    obtener_pitchers_reales
)
from src.features import construir_dataset
from src.utils import logger, guardar_json


RUTA_MODELO = os.path.join(MODELS_DIR, "xgboost_mlb.pkl")
RUTA_CALIBRADOR = os.path.join(MODELS_DIR, "calibrador_mlb.pkl")
RUTA_HISTORIAL = os.path.join(DATA_PROCESSED, "historial_entrenamiento.csv")


def _clip_probabilidad(prob: float) -> float:
    """Limita probabilidades a rangos realistas del béisbol."""
    return max(PROB_MIN, min(PROB_MAX, prob))


def _entrenar_con_historial(temporada: int = TEMPORADA_ACTUAL) -> tuple:
    """
    Entrena modelo con juegos ya finalizados.
    CORRECCIÓN CRÍTICA: El target se asigna FILA POR FILA usando el game_pk,
    nunca por lista desplazada. Se filtran juegos sin resultado antes de construir features.
    """
    logger.info("Construyendo dataset histórico...")

    hoy = datetime.now()
    juegos_con_resultado = []

    # Rango de 45 días hacia atrás para acumular muestras suficientes
    for i in range(1, 46):
        fecha_pasada = (hoy - timedelta(days=i)).strftime("%Y-%m-%d")
        juegos_dia = obtener_calendario_diario(fecha_pasada, temporada, solo_futuros=False)

        for juego in juegos_dia:
            # Solo juegos finalizados
            if juego.get("estado") != "Final":
                continue

            # Obtener resultado REAL
            target = obtener_resultado_juego(juego["game_pk"])
            if target is None:
                continue

            # Obtener pitchers REALES que lanzaron (no probables)
            pitchers = obtener_pitchers_reales(juego["game_pk"])
            juego.update(pitchers)

            # Guardar target en el propio dict para evitar desalineación
            juego["target"] = target
            juegos_con_resultado.append(juego)

    if len(juegos_con_resultado) < 15:
        logger.warning(f"Solo {len(juegos_con_resultado)} juegos válidos. Entrenando modelo mínimo.")
        return _entrenar_minimo()

    # Construir features SOLO para juegos con resultado confirmado
    df = construir_dataset(juegos_con_resultado)
    if df.empty or len(df) < 10:
        return _entrenar_minimo()

    # Extraer target del DataFrame (ya alineado porque venía del mismo dict)
    # Los juegos que fallaron en features se perdieron en construir_dataset,
    # pero como filtramos antes, df y targets están alineados por game_pk
    targets = []
    for _, row in df.iterrows():
        # Buscar el target original por game_pk
        original = next((j for j in juegos_con_resultado if j["game_pk"] == row["game_pk"]), None)
        if original:
            targets.append(original["target"])
        else:
            targets.append(None)

    df["target"] = targets
    df = df.dropna(subset=["target"])

    if len(df) < 10:
        return _entrenar_minimo()

    # Ordenar cronológicamente para split temporal
    df = df.sort_values("fecha").reset_index(drop=True)

    X = df[FEATURES_MODELO].astype(float)
    y = df["target"].astype(int)

    # Split temporal 80/20 (no aleatorio)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # XGBoost con regularización agresiva para evitar probabilidades extremas
    modelo = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        n_jobs=2
    )

    modelo.fit(X_train, y_train)

    # Métricas
    if len(y_test) > 0:
        probs_test = modelo.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds_test)
        loss = log_loss(y_test, probs_test)
        brier = brier_score_loss(y_test, probs_test)
        logger.info(f"Validación — Acc: {acc:.3f} | LogLoss: {loss:.3f} | Brier: {brier:.3f}")

    # Calibración logística (Platt scaling) para probabilidades bien calibradas
    calibrador = LogisticRegression(C=1.0, max_iter=1000)
    calibrador.fit(modelo.predict_proba(X_train)[:, 1].reshape(-1, 1), y_train)

    # Guardar
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    with open(RUTA_CALIBRADOR, "wb") as f:
        pickle.dump(calibrador, f)
    df.to_csv(RUTA_HISTORIAL, index=False)
    logger.info(f"Modelo + calibrador guardados. Muestras: {len(df)}")

    return modelo, calibrador


def _entrenar_minimo() -> tuple:
    """
    Modelo de respaldo con datos sintéticos razonables.
    CORRECCIÓN: Distribuciones más realistas, feature is_home_advantage eliminada.
    """
    logger.warning("Entrenando modelo mínimo de respaldo...")
    np.random.seed(42)
    n = 200

    # Distribuciones realistas de MLB
    X = pd.DataFrame({
        "era_pitcher_home": np.clip(np.random.normal(4.0, 0.8, n), 2.0, 6.5),
        "whip_pitcher_home": np.clip(np.random.normal(1.28, 0.12, n), 1.0, 1.6),
        "ops_team_home": np.clip(np.random.normal(0.735, 0.035, n), 0.650, 0.820),
        "era_pitcher_away": np.clip(np.random.normal(4.2, 0.9, n), 2.0, 6.5),
        "whip_pitcher_away": np.clip(np.random.normal(1.30, 0.13, n), 1.0, 1.6),
        "ops_team_away": np.clip(np.random.normal(0.725, 0.035, n), 0.650, 0.820),
        "win_pct_home": np.clip(np.random.normal(0.515, 0.06, n), 0.350, 0.650),
        "win_pct_away": np.clip(np.random.normal(0.485, 0.06, n), 0.350, 0.650),
        "run_diff_home": np.random.normal(15, 40, n),
        "run_diff_away": np.random.normal(-10, 40, n),
        "forma_home": np.clip(np.random.normal(0.52, 0.15, n), 0.1, 0.9),
        "forma_away": np.clip(np.random.normal(0.48, 0.15, n), 0.1, 0.9),
        "rest_days_home": np.random.choice([3, 4, 5, 6], n),
        "rest_days_away": np.random.choice([3, 4, 5, 6], n),
        "era_diff": np.random.normal(-0.2, 1.0, n),
        "ops_diff": np.random.normal(0.01, 0.04, n),
        "win_pct_diff": np.random.normal(0.03, 0.08, n),
        "run_diff_net": np.random.normal(25, 50, n),
    })

    # Target: local gana si ventaja combinada es positiva (con ruido realista)
    score = (
        -0.10 * X["era_diff"]
        + 2.0 * X["ops_diff"]
        + 1.5 * X["win_pct_diff"]
        + 0.005 * X["run_diff_net"]
        + 0.3 * (X["forma_home"] - X["forma_away"])
        + np.random.normal(0, 0.3, n)
    )
    y = (score > 0).astype(int)

    modelo = xgb.XGBClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", use_label_encoder=False
    )
    modelo.fit(X, y)

    calibrador = LogisticRegression(C=1.0, max_iter=1000)
    calibrador.fit(modelo.predict_proba(X)[:, 1].reshape(-1, 1), y)

    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    with open(RUTA_CALIBRADOR, "wb") as f:
        pickle.dump(calibrador, f)

    return modelo, calibrador


def cargar_o_entrenar_modelo() -> tuple:
    """Carga modelo y calibrador, o entrena si no existen."""
    if os.path.exists(RUTA_MODELO) and os.path.exists(RUTA_CALIBRADOR):
        logger.info("Cargando modelo existente...")
        with open(RUTA_MODELO, "rb") as f:
            modelo = pickle.load(f)
        with open(RUTA_CALIBRADOR, "rb") as f:
            calibrador = pickle.load(f)
        return modelo, calibrador
    else:
        logger.info("Entrenando modelo nuevo...")
        return _entrenar_con_historial()


def predecir_juegos(df_features: pd.DataFrame, modelo, calibrador) -> List[Dict]:
    """
    Genera predicciones con clipping y calibración.
    CORRECCIÓN: prob_home_win siempre corresponde al equipo LOCAL.
    """
    if df_features.empty:
        return []

    X = df_features[FEATURES_MODELO].astype(float)
    probs_raw = modelo.predict_proba(X)[:, 1]

    # Calibrar probabilidades
    probs_cal = calibrador.predict_proba(probs_raw.reshape(-1, 1))[:, 1]

    # Clip a rangos realistas del béisbol
    probs = np.array([_clip_probabilidad(p) for p in probs_cal])
    preds = (probs >= 0.5).astype(int)

    resultados = []
    for i, (_, row) in enumerate(df_features.iterrows()):
        prob_home = round(float(probs[i]), 3)
        prob_away = round(1.0 - prob_home, 3)
        pick = "HOME" if preds[i] == 1 else "AWAY"

        # Determinar confianza basada en distancia al 50%
        margen = abs(prob_home - 0.5)
        if margen > 0.15:
            confianza = "ALTA"
        elif margen > 0.08:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"

        resultados.append({
            "game_pk": int(row["game_pk"]),
            "fecha": row["fecha"],
            "matchup": f"{row['away_team_name']} @ {row['home_team_name']}",
            "prediccion": pick,
            "prob_home_win": prob_home,
            "prob_away_win": prob_away,
            "confianza": confianza,
            "home_pitcher": row.get("home_pitcher_name", "TBD"),
            "away_pitcher": row.get("away_pitcher_name", "TBD"),
            "features_destacadas": {
                "era_home": round(row.get("era_pitcher_home", 0), 2),
                "era_away": round(row.get("era_pitcher_away", 0), 2),
                "ops_home": round(row.get("ops_team_home", 0), 3),
                "ops_away": round(row.get("ops_team_away", 0), 3),
                "win_pct_home": round(row.get("win_pct_home", 0), 3),
                "win_pct_away": round(row.get("win_pct_away", 0), 3),
                "era_diff": round(row.get("era_diff", 0), 2),
                "ops_diff": round(row.get("ops_diff", 0), 3),
            }
        })

    return resultados


def generar_reporte(resultados: List[Dict], fecha: str) -> str:
    """Genera reporte Markdown con validación de coherencia."""
    total = len(resultados)
    altas = sum(1 for r in resultados if r["confianza"] == "ALTA")
    medias = sum(1 for r in resultados if r["confianza"] == "MEDIA")
    bajas = total - altas - medias

    # Validación: alerta si hay picks inconsistentes
    inconsistencias = []
    for r in resultados:
        if r["prediccion"] == "HOME" and r["prob_home_win"] < r["prob_away_win"]:
            inconsistencias.append(r["matchup"])
        elif r["prediccion"] == "AWAY" and r["prob_away_win"] < r["prob_home_win"]:
            inconsistencias.append(r["matchup"])

    alerta = ""
    if inconsistencias:
        alerta = (
            "\n> ⚠️ **ALERTA DE CONSISTENCIA**: Los siguientes juegos tienen "
            f"probabilidades cruzadas: {', '.join(inconsistencias)}\n"
        )

    lineas = [
        f"# ⚾ Predicciones MLB — {fecha}",
        "",
        f"> Generado: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        alerta,
        "",
        "---",
        "",
        "## 📊 Resumen",
        f"- **Juegos**: {total}",
        f"- **Alta confianza**: {altas}",
        f"- **Media confianza**: {medias}",
        f"- **Baja confianza**: {bajas}",
        "",
        "---",
        "",
        "## 🎯 Predicciones",
        "",
        "| Matchup | Pick | Prob. Local | Prob. Visitante | Confianza | Lanzador Local | Lanzador Visitante |",
        "|---------|------|-------------|-----------------|-----------|----------------|--------------------|",
    ]

    for r in resultados:
        lineas.append(
            f"| {r['matchup']} | **{r['prediccion']}** | {r['prob_home_win']:.1%} | "
            f"{r['prob_away_win']:.1%} | {r['confianza']} | {r['home_pitcher']} | {r['away_pitcher']} |"
        )

    lineas.extend([
        "",
        "---",
        "",
        "## 🧠 Metodología",
        "",
        "Modelo **XGBoost** regularizado (max_depth=4, reg_alpha=0.1) con calibración logística.",
        "Probabilidades clippeadas al rango [20%, 80%] para reflejar la varianza inherente del béisbol.",
        "",
        "**Features principales:**",
        "- ERA, WHIP del lanzador abridor",
        "- OPS del equipo",
        "- Win % y diferencial de carreras",
        "- Forma reciente (últimos 10 juegos)",
        "- **Diferenciales**: `era_diff`, `ops_diff`, `win_pct_diff`",
        "",
        "Fuente: [MLB Stats API](https://statsapi.mlb.com)",
        "",
        "---",
        "",
        "*Actualizado automáticamente vía GitHub Actions.*"
    ])

    return "\n".join(lineas)


def ejecutar_pipeline(fecha: str = None):
    """Pipeline completo: ingesta → features → predicción → reporte."""
    if fecha is None:
        fecha = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"🚀 Pipeline para {fecha}")

    # 1. Ingesta (solo_futuros=True para obtener probablePitcher)
    juegos = obtener_calendario_diario(fecha, solo_futuros=True)
    if not juegos:
        logger.info("Sin juegos hoy.")
        return

    # 2. Features
    df = construir_dataset(juegos)

    # 3. Modelo + calibrador
    modelo, calibrador = cargar_o_entrenar_modelo()

    # 4. Predicción
    resultados = predecir_juegos(df, modelo, calibrador)

    # 5. Guardar JSON
    ruta_json = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.json")
    guardar_json({"fecha": fecha, "predicciones": resultados}, ruta_json)

    # 6. Reporte Markdown
    reporte_md = generar_reporte(resultados, fecha)
    ruta_md = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    # README diario
    ruta_readme = os.path.join(DATA_PREDICTIONS, "README.md")
    with open(ruta_readme, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    logger.info("✅ Pipeline completado")
