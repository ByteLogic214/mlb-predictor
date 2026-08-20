"""
========================================
MLB Predictor — Modelo y Predicción
========================================
Pipeline con calibración sigmoide cruzada, decay temporal y notificación a Telegram.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from src.config import (
    FEATURES_MODELO, MODELS_DIR, DATA_PREDICTIONS, DATA_PROCESSED,
    TEMPORADA_ACTUAL, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
    COLSAMPLE_BYTREE, REG_ALPHA, REG_LAMBDA, RANDOM_STATE
)
from src.fetcher import (
    obtener_calendario_diario,
    obtener_resultado_juego,
    obtener_pitchers_reales
)
from src.features import construir_dataset
from src.telegram_bot import enviar_predicciones_telegram
from src.utils import logger, guardar_json

RUTA_MODELO_CALIBRADO = os.path.join(MODELS_DIR, "calibrated_xgb_mlb.pkl")
RUTA_HISTORIAL = os.path.join(DATA_PROCESSED, "historial_entrenamiento.csv")


def _calcular_pesos_temporales(df_fechas: pd.Series, decay_rate: float = 0.005) -> np.ndarray:
    """Calcula el decaimiento temporal asignando más peso a partidos recientes."""
    fechas_dt = pd.to_datetime(df_fechas)
    max_fecha = fechas_dt.max()
    dias_diferencia = (max_fecha - fechas_dt).dt.days
    return np.exp(-decay_rate * dias_diferencia)


def _entrenar_con_historial_real(temporada: int = TEMPORADA_ACTUAL) -> CalibratedClassifierCV:
    """Entrena exclusivamente con datos reales de la MLB extraídos de los últimos 60 días."""
    logger.info("Construyendo dataset histórico real desde la API de la MLB...")

    hoy = datetime.now()
    juegos_con_resultado = []

    for i in range(1, 61):
        fecha_pasada = (hoy - timedelta(days=i)).strftime("%Y-%m-%d")
        juegos_dia = obtener_calendario_diario(fecha_pasada, temporada, solo_futuros=False)

        for juego in juegos_dia:
            if juego.get("estado") != "Final":
                continue

            target = obtener_resultado_juego(juego["game_pk"])
            if target is None:
                continue

            pitchers = obtener_pitchers_reales(juego["game_pk"])
            juego.update(pitchers)
            juego["target"] = target
            juegos_con_resultado.append(juego)

    if len(juegos_con_resultado) < 30:
        raise ValueError(
            f"Muestra insuficiente ({len(juegos_con_resultado)} juegos). "
            "No se pueden generar predicciones confiables sin historial real suficiente."
        )

    df = construir_dataset(juegos_con_resultado)
    if df.empty:
        raise ValueError("Error al procesar el dataset histórico desde las features.")

    targets = []
    for _, row in df.iterrows():
        original = next((j for j in juegos_con_resultado if j["game_pk"] == row["game_pk"]), None)
        targets.append(original["target"] if original else None)

    df["target"] = targets
    df = df.dropna(subset=["target"]).sort_values("fecha").reset_index(drop=True)

    X = df[FEATURES_MODELO].astype(float)
    y = df["target"].astype(int)

    base_xgb = xgb.XGBClassifier(
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

    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method="sigmoid",
        cv=5
    )

    sample_weights = _calcular_pesos_temporales(df["fecha"])
    calibrated_model.fit(X, y, sample_weight=sample_weights)

    probs = calibrated_model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, (probs >= 0.5).astype(int))
    loss = log_loss(y, probs)
    brier = brier_score_loss(y, probs)
    logger.info(f"Entrenamiento Real Finalizado — Acc: {acc:.3f} | LogLoss: {loss:.3f} | Brier: {brier:.3f}")

    with open(RUTA_MODELO_CALIBRADO, "wb") as f:
        pickle.dump(calibrated_model, f)
    df.to_csv(RUTA_HISTORIAL, index=False)

    return calibrated_model


def cargar_o_entrenar_modelo() -> CalibratedClassifierCV:
    """Carga el modelo calibrado si existe; de lo contrario, entrena con la API real."""
    if os.path.exists(RUTA_MODELO_CALIBRADO):
        logger.info("Cargando modelo calibrado existente...")
        with open(RUTA_MODELO_CALIBRADO, "rb") as f:
            return pickle.load(f)
    return _entrenar_con_historial_real()


def predecir_juegos(df_features: pd.DataFrame, modelo: CalibratedClassifierCV) -> List[Dict]:
    """Genera las probabilidades reales calibradas."""
    if df_features.empty:
        return []

    X = df_features[FEATURES_MODELO].astype(float)
    probs_home_raw = modelo.predict_proba(X)[:, 1]

    resultados = []
    for i, (_, row) in enumerate(df_features.iterrows()):
        prob_home = round(float(probs_home_raw[i]), 3)
        prob_away = round(float(1.0 - prob_home), 3)
        pick = "HOME" if prob_home >= 0.5 else "AWAY"

        margen = abs(prob_home - 0.5)
        if margen >= 0.12:
            confianza = "ALTA"
        elif margen >= 0.05:
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
                "game_num_series": int(row.get("game_num_series", 1)),
                "prev_game_winner_home": int(row.get("prev_game_winner_home", 0)),
                "bullpen_load_diff": round(row.get("bullpen_load_diff", 0), 2)
            }
        })

    return resultados


def generar_reporte(resultados: List[Dict], fecha: str) -> str:
    """Genera reporte Markdown."""
    total = len(resultados)
    altas = sum(1 for r in resultados if r["confianza"] == "ALTA")
    medias = sum(1 for r in resultados if r["confianza"] == "MEDIA")
    bajas = total - altas - medias

    lineas = [
        f"# ⚾ Predicciones MLB — {fecha}",
        "",
        f"> Generado: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
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
        "Modelo **XGBoost** calibrado con features dinámicas de serie (`game_num_series`, `prev_game_winner_home`) y desbalance de bullpen.",
        "Ajuste bayesiano de métricas de lanzadores abridores.",
        "",
        "Fuente: [MLB Stats API](https://statsapi.mlb.com)",
        "",
        "---",
        "",
        "*Actualizado automáticamente vía GitHub Actions.*"
    ])

    return "\n".join(lineas)


def ejecutar_pipeline(fecha: str = None):
    """Pipeline completo con notificación automática."""
    if fecha is None:
        fecha = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"🚀 Pipeline para {fecha}")

    juegos = obtener_calendario_diario(fecha, solo_futuros=True)
    if not juegos:
        logger.info("Sin juegos hoy.")
        return

    df = construir_dataset(juegos)
    modelo = cargar_o_entrenar_modelo()
    resultados = predecir_juegos(df, modelo)

    # Guardar archivos locales
    ruta_json = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.json")
    guardar_json({"fecha": fecha, "predicciones": resultados}, ruta_json)

    reporte_md = generar_reporte(resultados, fecha)
    ruta_md = os.path.join(DATA_PREDICTIONS, f"predicciones_{fecha}.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    ruta_readme = os.path.join(DATA_PREDICTIONS, "README.md")
    with open(ruta_readme, "w", encoding="utf-8") as f:
        f.write(reporte_md)

    # Disparar alerta a Telegram
    enviar_predicciones_telegram(resultados, fecha)

    logger.info("✅ Pipeline completado exitosamente")
