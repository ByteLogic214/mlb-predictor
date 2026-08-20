"""
========================================
MLB Predictor — Módulo de Telegram
========================================
Formatea y envía las predicciones calculadas al canal de Telegram.
"""

import os
import requests
from typing import List, Dict
from src.utils import logger

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def enviar_predicciones_telegram(resultados: List[Dict], fecha: str) -> bool:
    """Envía el resumen de picks calibrados a Telegram vía HTTP POST."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Credenciales de Telegram no configuradas en Secrets. Omitiendo envío.")
        return False

    if not resultados:
        logger.warning("Sin resultados para enviar a Telegram.")
        return False

    mensaje = f"⚾ <b>PREDICCIONES MLB — {fecha}</b>\n"
    mensaje += f"<i>Modelo XGBoost + Dinámica de Series & Bullpen</i>\n\n"

    for r in resultados:
        emoji_conf = "🔥" if r["confianza"] == "ALTA" else ("⚡" if r["confianza"] == "MEDIA" else "⚠️")
        prob_pick = r["prob_home_win"] if r["prediccion"] == "HOME" else r["prob_away_win"]
        
        mensaje += f"{emoji_conf} <b>{r['matchup']}</b>\n"
        mensaje += f"└ <b>Pick:</b> {r['prediccion']} ({prob_pick:.1%})\n"
        mensaje += f"└ <b>Pitchers:</b> {r['home_pitcher']} vs {r['away_pitcher']}\n\n"

    mensaje += "📊 <i>Probabilidades calibradas por modelo cuantitativo.</i>"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("✅ Mensaje enviado exitosamente a Telegram.")
        return True
    except Exception as e:
        logger.error(f"Fallo al enviar mensaje a Telegram: {e}")
        return False
