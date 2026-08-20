"""
========================================
MLB Predictor — Utilidades
========================================
Funciones auxiliares para logging, persistencia y manejo de errores.
"""

import json
import os
import logging
from datetime import datetime

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("mlb_predictor")


def guardar_json(datos: dict, ruta: str):
    """Guarda un diccionario como archivo JSON con formato legible."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON guardado: {ruta}")


def cargar_json(ruta: str) -> dict:
    """Carga un archivo JSON. Retorna dict vacío si no existe."""
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_csv(df, ruta: str):
    """Guarda un DataFrame como CSV."""
    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"CSV guardado: {ruta}")


def timestamp_archivo() -> str:
    """Genera un timestamp para nombrar archivos."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
