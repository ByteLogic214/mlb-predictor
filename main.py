#!/usr/bin/env python3
"""
========================================
MLB Predictor — Punto de Entrada Principal
========================================
Orquesta la ejecución completa del sistema.
Puede recibir la fecha como argumento o usar la fecha actual.
"""

import sys
import os

# Asegurar que src/ esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import ejecutar_pipeline
from src.utils import logger


if __name__ == "__main__":
    fecha = sys.argv[1] if len(sys.argv) > 1 else None
    ejecutar_pipeline(fecha)
