# ⚾ MLB Predictor

Sistema automatizado de predicción de ganadores de partidos de la MLB utilizando Machine Learning (XGBoost) y la API oficial gratuita de MLB (`statsapi.mlb.com`).

## 🚀 Características

- **100% en la nube**: Funciona sin terminal local, gestionable desde el celular vía GitHub.
- **Automatizado**: Se ejecuta diariamente vía GitHub Actions.
- **API Gratuita**: Usa exclusivamente `statsapi.mlb.com`, sin claves ni costos.
- **Modelo XGBoost**: Entrenado con variables de sabermetría (ERA, WHIP, OPS, Win%, Run Diff).
- **Reportes en Markdown**: Predicciones legibles generadas automáticamente.

## 📁 Estructura
mlb-predictor/
├── .github/workflows/   # Automatización (GitHub Actions)
├── data/                # Datos crudos, procesados y predicciones
├── src/                 # Módulos Python (fetcher, features, model)
├── models/              # Modelos entrenados (.pkl)
├── main.py              # Punto de entrada
└── requirements.txt     # Dependencias

## ▶️ Cómo usar

### Desde GitHub (Web o Móvil)

1. Crea un repositorio nuevo en GitHub.
2. Copia y pega cada archivo de este proyecto en su ruta correspondiente.
3. Ve a la pestaña **Actions** → **MLB Daily Predictions** → **Run workflow**.
4. Los resultados aparecerán automáticamente en `data/predictions/`.


| Variable            | Descripción                                     |
| ------------------- | ----------------------------------------------- |
| `era_pitcher_*`     | Earned Run Average del lanzador abridor         |
| `whip_pitcher_*`    | Walks + Hits per Inning Pitched                 |
| `ops_team_*`        | On-base Plus Slugging (ofensiva)                |
| `win_pct_*`         | Porcentaje de victorias del equipo              |
| `run_diff_*`        | Diferencial de carreras (anotadas - permitidas) |
| `forma_*`           | Victorias en los últimos 10 juegos              |
| `is_home_advantage` | Ventaja de jugar en casa (1/0)                  |
