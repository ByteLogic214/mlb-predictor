# ⚾ Predicciones MLB — 2026-09-07

> Generado: 2026-09-07 20:22 UTC

---

## 📊 Resumen
- **Juegos**: 11
- **Alta confianza**: 3
- **Media confianza**: 3
- **Baja confianza**: 5

---

## 🎯 Predicciones

| Matchup | Pick | Prob. Local | Prob. Visitante | Confianza | Lanzador Local | Lanzador Visitante |
|---------|------|-------------|-----------------|-----------|----------------|--------------------|
| Atlanta Braves @ Philadelphia Phillies | **HOME** | 50.7% | 49.3% | BAJA | Jesús Luzardo | Grant Holmes |
| New York Mets @ Miami Marlins | **HOME** | 57.8% | 42.2% | MEDIA | Eury Pérez | Jonah Tong |
| Los Angeles Angels @ Boston Red Sox | **HOME** | 53.3% | 46.7% | BAJA | Brayan Bello | Grayson Rodriguez |
| Cleveland Guardians @ Baltimore Orioles | **HOME** | 51.1% | 48.9% | BAJA | Trevor Rogers | Joey Cantillo |
| Arizona Diamondbacks @ Kansas City Royals | **AWAY** | 40.8% | 59.2% | MEDIA | Noah Cameron | Derek Law |
| Chicago Cubs @ Milwaukee Brewers | **AWAY** | 49.1% | 50.9% | BAJA | Robert Gasser | Matthew Boyd |
| Minnesota Twins @ Detroit Tigers | **HOME** | 62.4% | 37.6% | ALTA | Troy Melton | Joe Ryan |
| Washington Nationals @ San Diego Padres | **HOME** | 63.0% | 37.0% | ALTA | Nick Pivetta | Jake Irvin |
| St. Louis Cardinals @ San Francisco Giants | **AWAY** | 42.2% | 57.8% | MEDIA | Logan Webb | Michael McGreevy |
| Cincinnati Reds @ Los Angeles Dodgers | **HOME** | 52.6% | 47.4% | BAJA | nan | Chase Burns |
| Toronto Blue Jays @ Athletics | **AWAY** | 37.3% | 62.7% | ALTA | Jacob Lopez | Dylan Cease |

---

## 🧠 Metodología

Modelo **XGBoost** calibrado con features dinámicas de serie (`game_num_series`, `prev_game_winner_home`) y desbalance de bullpen.
Ajuste bayesiano de métricas de lanzadores abridores.

Fuente: [MLB Stats API](https://statsapi.mlb.com)

---

*Actualizado automáticamente vía GitHub Actions.*