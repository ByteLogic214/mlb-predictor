# ⚾ Predicciones MLB — 2026-09-21

> Generado: 2026-09-21 20:48 UTC

---

## 📊 Resumen
- **Juegos**: 3
- **Alta confianza**: 0
- **Media confianza**: 0
- **Baja confianza**: 3

---

## 🎯 Predicciones

| Matchup | Pick | Prob. Local | Prob. Visitante | Confianza | Lanzador Local | Lanzador Visitante |
|---------|------|-------------|-----------------|-----------|----------------|--------------------|
| Toronto Blue Jays @ Baltimore Orioles | **AWAY** | 49.0% | 51.0% | BAJA | Shane Baz | Trey Yesavage |
| Washington Nationals @ Detroit Tigers | **HOME** | 53.9% | 46.1% | BAJA | River Ryan | DJ Herz |
| Minnesota Twins @ San Francisco Giants | **HOME** | 50.4% | 49.6% | BAJA | Blade Tidwell | Zebby Matthews |

---

## 🧠 Metodología

Modelo **XGBoost** calibrado con features dinámicas de serie (`game_num_series`, `prev_game_winner_home`) y desbalance de bullpen.
Ajuste bayesiano de métricas de lanzadores abridores.

Fuente: [MLB Stats API](https://statsapi.mlb.com)

---

*Actualizado automáticamente vía GitHub Actions.*