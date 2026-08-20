# ⚾ Predicciones MLB — 2026-08-20

> Generado: 2026-08-20 05:59 UTC


---

## 📊 Resumen
- **Juegos**: 9
- **Alta confianza**: 9
- **Media confianza**: 0
- **Baja confianza**: 0

---

## 🎯 Predicciones

| Matchup | Pick | Prob. Local | Prob. Visitante | Confianza | Lanzador Local | Lanzador Visitante |
|---------|------|-------------|-----------------|-----------|----------------|--------------------|
| St. Louis Cardinals @ Cincinnati Reds | **AWAY** | 20.0% | 80.0% | ALTA | Brady Singer | Michael McGreevy |
| San Francisco Giants @ Cleveland Guardians | **AWAY** | 24.5% | 75.5% | ALTA | Gavin Williams | Landen Roupp |
| Toronto Blue Jays @ Tampa Bay Rays | **HOME** | 80.0% | 20.0% | ALTA | Ian Seymour | Shane Bieber |
| Athletics @ Kansas City Royals | **HOME** | 80.0% | 20.0% | ALTA | Randy Dobnak | Gage Jump |
| Atlanta Braves @ Chicago White Sox | **AWAY** | 20.0% | 80.0% | ALTA | Anthony Kay | Grant Holmes |
| Seattle Mariners @ Milwaukee Brewers | **HOME** | 80.0% | 20.0% | ALTA | Robert Gasser | George Kirby |
| New York Yankees @ Baltimore Orioles | **AWAY** | 20.0% | 80.0% | ALTA | Kyle Bradish | Gerrit Cole |
| Washington Nationals @ Texas Rangers | **HOME** | 73.8% | 26.2% | ALTA | Jacob deGrom | Andrew Alvarez |
| Los Angeles Angels @ Houston Astros | **HOME** | 80.0% | 20.0% | ALTA | Peter Lambert | Grayson Rodriguez |

---

## 🧠 Metodología

Modelo **XGBoost** regularizado (max_depth=4, reg_alpha=0.1) con calibración logística.
Probabilidades clippeadas al rango [20%, 80%] para reflejar la varianza inherente del béisbol.

**Features principales:**
- ERA, WHIP del lanzador abridor
- OPS del equipo
- Win % y diferencial de carreras
- Forma reciente (últimos 10 juegos)
- **Diferenciales**: `era_diff`, `ops_diff`, `win_pct_diff`

Fuente: [MLB Stats API](https://statsapi.mlb.com)

---

*Actualizado automáticamente vía GitHub Actions.*