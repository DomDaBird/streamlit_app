# Automotive Insights

Lokale Streamlit-App fuer ein Automotive-Data-Science-Projekt mit Markttrends,
Gebrauchtwagenanalyse und ML-Preisprognose.

## Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Die App ist danach lokal unter `http://localhost:8501` erreichbar.

## Struktur

- `streamlit_app.py` startet die App.
- `automotive_app/data.py` laedt und bereinigt die CSV-Daten.
- `automotive_app/ml.py` trainiert die Preisprognose.
- `automotive_app/charts.py` enthaelt interaktive Plotly-Visualisierungen.
- `automotive_app/app.py` rendert die Streamlit-Oberflaeche.

## ML-Ansatz

Das Preisprognosemodell nutzt bereinigte Gebrauchtwagenangebote mit globalem
Train-/Test-Split. Kategorische Merkmale werden robust codiert, der Zielwert
wird logarithmisch transformiert und die App zeigt MAE, Medianfehler, R2 und
eine Median-Baseline zum Vergleich.
