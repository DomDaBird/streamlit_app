from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILES = {
    "used_cars": BASE_DIR / "gebrauchtwagen.csv",
    "registrations": BASE_DIR / "neuzulassung.csv",
    "consumption": BASE_DIR / "durchschnitt_verbrauch.csv",
    "stock": BASE_DIR / "pkw_bestand_kraftstoffart_neu.csv",
    "charging": BASE_DIR / "ladesaeulen.csv",
}

ASSETS = {
    "banner": BASE_DIR / "banner1.jpg",
    "new_cars": BASE_DIR / "neuwagen.jpg",
    "refuel": BASE_DIR / "refuel.jpg",
    "map": BASE_DIR / "Deutschlandkarte1.jpg",
    "autoscout": BASE_DIR / "autoscout24logo.png",
    "ml": BASE_DIR / "ml_forestcar.jpg",
}

APP_TITLE = "Automotive Insights"
CURRENT_YEAR = 2026

