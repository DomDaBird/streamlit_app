from __future__ import annotations

import pandas as pd
import streamlit as st

from .charts import (
    charging_forecast_chart,
    europe_registration_chart,
    forecast_charging,
    histogram,
    line_by_columns,
    registration_chart,
    scatter_price_mileage,
    top_bar,
)
from .config import APP_TITLE, ASSETS, CURRENT_YEAR
from .data import (
    available_models,
    load_charging,
    load_consumption,
    load_registrations,
    load_stock,
    load_used_cars,
    used_car_quality,
)
from .ml import build_prediction_row, comparable_market, predict_price, train_price_model
from .styles import apply_styles


def format_eur(value: float) -> str:
    return f"{value:,.0f} EUR".replace(",", ".")


def format_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def metric_grid(used_cars: pd.DataFrame, stock: pd.DataFrame, charging: pd.DataFrame) -> None:
    electric_share = (
        stock["Elektro (BEV)"].iloc[-1] / stock["Pkw gesamt"].iloc[-1] * 100
        if "Pkw gesamt" in stock
        else 0
    )
    charging_latest = charging.drop(columns="Jahr").iloc[-1].sum()
    cols = st.columns(4)
    cols[0].metric("Gebrauchtwagen", format_int(len(used_cars)))
    cols[1].metric("Medianpreis", format_eur(used_cars["price_in_euro"].median()))
    cols[2].metric("BEV-Bestand", f"{electric_share:.2f} %")
    cols[3].metric("Ladesaeulen", format_int(charging_latest))


def render_overview(used_cars: pd.DataFrame, stock: pd.DataFrame, charging: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Automotive Insights</h1>
            <p>Markttrends, Fahrzeugbestand und belastbare Gebrauchtwagen-Preisprognosen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_grid(used_cars, stock, charging)

    left, right = st.columns((1.1, 0.9))
    with left:
        cols = [col for col in ["Benzin", "Diesel", "Elektro (BEV)", "Hybrid gesamt"] if col in stock]
        st.plotly_chart(line_by_columns(stock, "Jahr", cols, "PKW-Bestand nach Antrieb", "Bestand"), width="stretch")
    with right:
        top_brands = used_cars["brand"].value_counts().head(12)
        st.plotly_chart(top_bar(top_brands, "Stärkste Marken im Datensatz"), width="stretch")

    st.plotly_chart(
        histogram(used_cars[used_cars["price_in_euro"] <= used_cars["price_in_euro"].quantile(0.98)], "price_in_euro", "Preisverteilung"),
        width="stretch",
    )


def render_market(registrations: pd.DataFrame, stock: pd.DataFrame, consumption: pd.DataFrame, charging: pd.DataFrame) -> None:
    tab_new, tab_stock, tab_consumption, tab_charging = st.tabs(
        ["Neuzulassungen", "Bestand", "Verbrauch", "Ladeinfrastruktur"]
    )

    with tab_new:
        countries = list(registrations.loc[:, "Belgium":"Kosovo*"].columns)
        country = st.selectbox("Land", countries, index=countries.index("Germany"))
        st.plotly_chart(registration_chart(registrations, country), width="stretch")
        st.plotly_chart(europe_registration_chart(registrations), width="stretch")

    with tab_stock:
        available = [col for col in stock.columns if col not in {"Jahr", "Pkw gesamt"}]
        defaults = [col for col in ["Benzin", "Diesel", "Elektro (BEV)", "Hybrid gesamt"] if col in available]
        selected = st.multiselect("Antriebe", available, default=defaults)
        if selected:
            st.plotly_chart(line_by_columns(stock, "Jahr", selected, "Bestand nach Kraftstoffart", "Bestand"), width="stretch")
            latest = stock.set_index("Jahr")[selected].iloc[-1].sort_values()
            st.plotly_chart(top_bar(latest, f"Bestand {stock['Jahr'].iloc[-1]}", "Bestand"), width="stretch")

    with tab_consumption:
        selected = st.multiselect(
            "Verbrauchsreihen",
            [col for col in consumption.columns if col != "Jahr"],
            default=["Benzin", "Diesel"],
        )
        if selected:
            st.plotly_chart(line_by_columns(consumption, "Jahr", selected, "Durchschnittsverbrauch", "Liter / 100 km"), width="stretch")

    with tab_charging:
        states = [col for col in charging.columns if col != "Jahr"]
        state = st.selectbox("Bundesland", states, index=states.index("Bayern") if "Bayern" in states else 0)
        end_year = st.slider("Prognose bis", int(charging["Jahr"].max()) + 1, 2035, 2030)
        forecast, score = forecast_charging(charging, state, end_year)
        st.plotly_chart(charging_forecast_chart(forecast, state), width="stretch")
        st.metric("Trendgüte historisch", f"{score:.2f}")


def render_used_cars(used_cars: pd.DataFrame) -> None:
    brands = sorted(used_cars["brand"].dropna().unique())
    brand = st.selectbox("Marke", brands, index=brands.index("volkswagen") if "volkswagen" in brands else 0)
    models = available_models(used_cars, brand)
    model = st.selectbox("Modell", models[:250] if len(models) > 250 else models)

    subset = used_cars[(used_cars["brand"] == brand) & (used_cars["model"] == model)]
    if subset.empty:
        st.info("Fuer diese Auswahl liegen keine bereinigten Angebote vor.")
        return

    cols = st.columns(4)
    cols[0].metric("Angebote", format_int(len(subset)))
    cols[1].metric("Medianpreis", format_eur(subset["price_in_euro"].median()))
    cols[2].metric("Median-km", format_int(subset["mileage_in_km"].median()))
    cols[3].metric("Median-PS", format_int(subset["power_ps"].median()))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(histogram(subset, "price_in_euro", f"Preise: {model}"), width="stretch")
    with right:
        st.plotly_chart(top_bar(subset["fuel_type"].value_counts().head(10), "Kraftstoffarten"), width="stretch")

    st.plotly_chart(scatter_price_mileage(subset, f"Preis vs. Kilometer: {model}"), width="stretch")


def render_ml(used_cars: pd.DataFrame) -> None:
    bundle = train_price_model(used_cars)

    brands = sorted(used_cars["brand"].dropna().unique())
    brand = st.selectbox("Marke", brands, index=brands.index("volkswagen") if "volkswagen" in brands else 0, key="ml_brand")
    models = available_models(used_cars, brand)
    model = st.selectbox("Modell", models[:300] if len(models) > 300 else models, key="ml_model")
    market = comparable_market(used_cars, brand, model)

    defaults = market.median(numeric_only=True)
    modes = market.mode(dropna=True).iloc[0] if not market.empty else used_cars.mode(dropna=True).iloc[0]
    left, right = st.columns(2)
    with left:
        year = st.slider("Baujahr", 1990, CURRENT_YEAR, int(defaults.get("year", 2019)))
        mileage = st.number_input("Kilometerstand", min_value=0, max_value=500_000, value=int(defaults.get("mileage_in_km", 80_000)), step=5_000)
        power = st.number_input("Leistung (PS)", min_value=20, max_value=1_200, value=int(defaults.get("power_ps", 150)), step=5)
    with right:
        colors = sorted(used_cars["color"].dropna().unique())
        transmissions = sorted(used_cars["transmission_type"].dropna().unique())
        fuels = sorted(used_cars["fuel_type"].dropna().unique())
        color = st.selectbox("Farbe", colors, index=colors.index(modes["color"]) if modes["color"] in colors else 0)
        transmission = st.selectbox(
            "Getriebe",
            transmissions,
            index=transmissions.index(modes["transmission_type"]) if modes["transmission_type"] in transmissions else 0,
        )
        fuel = st.selectbox("Kraftstoff", fuels, index=fuels.index(modes["fuel_type"]) if modes["fuel_type"] in fuels else 0)

    row = build_prediction_row(brand, model, color, transmission, fuel, year, mileage, power, CURRENT_YEAR)
    prediction = predict_price(bundle, row)

    cols = st.columns(4)
    cols[0].metric("Prognose", format_eur(prediction))
    cols[1].metric("MAE", format_eur(bundle.metrics["mae"]))
    cols[2].metric("Medianfehler", format_eur(bundle.metrics["median_ae"]))
    cols[3].metric("R2", f"{bundle.metrics['r2']:.2f}")

    st.caption(
        f"Trainiert auf {format_int(bundle.training_rows)} Angeboten, getestet auf {format_int(bundle.test_rows)} Angeboten. "
        f"Baseline-MAE: {format_eur(bundle.metrics['baseline_mae'])}."
    )

    if not market.empty:
        st.plotly_chart(scatter_price_mileage(market, "Vergleichbare Angebote"), width="stretch")
        comparison = market[["brand", "model", "year", "mileage_in_km", "power_ps", "fuel_type", "price_in_euro"]].sort_values(
            "price_in_euro"
        )
        st.dataframe(comparison.head(50), width="stretch", hide_index=True)


def render_quality(used_cars: pd.DataFrame) -> None:
    quality = used_car_quality()
    cols = st.columns(3)
    cols[0].metric("Rohdaten", format_int(quality.raw_rows))
    cols[1].metric("Nach Bereinigung", format_int(quality.clean_rows))
    cols[2].metric("Nutzbarer Anteil", f"{quality.clean_share:.1%}")

    st.dataframe(quality.missing_by_column, width="stretch", hide_index=True)
    st.dataframe(
        used_cars[["brand", "model", "year", "price_in_euro", "mileage_in_km", "power_ps", "fuel_type"]].head(200),
        width="stretch",
        hide_index=True,
    )


def render_sources() -> None:
    st.markdown(
        """
        **Datenquellen**

        - [Deutschlandatlas](https://www.deutschlandatlas.bund.de/DE/Karten/Wie-wir-uns-bewegen/111/_node.html#_t11lwjbxk)
        - [Eurostat Road Transport Data](https://ec.europa.eu/eurostat/databrowser/view/road_eqr_carpda__custom_13451775/default/table?lang=en)
        - [Bundesnetzagentur - E-Mobilitaet](https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html)
        - [Umweltbundesamt - Durchschnittlicher Kraftstoffverbrauch](https://www.umweltbundesamt.de/bild/durchschnittlicher-kraftstoffverbrauch-von-pkw)
        - [Umweltbundesamt - PKW-Neuzulassungen](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-neuzulassungen-nach)
        - [Umweltbundesamt - PKW-Bestand nach Kraftstoffart](https://www.umweltbundesamt.de/bild/entwicklung-der-pkw-im-bestand-nach-kraftstoffart)
        - [Kaggle - Germany Used Cars Dataset](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/data)
        """
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="wide")
    apply_styles()

    st.sidebar.title(APP_TITLE)
    if ASSETS["banner"].exists():
        st.sidebar.image(str(ASSETS["banner"]), width="stretch")

    page = st.sidebar.radio(
        "Bereich",
        ["Dashboard", "Markttrends", "Gebrauchtwagen", "ML Preischeck", "Datenqualität", "Quellen"],
    )

    used_cars = load_used_cars()
    registrations = load_registrations()
    stock = load_stock()
    consumption = load_consumption()
    charging = load_charging()

    if page == "Dashboard":
        render_overview(used_cars, stock, charging)
    elif page == "Markttrends":
        render_market(registrations, stock, consumption, charging)
    elif page == "Gebrauchtwagen":
        render_used_cars(used_cars)
    elif page == "ML Preischeck":
        render_ml(used_cars)
    elif page == "Datenqualität":
        render_quality(used_cars)
    else:
        render_sources()
