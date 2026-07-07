from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


COLORWAY = ["#2563eb", "#14b8a6", "#f97316", "#7c3aed", "#dc2626", "#64748b", "#16a34a"]


def apply_chart_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        colorway=COLORWAY,
        margin=dict(l=10, r=10, t=48, b=10),
        legend_title_text="",
        hovermode="x unified",
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e5e7eb")
    return fig


def line_by_columns(data: pd.DataFrame, x: str, columns: list[str], title: str, value_name: str) -> go.Figure:
    chart_data = data.melt(id_vars=x, value_vars=columns, var_name="Kategorie", value_name=value_name)
    fig = px.line(chart_data, x=x, y=value_name, color="Kategorie", markers=True, title=title)
    return apply_chart_style(fig)


def registration_chart(data: pd.DataFrame, country: str) -> go.Figure:
    fig = px.line(
        data,
        x="Jahr",
        y=country,
        color="Antrieb",
        markers=True,
        title=f"Neuzulassungen in {country}",
    )
    fig.update_yaxes(title="Fahrzeuge")
    return apply_chart_style(fig)


def europe_registration_chart(data: pd.DataFrame) -> go.Figure:
    fig = px.area(data, x="Jahr", y="Europe", color="Antrieb", title="Europa nach Antriebsart")
    fig.update_yaxes(title="Fahrzeuge")
    return apply_chart_style(fig)


def histogram(data: pd.DataFrame, column: str, title: str, nbins: int = 45) -> go.Figure:
    fig = px.histogram(data, x=column, nbins=nbins, title=title, opacity=0.86)
    return apply_chart_style(fig)


def top_bar(series: pd.Series, title: str, x_title: str = "Anzahl") -> go.Figure:
    values = series.sort_values(ascending=True)
    fig = px.bar(x=values.values, y=values.index, orientation="h", title=title)
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="")
    return apply_chart_style(fig, height=max(360, min(720, len(values) * 34)))


def scatter_price_mileage(data: pd.DataFrame, title: str) -> go.Figure:
    sample = data.sample(min(len(data), 4_000), random_state=42) if len(data) > 4_000 else data
    fig = px.scatter(
        sample,
        x="mileage_in_km",
        y="price_in_euro",
        color="fuel_type",
        hover_data=["year", "power_ps"],
        title=title,
        opacity=0.55,
    )
    fig.update_xaxes(title="Kilometer")
    fig.update_yaxes(title="Preis in Euro")
    return apply_chart_style(fig)


def forecast_charging(data: pd.DataFrame, state: str, end_year: int) -> tuple[pd.DataFrame, float]:
    history = data[["Jahr", state]].rename(columns={state: "Ladesaeulen"}).copy()
    year0 = int(history["Jahr"].min())
    x = (history["Jahr"] - year0).to_numpy().reshape(-1, 1)
    y = np.log1p(history["Ladesaeulen"].to_numpy())
    model = LinearRegression().fit(x, y)

    future_years = np.arange(int(history["Jahr"].max()) + 1, end_year + 1)
    future_x = (future_years - year0).reshape(-1, 1)
    future_values = np.expm1(model.predict(future_x)).clip(min=0)

    historical_predictions = np.expm1(model.predict(x))
    score = float(r2_score(history["Ladesaeulen"], historical_predictions))

    forecast = pd.DataFrame({"Jahr": future_years, "Ladesaeulen": future_values, "Typ": "Trend"})
    history["Typ"] = "Ist"
    return pd.concat([history, forecast], ignore_index=True), score


def charging_forecast_chart(forecast: pd.DataFrame, state: str) -> go.Figure:
    fig = px.line(forecast, x="Jahr", y="Ladesaeulen", color="Typ", markers=True, title=f"Ladesaeulen: {state}")
    fig.update_yaxes(title="Ladesaeulen")
    return apply_chart_style(fig)

