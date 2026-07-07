from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .config import CURRENT_YEAR, DATA_FILES


COUNTRY_START = "Belgium"
COUNTRY_END = "Kosovo*"
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d.,]*")


def _to_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    def parse_value(value: object) -> float | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text == ":":
            return None
        match = NUMBER_PATTERN.search(text)
        if match is None:
            return None
        number = match.group(0)
        if "," in number:
            number = number.replace(".", "").replace(",", ".")
        elif number.count(".") > 1:
            number = number.replace(".", "")
        elif "." in number:
            left, right = number.rsplit(".", 1)
            if len(right) == 3 and len(left) <= 3:
                number = left + right
        try:
            return float(number)
        except ValueError:
            return None

    return series.map(parse_value).astype(float)


@st.cache_data(show_spinner=False)
def load_registrations() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILES["registrations"], sep=";", encoding="utf-8-sig")
    country_cols = list(data.loc[:, COUNTRY_START:COUNTRY_END].columns)
    data[country_cols] = data[country_cols].apply(_to_number).fillna(0)
    data["Europe"] = data[country_cols].sum(axis=1)
    return data


@st.cache_data(show_spinner=False)
def load_consumption() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILES["consumption"], sep=";")
    for column in data.columns.drop("Jahr"):
        data[column] = _to_number(data[column])
    return data


@st.cache_data(show_spinner=False)
def load_stock() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILES["stock"], sep=";", encoding="utf-8-sig")
    for column in data.columns.drop("Jahr"):
        data[column] = _to_number(data[column]).astype("Int64")
    return data


@st.cache_data(show_spinner=False)
def load_charging() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES["charging"])


@st.cache_data(show_spinner="Gebrauchtwagendaten werden bereinigt ...")
def load_used_cars() -> pd.DataFrame:
    raw = pd.read_csv(DATA_FILES["used_cars"], low_memory=False)
    return clean_used_cars(raw)


def clean_used_cars(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data = data.drop(columns=[col for col in data.columns if col.startswith("Unnamed")], errors="ignore")

    numeric_columns = ["year", "price_in_euro", "power_kw", "power_ps", "mileage_in_km"]
    for column in numeric_columns:
        data[column] = _to_number(data[column])

    for column in ["brand", "model", "color", "transmission_type", "fuel_type"]:
        data[column] = data[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        data[column] = data[column].fillna("Unknown")

    data["brand"] = data["brand"].str.lower()
    data["vehicle_age"] = (CURRENT_YEAR - data["year"]).clip(lower=0)
    data["km_per_year"] = data["mileage_in_km"] / np.maximum(data["vehicle_age"], 1)

    realistic = (
        data["price_in_euro"].between(500, 300_000)
        & data["year"].between(1990, CURRENT_YEAR)
        & data["mileage_in_km"].between(0, 500_000)
        & data["power_ps"].between(20, 1_200)
    )
    data = data.loc[realistic].copy()

    likely_fuels = {
        "Petrol",
        "Diesel",
        "Hybrid",
        "Electric",
        "LPG",
        "CNG",
        "Diesel Hybrid",
        "Other",
        "Unknown",
        "Hydrogen",
        "Ethanol",
    }
    data = data[data["fuel_type"].isin(likely_fuels)].copy()

    data["year"] = data["year"].astype(int)
    return data.reset_index(drop=True)


@dataclass(frozen=True)
class DataQuality:
    raw_rows: int
    clean_rows: int
    clean_share: float
    missing_by_column: pd.DataFrame


@st.cache_data(show_spinner=False)
def used_car_quality() -> DataQuality:
    raw = pd.read_csv(DATA_FILES["used_cars"], low_memory=False)
    clean = clean_used_cars(raw)
    missing = (
        raw.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .rename("Fehlende Werte (%)")
        .reset_index()
        .rename(columns={"index": "Spalte"})
    )
    return DataQuality(
        raw_rows=len(raw),
        clean_rows=len(clean),
        clean_share=len(clean) / max(len(raw), 1),
        missing_by_column=missing,
    )


def available_models(data: pd.DataFrame, brand: str) -> list[str]:
    models = data.loc[data["brand"] == brand, "model"].dropna().sort_values().unique()
    return list(models)


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime
