from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


CATEGORICAL_FEATURES = ["brand", "model", "color", "transmission_type", "fuel_type"]
NUMERIC_FEATURES = ["year", "mileage_in_km", "power_ps", "vehicle_age", "km_per_year"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class PriceModelBundle:
    model: Pipeline
    metrics: dict[str, float]
    training_rows: int
    test_rows: int
    features: list[str]


@st.cache_resource(show_spinner="Preisprognose-Modell wird trainiert ...")
def train_price_model(data: pd.DataFrame, sample_size: int = 140_000) -> PriceModelBundle:
    model_data = data.dropna(subset=FEATURES + ["price_in_euro"]).copy()
    if len(model_data) > sample_size:
        model_data = model_data.sample(sample_size, random_state=42)

    x = model_data[FEATURES]
    y = model_data["price_in_euro"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    regressor = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.08,
            max_iter=260,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    baseline = np.full_like(y_test, fill_value=float(np.median(y_train)), dtype=float)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "median_ae": float(median_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
    }

    return PriceModelBundle(
        model=pipeline,
        metrics=metrics,
        training_rows=len(x_train),
        test_rows=len(x_test),
        features=FEATURES,
    )


def build_prediction_row(
    brand: str,
    model: str,
    color: str,
    transmission_type: str,
    fuel_type: str,
    year: int,
    mileage_in_km: float,
    power_ps: float,
    current_year: int,
) -> pd.DataFrame:
    age = max(current_year - int(year), 0)
    km_per_year = float(mileage_in_km) / max(age, 1)
    return pd.DataFrame(
        [
            {
                "year": int(year),
                "mileage_in_km": float(mileage_in_km),
                "power_ps": float(power_ps),
                "vehicle_age": age,
                "km_per_year": km_per_year,
                "brand": brand,
                "model": model,
                "color": color,
                "transmission_type": transmission_type,
                "fuel_type": fuel_type,
            }
        ]
    )


def predict_price(bundle: PriceModelBundle, row: pd.DataFrame) -> float:
    return float(bundle.model.predict(row[bundle.features])[0])


def comparable_market(data: pd.DataFrame, brand: str, model: str) -> pd.DataFrame:
    subset = data[(data["brand"] == brand) & (data["model"] == model)].copy()
    if len(subset) < 20:
        subset = data[data["brand"] == brand].copy()
    return subset

