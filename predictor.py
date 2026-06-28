"""Availability prediction, stock-out risk, and pharmacy ranking."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd


class PredictorModel(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


from model import canonicalize_medicine_name, engineer_features, get_feature_columns, load_dataset, load_or_train_model


def _compute_inventory_trend(inventory_df: pd.DataFrame, pharmacy_name: str, medicine_name: str) -> float:
    """Return normalized inventory trend (-1 declining, +1 rising)."""
    canonical_medicine = canonicalize_medicine_name(medicine_name)
    subset = inventory_df[
        (inventory_df["pharmacy_name"] == pharmacy_name)
        & (inventory_df["medicine_name"].str.lower() == canonical_medicine.lower())
    ].copy()

    if len(subset) < 2:
        return 0.0

    subset["last_reported"] = pd.to_datetime(subset["last_reported"])
    subset = subset.sort_values("last_reported")
    quantities = subset["quantity"].values.astype(float)

    if len(quantities) < 2:
        return 0.0

    x = np.arange(len(quantities))
    slope = np.polyfit(x, quantities, 1)[0]
    return float(np.clip(slope / 10.0, -1.0, 1.0))


def build_prediction_features(
    pharmacy_records: pd.DataFrame,
    inventory_df: pd.DataFrame,
    medicine_name: str,
) -> pd.DataFrame:
    """Build feature matrix for each pharmacy record."""
    full_df = inventory_df.copy()
    full_df["last_reported"] = pd.to_datetime(full_df["last_reported"])
    reference_date = full_df["last_reported"].max()

    pharmacy_freq = full_df.groupby("pharmacy_name").size().to_dict()
    medicine_demand = full_df.groupby("medicine_name").size().to_dict()
    pharmacy_avail = (
        full_df.assign(is_avail=(full_df["availability_status"] == "Available").astype(int))
        .groupby("pharmacy_name")["is_avail"]
        .mean()
        .to_dict()
    )

    pharmacy_reliability = {}
    for pharm in full_df["pharmacy_name"].unique():
        pharm_data = full_df[full_df["pharmacy_name"] == pharm]
        dates = pd.to_datetime(pharm_data["last_reported"]).sort_values()
        if len(dates) > 1:
            days_span = (dates.max() - dates.min()).days + 1
            reports = len(dates)
            consistency = min(reports / max(days_span / 7, 1.0), 1.0)
            pharmacy_reliability[pharm] = float(consistency)
        else:
            pharmacy_reliability[pharm] = 0.3

    medicine_pharmacy_affinity = {}
    for pharm in full_df["pharmacy_name"].unique():
        pharm_medicines = full_df[full_df["pharmacy_name"] == pharm]["medicine_name"].nunique()
        pharm_records = len(full_df[full_df["pharmacy_name"] == pharm])
        medicine_pharmacy_affinity[pharm] = (
            min(pharm_records / (pharm_medicines * 10), 1.0) if pharm_medicines > 0 else 0.5
        )

    selected_medicine = canonicalize_medicine_name(medicine_name)
    rows: list[dict[str, Any]] = []
    for _, record in pharmacy_records.iterrows():
        pharmacy = record["pharmacy_name"]
        last_reported = pd.to_datetime(record.get("last_reported", reference_date))
        recency = max((reference_date - last_reported).days, 0)
        medicine_key = canonicalize_medicine_name(str(record.get("medicine_name", selected_medicine)))
        quantity = float(record.get("quantity", 0))
        recency_weight = float(np.exp(-recency / 30.0))

        rows.append(
            {
                "pharmacy_name": pharmacy,
                "historical_availability": pharmacy_avail.get(pharmacy, 0.5),
                "inventory_quantity": quantity,
                "pharmacy_frequency": float(pharmacy_freq.get(pharmacy, 1)),
                "report_recency_days": float(recency),
                "recency_weight": recency_weight,
                "medicine_demand_frequency": float(
                    medicine_demand.get(selected_medicine, medicine_demand.get(medicine_key, 1))
                ),
                "pharmacy_reliability": pharmacy_reliability.get(pharmacy, 0.3),
                "medicine_pharmacy_affinity": medicine_pharmacy_affinity.get(pharmacy, 0.5),
                "distance_km": float(record.get("distance_km", 5.0)),
                "latitude": float(record.get("latitude", 0)),
                "longitude": float(record.get("longitude", 0)),
                "inventory_trend": _compute_inventory_trend(inventory_df, pharmacy, selected_medicine),
            }
        )

    return pd.DataFrame(rows)


def predict_stock_out_risk(availability_prob: float, quantity: float, recency_days: float, pharmacy_reliability: float = 0.5) -> tuple[str, float]:
    """Estimate stock-out risk level and probability."""
    qty_factor = 1.0 - min(quantity / 100.0, 1.0)
    recency_factor = min(recency_days / 30.0, 1.0)
    reliability_factor = 1.0 - pharmacy_reliability
    risk_prob = float(
        np.clip(
            (1 - availability_prob) * 0.40
            + qty_factor * 0.35
            + recency_factor * 0.15
            + reliability_factor * 0.10,
            0,
            1,
        )
    )

    if risk_prob < 0.35:
        level = "Low"
    elif risk_prob < 0.65:
        level = "Medium"
    else:
        level = "High"

    return level, risk_prob


def rank_pharmacies(feature_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Rank pharmacies using availability, distance, history, trend, and reliability."""
    ranked = feature_df.copy()
    ranked["availability_probability"] = probabilities

    max_dist = ranked["distance_km"].max() if ranked["distance_km"].max() > 0 else 1.0
    ranked["distance_score"] = 1.0 - (ranked["distance_km"] / max_dist)
    ranked["history_score"] = ranked["historical_availability"]
    ranked["trend_score"] = (ranked["inventory_trend"] + 1) / 2
    ranked["reliability_score"] = ranked.get("pharmacy_reliability", 0.5)
    ranked["quantity_score"] = np.minimum(ranked["inventory_quantity"] / 100.0, 1.0)
    ranked["recency_score"] = ranked.get("recency_weight", 1.0)
    ranked["affinity_score"] = ranked.get("medicine_pharmacy_affinity", 0.5)

    ranked["composite_score"] = (
        ranked["availability_probability"] * 0.35
        + ranked["distance_score"] * 0.20
        + ranked["history_score"] * 0.15
        + ranked["trend_score"] * 0.10
        + ranked["reliability_score"] * 0.10
        + ranked["quantity_score"] * 0.05
        + ranked["recency_score"] * 0.05
    )

    risk_levels: list[str] = []
    risk_probs: list[float] = []
    for _, row in ranked.iterrows():
        pharmacy_rel = row.get("pharmacy_reliability", 0.5)
        level, prob = predict_stock_out_risk(
            row["availability_probability"],
            row["inventory_quantity"],
            row["report_recency_days"],
            pharmacy_reliability=pharmacy_rel,
        )
        risk_levels.append(level)
        risk_probs.append(prob)

    ranked["stock_out_risk"] = risk_levels
    ranked["stock_out_probability"] = risk_probs
    ranked["predicted_status"] = np.where(
        ranked["availability_probability"] >= 0.5, "Available", "Not Available"
    )
    ranked = ranked.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def predict_for_pharmacies(
    pharmacy_records: pd.DataFrame,
    medicine_name: str,
    model: PredictorModel | None = None,
    inventory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run full prediction and ranking pipeline for nearby pharmacies."""
    if inventory_df is None:
        inventory_df = load_dataset()
    if model is None:
        model = load_or_train_model()

    features_df = build_prediction_features(pharmacy_records, inventory_df, medicine_name)
    feature_cols = get_feature_columns()
    x_features = features_df[feature_cols].values.astype(float)
    probabilities = model.predict_proba(x_features)[:, 1]
    return rank_pharmacies(features_df, probabilities)


def get_analytics_summary(ranked: pd.DataFrame) -> dict[str, Any]:
    """Compute dashboard analytics from ranked pharmacy results."""
    if ranked.empty:
        return {
            "total_pharmacies": 0,
            "average_availability": 0.0,
            "high_risk_count": 0,
            "top_pharmacy": "N/A",
            "top_probability": 0.0,
            "top_distance": 0.0,
        }

    top = ranked.iloc[0]
    return {
        "total_pharmacies": len(ranked),
        "average_availability": float(ranked["availability_probability"].mean()),
        "high_risk_count": int((ranked["stock_out_risk"] == "High").sum()),
        "top_pharmacy": str(top["pharmacy_name"]),
        "top_probability": float(top["availability_probability"]),
        "top_distance": float(top["distance_km"]),
    }


def get_availability_distribution(inventory_df: pd.DataFrame, medicine_name: str) -> pd.DataFrame:
    """Return availability status counts for a medicine."""
    canonical_medicine = canonicalize_medicine_name(medicine_name)
    subset = inventory_df[inventory_df["medicine_name"].str.lower() == canonical_medicine.lower()]
    if subset.empty:
        subset = inventory_df
    return subset["availability_status"].value_counts().reset_index()


def get_inventory_trend_series(
    inventory_df: pd.DataFrame,
    pharmacy_name: str,
    medicine_name: str,
) -> pd.DataFrame:
    """Return time series of inventory quantity for charts."""
    subset = inventory_df[
        (inventory_df["pharmacy_name"] == pharmacy_name)
        & (inventory_df["medicine_name"].str.lower() == medicine_name.lower())
    ].copy()

    if subset.empty:
        subset = inventory_df[inventory_df["medicine_name"].str.lower() == medicine_name.lower()].head(20)

    subset["last_reported"] = pd.to_datetime(subset["last_reported"])
    trend = (
        subset.groupby("last_reported", as_index=False)["quantity"]
        .mean()
        .sort_values("last_reported")
    )
    return trend
