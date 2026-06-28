"""Dataset generation, feature engineering, and model training pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from fallback_rf import NumpyModelPipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = DATA_DIR / "medicine_inventory.csv"
MODEL_PATH = MODEL_DIR / "medicine_model.pkl"
MODEL_SCHEMA_VERSION = 2

MEDICINES = [
    "Paracetamol",
    "Cetirizine",
    "Azithromycin",
    "Amoxicillin",
    "Metformin",
    "Insulin",
    "Omeprazole",
    "ORS",
]

MEDICINE_ALIASES: dict[str, str] = {
    "PCM": "Paracetamol",
    "Acetaminophen": "Paracetamol",
    "Dolo": "Paracetamol",
    "Dolo 650": "Paracetamol",
    "Crocin": "Paracetamol",
    "Calpol": "Paracetamol",
    "Crocin Advance": "Paracetamol",
    "Cetirizine HCl": "Cetirizine",
    "Cetzine": "Cetirizine",
    "Azithro": "Azithromycin",
    "Amox": "Amoxicillin",
    "Mox": "Amoxicillin",
    "Metformin XR": "Metformin",
    "Glycomet": "Metformin",
    "Omeprazole DR": "Omeprazole",
    "ORS Salt": "ORS",
}

MEDICINE_OPTIONS = [
    *MEDICINES,
    *[alias for alias in MEDICINE_ALIASES if alias not in MEDICINES],
]
PHARMACY_PREFIXES = [
    "Apollo",
    "MedPlus",
    "Wellness",
    "HealthCare",
    "CityMed",
    "LifePharm",
    "GreenCross",
    "CarePlus",
    "PharmaHub",
    "MediStore",
]


CANONICAL_MEDICINE_MAP: dict[str, str] = {
    **{medicine.lower(): medicine for medicine in MEDICINES},
    **{alias.lower(): canonical for alias, canonical in MEDICINE_ALIASES.items()},
}


def canonicalize_medicine_name(raw_name: str) -> str:
    """Map medicine aliases and brand names to canonical medicine names."""
    normalized = raw_name.strip().lower()
    if not normalized:
        return ""
    if normalized in CANONICAL_MEDICINE_MAP:
        return CANONICAL_MEDICINE_MAP[normalized]
    for alias, canonical in MEDICINE_ALIASES.items():
        if alias.lower() in normalized:
            return canonical
    for medicine in MEDICINES:
        if medicine.lower() in normalized:
            return medicine
    return raw_name.strip()


def ensure_directories() -> None:
    """Create data and model directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic medicine inventory dataset."""
    rng = np.random.default_rng(seed)
    ensure_directories()

    n_pharmacies = 120
    pharmacy_names = [
        f"{rng.choice(PHARMACY_PREFIXES)} Pharmacy {i + 1}" for i in range(n_pharmacies)
    ]

    base_lat, base_lon = 28.6139, 77.2090
    pharmacy_coords = {
        name: (
            base_lat + rng.uniform(-0.15, 0.15),
            base_lon + rng.uniform(-0.15, 0.15),
        )
        for name in pharmacy_names
    }

    medicine_demand = {med: rng.uniform(0.3, 1.0) for med in MEDICINES}
    pharmacy_reliability = {name: rng.uniform(0.4, 0.95) for name in pharmacy_names}

    rows: list[dict] = []
    now = datetime.now()

    for _ in range(n_rows):
        pharmacy = rng.choice(pharmacy_names)
        medicine = rng.choice(MEDICINES)
        lat, lon = pharmacy_coords[pharmacy]

        demand = medicine_demand[medicine]
        reliability = pharmacy_reliability[pharmacy]
        base_prob = 0.25 + 0.55 * reliability * demand

        available = rng.random() < base_prob
        quantity = int(rng.integers(0, 150)) if available else int(rng.integers(0, 8))
        if available and quantity == 0:
            quantity = int(rng.integers(5, 80))

        days_ago = int(rng.integers(0, 45))
        last_reported = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        rows.append(
            {
                "pharmacy_name": pharmacy,
                "medicine_name": medicine,
                "quantity": quantity,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "last_reported": last_reported,
                "availability_status": "Available" if available else "Not Available",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Load dataset from disk or generate if missing."""
    ensure_directories()
    if not DATA_PATH.exists():
        return generate_synthetic_dataset()

    df = pd.read_csv(DATA_PATH)
    if "medicine_name" in df.columns:
        df["medicine_name"] = (
            df["medicine_name"].fillna("")
            .astype(str)
            .apply(canonicalize_medicine_name)
        )
    return df


def save_report(
    pharmacy_name: str,
    medicine_name: str,
    availability_status: str,
    quantity: int,
    report_date: str,
) -> bool:
    """Save a crowdsourced medicine availability report to the dataset.
    
    Args:
        pharmacy_name: Name of the pharmacy
        medicine_name: Name of the medicine (will be canonicalized)
        availability_status: "Available" or "Not Available"
        quantity: Quantity in stock (0+ for available, typically 0 if not available)
        report_date: Date of report in YYYY-MM-DD format
    
    Returns:
        True if save successful, False otherwise
    """
    try:
        ensure_directories()
        
        canonical_medicine = canonicalize_medicine_name(medicine_name)
        if not canonical_medicine:
            return False
        
        # Create report row
        report = {
            "pharmacy_name": pharmacy_name.strip(),
            "medicine_name": canonical_medicine,
            "quantity": max(0, int(quantity)),
            "latitude": 0.0,  # User reports don't have exact coordinates
            "longitude": 0.0,
            "last_reported": report_date,
            "availability_status": availability_status,
        }
        
        # Load existing data
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
        else:
            df = generate_synthetic_dataset()
        
        # Append new report
        report_df = pd.DataFrame([report])
        df = pd.concat([df, report_df], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(DATA_PATH, index=False)
        return True
    except Exception as e:
        print(f"Error saving report: {e}")
        return False


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build ML features and target from raw inventory records."""
    work = df.copy()
    work["last_reported"] = pd.to_datetime(work["last_reported"])
    reference_date = work["last_reported"].max()

    pharmacy_freq = work.groupby("pharmacy_name").size().rename("pharmacy_frequency")
    medicine_demand = work.groupby("medicine_name").size().rename("medicine_demand_frequency")

    pharmacy_avail = (
        work.assign(is_avail=(work["availability_status"] == "Available").astype(int))
        .groupby("pharmacy_name")["is_avail"]
        .mean()
        .rename("historical_availability")
    )

    work = work.merge(pharmacy_freq, on="pharmacy_name", how="left")
    work = work.merge(medicine_demand, on="medicine_name", how="left")
    work = work.merge(pharmacy_avail, on="pharmacy_name", how="left")

    work["report_recency_days"] = (reference_date - work["last_reported"]).dt.days
    work["inventory_quantity"] = work["quantity"]
    work["target"] = (work["availability_status"] == "Available").astype(int)

    trend = (
        work.sort_values("last_reported")
        .groupby(["pharmacy_name", "medicine_name"])["quantity"]
        .apply(lambda series: np.polyfit(np.arange(len(series)), series.astype(float), 1)[0]
               if len(series) > 1 else 0.0)
        .reset_index(name="inventory_trend")
    )
    trend["inventory_trend"] = trend["inventory_trend"].clip(-10.0, 10.0) / 10.0
    work = work.merge(trend, on=["pharmacy_name", "medicine_name"], how="left")
    work["inventory_trend"] = work["inventory_trend"].fillna(0.0)

    feature_cols = [
        "historical_availability",
        "inventory_quantity",
        "pharmacy_frequency",
        "report_recency_days",
        "medicine_demand_frequency",
        "inventory_trend",
    ]

    features = work[feature_cols].fillna(0)
    target = work["target"]
    return features, target


def _split_data(
    features: pd.DataFrame, target: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and target into train/test arrays."""
    x = features.values.astype(float)
    y = target.values.astype(int)
    if SKLEARN_AVAILABLE:
        return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    rng = np.random.default_rng(42)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    split = int(len(y) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def train_model(df: pd.DataFrame | None = None) -> SklearnPipeline | NumpyModelPipeline:
    """Train Random Forest classifier and persist to disk."""
    ensure_directories()
    if df is None:
        df = load_dataset()

    features, target = engineer_features(df)
    x_train, _, y_train, _ = _split_data(features, target)

    if SKLEARN_AVAILABLE:
        pipeline: SklearnPipeline | NumpyModelPipeline = SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
    else:
        pipeline = NumpyModelPipeline(n_estimators=25, max_depth=6, random_state=42)
        pipeline.fit(x_train, y_train)

    _save_model(pipeline)
    return pipeline


def _save_model(pipeline: SklearnPipeline | NumpyModelPipeline) -> None:
    joblib.dump(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_columns": get_feature_columns(),
            "model": pipeline,
        },
        MODEL_PATH,
    )


def _validate_model_input_shape(model: SklearnPipeline | NumpyModelPipeline) -> bool:
    try:
        if hasattr(model, "named_steps"):
            scaler = model.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "n_features_in_"):
                return scaler.n_features_in_ == len(get_feature_columns())
        elif hasattr(model, "scaler") and hasattr(model.scaler, "n_features_in_"):
            return model.scaler.n_features_in_ == len(get_feature_columns())
    except Exception:
        return False
    return True


def _load_saved_model() -> SklearnPipeline | NumpyModelPipeline | None:
    if not MODEL_PATH.exists():
        return None
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict):
        if loaded.get("schema_version") != MODEL_SCHEMA_VERSION:
            return None
        model = loaded.get("model")
        if model is None:
            return None
        if not _validate_model_input_shape(model):
            return None
        return model
    if _validate_model_input_shape(loaded):
        return loaded
    return None


def load_or_train_model(force_retrain: bool = False) -> SklearnPipeline | NumpyModelPipeline:
    """Load saved model or train a new one if missing."""
    ensure_directories()
    if not DATA_PATH.exists():
        generate_synthetic_dataset()

    if force_retrain:
        return train_model()

    model = _load_saved_model()
    if model is None:
        return train_model()

    return model


def get_feature_columns() -> list[str]:
    """Return ordered feature column names used by the model."""
    return [
        "historical_availability",
        "inventory_quantity",
        "pharmacy_frequency",
        "report_recency_days",
        "medicine_demand_frequency",
        "inventory_trend",
    ]

    df = load_dataset()
    print(f"Dataset rows: {len(df)}")
    model = load_or_train_model(force_retrain=True)
    print(f"Model saved to {MODEL_PATH}")
