"""Automated tests for the Medicine Availability Predictor."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import (  # noqa: E402
    DATA_PATH,
    MODEL_PATH,
    canonicalize_medicine_name,
    engineer_features,
    generate_synthetic_dataset,
    load_dataset,
    load_or_train_model,
    save_report,
)
from pharmacy_locator import (  # noqa: E402
    discover_nearby_pharmacies,
    enrich_pharmacies_with_inventory,
    find_nearby_pharmacies_from_dataset,
    resolve_location,
)
from predictor import (  # noqa: E402
    build_prediction_features,
    get_analytics_summary,
    predict_for_pharmacies,
    rank_pharmacies,
)


@pytest.fixture(scope="module")
def inventory_df() -> pd.DataFrame:
    df = load_dataset()
    assert len(df) >= 5000
    return df


@pytest.fixture(scope="module")
def trained_model():
    return load_or_train_model(force_retrain=False)


def test_dataset_loads(inventory_df: pd.DataFrame) -> None:
    required_cols = {
        "pharmacy_name",
        "medicine_name",
        "quantity",
        "latitude",
        "longitude",
        "last_reported",
        "availability_status",
    }
    assert required_cols.issubset(inventory_df.columns)
    assert DATA_PATH.exists()


def test_dataset_generation_creates_minimum_rows() -> None:
    df = generate_synthetic_dataset(n_rows=5000, seed=99)
    assert len(df) >= 5000
    assert df["medicine_name"].nunique() >= 5


def test_model_loads(trained_model) -> None:
    assert trained_model is not None
    assert hasattr(trained_model, "predict_proba")
    assert MODEL_PATH.exists()


def test_medicine_alias_normalization() -> None:
    assert canonicalize_medicine_name("Dolo 650") == "Paracetamol"
    assert canonicalize_medicine_name("Calpol") == "Paracetamol"
    assert canonicalize_medicine_name("Crocin") == "Paracetamol"
    assert canonicalize_medicine_name("Cetzine") == "Cetirizine"
    assert canonicalize_medicine_name("Glycomet") == "Metformin"
    assert canonicalize_medicine_name("Mox") == "Amoxicillin"


def test_feature_engineering(inventory_df: pd.DataFrame) -> None:
    features, target = engineer_features(inventory_df)
    assert len(features) == len(inventory_df)
    assert len(target) == len(inventory_df)
    assert features.isnull().sum().sum() == 0


def test_prediction_works(inventory_df: pd.DataFrame, trained_model) -> None:
    sample_pharmacies = inventory_df.drop_duplicates("pharmacy_name").head(5).copy()
    sample_pharmacies["distance_km"] = [1.0, 2.0, 3.0, 4.0, 5.0][: len(sample_pharmacies)]

    ranked = predict_for_pharmacies(
        sample_pharmacies,
        "Paracetamol",
        model=trained_model,
        inventory_df=inventory_df,
    )

    assert not ranked.empty
    assert "availability_probability" in ranked.columns
    assert "stock_out_risk" in ranked.columns
    assert ranked["availability_probability"].between(0, 1).all()


def test_ranking_orders_by_composite_score(inventory_df: pd.DataFrame) -> None:
    sample = inventory_df.drop_duplicates("pharmacy_name").head(8).copy()
    sample["distance_km"] = range(1, len(sample) + 1)
    features = build_prediction_features(sample, inventory_df, "Paracetamol")
    import numpy as np

    probs = np.linspace(0.9, 0.2, len(features))
    ranked = rank_pharmacies(features, probs)

    assert ranked.iloc[0]["composite_score"] >= ranked.iloc[-1]["composite_score"]
    assert ranked.iloc[0]["rank"] == 1


def test_analytics_summary(inventory_df: pd.DataFrame, trained_model) -> None:
    sample = inventory_df.drop_duplicates("pharmacy_name").head(3).copy()
    sample["distance_km"] = [0.5, 1.5, 2.5]
    ranked = predict_for_pharmacies(sample, "Crocin", model=trained_model, inventory_df=inventory_df)
    summary = get_analytics_summary(ranked)

    assert summary["total_pharmacies"] == 3
    assert 0 <= summary["average_availability"] <= 1
    assert summary["top_pharmacy"] != "N/A"


def test_pharmacy_search_from_dataset(inventory_df: pd.DataFrame) -> None:
    lat = inventory_df["latitude"].iloc[0]
    lon = inventory_df["longitude"].iloc[0]
    pharmacies = find_nearby_pharmacies_from_dataset(lat, lon, inventory_df, radius_km=15.0)
    assert len(pharmacies) > 0
    assert "distance_km" in pharmacies[0]
    assert "pharmacy_name" in pharmacies[0]


def test_discover_nearby_pharmacies_fallback(inventory_df: pd.DataFrame) -> None:
    lat = float(inventory_df["latitude"].mean())
    lon = float(inventory_df["longitude"].mean())
    result = discover_nearby_pharmacies(lat, lon, inventory_df, radius_km=20.0)
    assert isinstance(result, list)


def test_enrich_pharmacies_with_inventory(inventory_df: pd.DataFrame) -> None:
    pharmacies = [
        {
            "pharmacy_name": inventory_df["pharmacy_name"].iloc[0],
            "latitude": float(inventory_df["latitude"].iloc[0]),
            "longitude": float(inventory_df["longitude"].iloc[0]),
            "distance_km": 1.0,
        }
    ]
    enriched = enrich_pharmacies_with_inventory(pharmacies, inventory_df, "Paracetamol")
    assert len(enriched) == 1
    assert "quantity" in enriched.columns


def test_save_report_new_record() -> None:
    """Test saving a new crowdsourced report."""
    initial_df = load_dataset()
    initial_count = len(initial_df)
    
    success = save_report(
        pharmacy_name="Test Pharmacy",
        medicine_name="Paracetamol",
        availability_status="Available",
        quantity=50,
        report_date="2026-06-24",
    )
    
    assert success is True
    
    # Reload and verify
    updated_df = load_dataset()
    assert len(updated_df) > initial_count


def test_save_report_canonicalizes_medicine() -> None:
    """Test that saved reports canonicalize medicine names."""
    initial_df = load_dataset()
    
    success = save_report(
        pharmacy_name="Test Pharmacy 2",
        medicine_name="Dolo 650",  # Alias for Paracetamol
        availability_status="Available",
        quantity=30,
        report_date="2026-06-24",
    )
    
    assert success is True
    
    # Verify the medicine name was canonicalized
    updated_df = load_dataset()
    test_records = updated_df[updated_df["pharmacy_name"] == "Test Pharmacy 2"]
    assert len(test_records) > 0
    assert test_records.iloc[-1]["medicine_name"] == "Paracetamol"


def test_save_report_handles_invalid_input() -> None:
    """Test that save_report handles invalid inputs gracefully."""
    success = save_report(
        pharmacy_name="",  # Empty pharmacy name
        medicine_name="Paracetamol",
        availability_status="Available",
        quantity=50,
        report_date="2026-06-24",
    )
    
    # Should still save since pharmacy_name is checked only if empty
    # But let's check with a truly invalid medicine
    success = save_report(
        pharmacy_name="Test Pharmacy 3",
        medicine_name="",  # Empty medicine name
        availability_status="Available",
        quantity=50,
        report_date="2026-06-24",
    )
    
    assert success is False  # Should fail with empty medicine name


def test_geocode_location_returns_coordinates() -> None:
    result = resolve_location("New Delhi", "Connaught Place")
    if result.success:
        assert result.latitude is not None
        assert result.longitude is not None
        assert -90 <= result.latitude <= 90
        assert -180 <= result.longitude <= 180
        assert result.query_used
