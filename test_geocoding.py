"""Tests for improved geocoding and location search."""

from __future__ import annotations

import sys
from pathlib import Path
from requests import RequestException
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pharmacy_locator import (  # noqa: E402
    build_geocode_queries,
    geocode_location,
    get_supported_city,
    normalize_city_key,
    resolve_location,
)

MAJOR_CITIES = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Pune"]


def test_normalize_city_aliases() -> None:
    assert normalize_city_key("Madras") == "chennai"
    assert normalize_city_key("Bombay") == "mumbai"
    assert normalize_city_key("New Delhi") == "new delhi"


@pytest.mark.parametrize("city", MAJOR_CITIES)
def test_supported_cities_have_coordinates(city: str) -> None:
    supported = get_supported_city(city)
    assert supported is not None
    assert "latitude" in supported
    assert "longitude" in supported
    assert supported["queries"]


def test_build_geocode_queries_city_only() -> None:
    queries = build_geocode_queries("Pune", "")
    assert queries
    assert all(match_type == "city" for _, match_type in queries)
    assert any("Pune" in query for query, _ in queries)


def test_build_geocode_queries_address_first() -> None:
    queries = build_geocode_queries("Mumbai", "Andheri West")
    assert queries[0][1] == "address"
    assert "Andheri West" in queries[0][0]
    assert any(match_type == "city" for _, match_type in queries)


def test_empty_city_returns_friendly_error() -> None:
    result = resolve_location("", "Some Address")
    assert not result.success
    assert result.error is not None
    assert "city" in result.error.lower()


@patch("pharmacy_locator._nominatim_search")
def test_invalid_address_falls_back_to_city(mock_search) -> None:
    mock_search.side_effect = [
        [],
        [{"lat": "19.0760", "lon": "72.8777", "display_name": "Mumbai, Maharashtra, India"}],
    ]

    result = resolve_location("Mumbai", "ThisIsNotARealStreet12345")

    assert result.success
    assert result.used_fallback
    assert result.match_type == "city"
    assert "Mumbai" in result.query_used
    assert mock_search.call_count == 2


@patch("pharmacy_locator._nominatim_search")
def test_city_only_search(mock_search) -> None:
    mock_search.return_value = [
        {"lat": "13.0827", "lon": "80.2707", "display_name": "Chennai, Tamil Nadu, India"}
    ]

    result = resolve_location("Chennai", None)

    assert result.success
    assert result.match_type == "city"
    assert not result.used_fallback
    assert "Chennai" in result.query_used


@patch.dict("os.environ", {"GOOGLE_MAPS_API_KEY": "test-key"})
@patch("pharmacy_locator._google_geocode_search")
@patch("pharmacy_locator._nominatim_search")
def test_google_fallback_to_nominatim(mock_nominatim, mock_google) -> None:
    mock_google.side_effect = RequestException("Google API failure")
    mock_nominatim.return_value = [
        {
            "lat": "13.0827",
            "lon": "80.2707",
            "display_name": "Chennai, Tamil Nadu, India",
        }
    ]

    result = resolve_location("Chennai", "Flat 302, Sai Residency, T Nagar")

    assert result.success
    assert "Chennai" in result.query_used
    assert mock_google.call_count == 1
    assert mock_nominatim.call_count == 1


@patch("pharmacy_locator._nominatim_search")
def test_known_city_fallback_when_api_fails(mock_search) -> None:
    mock_search.return_value = []

    result = resolve_location("Hyderabad", "Invalid Address XYZ")

    assert result.success
    assert result.match_type == "known_city"
    assert result.used_fallback
    assert result.latitude == pytest.approx(17.3850, abs=0.01)
    assert result.longitude == pytest.approx(78.4867, abs=0.01)
    assert result.query_used


@patch("pharmacy_locator._nominatim_search")
def test_address_match_uses_address_query(mock_search) -> None:
    mock_search.return_value = [
        {
            "lat": "12.9352",
            "lon": "77.6245",
            "display_name": "Koramangala, Bengaluru, Karnataka, India",
        }
    ]

    result = resolve_location("Bangalore", "Koramangala")

    assert result.success
    assert result.match_type == "address"
    assert "Koramangala" in result.query_used
    assert not result.used_fallback


@patch("pharmacy_locator._nominatim_search")
def test_unsupported_city_without_api_results_fails(mock_search) -> None:
    mock_search.return_value = []

    result = resolve_location("Unknownville", "Main Street")

    assert not result.success
    assert result.error is not None
    assert result.attempted_queries


@patch("pharmacy_locator._nominatim_search")
@pytest.mark.parametrize("city", MAJOR_CITIES)
def test_major_cities_resolve_with_known_fallback(mock_search, city: str) -> None:
    mock_search.return_value = []
    result = resolve_location(city, "")
    assert result.success
    assert result.latitude is not None
    assert result.longitude is not None
    assert result.query_used


def test_geocode_location_backward_compatible() -> None:
    with patch("pharmacy_locator.resolve_location") as mock_resolve:
        from pharmacy_locator import GeocodeResult

        mock_resolve.return_value = GeocodeResult(
            success=True,
            latitude=28.6139,
            longitude=77.2090,
            query_used="New Delhi, Delhi, India",
            match_type="city",
        )
        coords = geocode_location("Delhi", "")
        assert coords == (28.6139, 77.2090)


@pytest.mark.parametrize("city", MAJOR_CITIES)
def test_live_geocode_major_cities(city: str) -> None:
    """Optional live test against Nominatim for supported Indian cities."""
    result = resolve_location(city, "")
    assert result.success, result.error
    assert result.query_used
    assert -90 <= result.latitude <= 90
    assert -180 <= result.longitude <= 180
