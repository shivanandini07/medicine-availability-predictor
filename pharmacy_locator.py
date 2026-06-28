"""Geocoding and nearby pharmacy discovery via OpenStreetMap APIs."""

from __future__ import annotations

import functools
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from geopy.distance import geodesic


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

from model import canonicalize_medicine_name

USER_AGENT = "medicine_availability_predictor/1.0"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GOOGLE_GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_API_KEY_ENV = "GOOGLE_MAPS_API_KEY"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
COUNTRY_SUFFIX = "India"

SUPPORTED_CITIES: dict[str, dict[str, Any]] = {
    "chennai": {
        "label": "Chennai",
        "queries": ["Chennai, Tamil Nadu, India"],
        "latitude": 13.0827,
        "longitude": 80.2707,
    },
    "bangalore": {
        "label": "Bangalore",
        "queries": ["Bengaluru, Karnataka, India", "Bangalore, Karnataka, India"],
        "latitude": 12.9716,
        "longitude": 77.5946,
    },
    "bengaluru": {
        "label": "Bengaluru",
        "queries": ["Bengaluru, Karnataka, India", "Bangalore, Karnataka, India"],
        "latitude": 12.9716,
        "longitude": 77.5946,
    },
    "hyderabad": {
        "label": "Hyderabad",
        "queries": ["Hyderabad, Telangana, India"],
        "latitude": 17.3850,
        "longitude": 78.4867,
    },
    "mumbai": {
        "label": "Mumbai",
        "queries": ["Mumbai, Maharashtra, India"],
        "latitude": 19.0760,
        "longitude": 72.8777,
    },
    "delhi": {
        "label": "Delhi",
        "queries": ["New Delhi, Delhi, India", "Delhi, India"],
        "latitude": 28.6139,
        "longitude": 77.2090,
    },
    "new delhi": {
        "label": "New Delhi",
        "queries": ["New Delhi, Delhi, India", "Delhi, India"],
        "latitude": 28.6139,
        "longitude": 77.2090,
    },
    "pune": {
        "label": "Pune",
        "queries": ["Pune, Maharashtra, India"],
        "latitude": 18.5204,
        "longitude": 73.8567,
    },
}

CITY_ALIASES: dict[str, str] = {
    "madras": "chennai",
    "bombay": "mumbai",
    "bangalore": "bengaluru",
}


@dataclass
class GeocodeResult:
    """Structured geocoding response for UI and tests."""

    success: bool
    latitude: float | None = None
    longitude: float | None = None
    query_used: str = ""
    display_name: str | None = None
    match_type: str = "none"
    used_fallback: bool = False
    message: str = ""
    error: str | None = None
    attempted_queries: tuple[str, ...] = ()


def _clean_input(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip())


def normalize_city_key(city: str) -> str:
    """Normalize city input for alias and supported-city lookup."""
    cleaned = _clean_input(city).lower()
    return CITY_ALIASES.get(cleaned, cleaned)


def get_supported_city(city: str) -> dict[str, Any] | None:
    """Return supported city metadata when the city is recognized."""
    return SUPPORTED_CITIES.get(normalize_city_key(city))


def build_geocode_queries(city: str, address: str | None = None) -> list[tuple[str, str]]:
    """Build ordered geocoding queries as (query, match_type) pairs."""
    cleaned_city = _clean_input(city)
    cleaned_address = _clean_input(address)

    if not cleaned_city:
        return []

    supported = get_supported_city(cleaned_city)
    city_queries: list[str] = []

    if supported:
        city_queries.extend(supported["queries"])
    else:
        city_queries.append(f"{cleaned_city}, {COUNTRY_SUFFIX}")

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []

    def add_query(query: str, match_type: str) -> None:
        normalized = query.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append((query, match_type))

    if cleaned_address:
        if cleaned_city and cleaned_city.lower() in cleaned_address.lower():
            add_query(cleaned_address, "address")
        else:
            for city_query in city_queries:
                add_query(f"{cleaned_address}, {city_query}", "address")

    for city_query in city_queries:
        add_query(city_query, "city")

    return ordered


def _get_google_api_key() -> str | None:
    return os.environ.get(GOOGLE_API_KEY_ENV)


@functools.lru_cache(maxsize=256)
def _google_geocode_search(query: str) -> list[dict[str, Any]]:
    """Query Google Maps Geocoding API and return raw JSON results."""
    api_key = _get_google_api_key()
    if not api_key:
        return []

    params = {
        "address": query,
        "key": api_key,
        "components": "country:IN",
        "language": "en",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    response = requests.get(GOOGLE_GEOCODING_URL, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    payload = response.json()
    status = payload.get("status")
    if status == "OK":
        return payload.get("results", [])
    if status == "ZERO_RESULTS":
        return []

    raise requests.RequestException(
        f"Google Geocoding API error: {status}"
        + (f" - {payload.get('error_message')}" if payload.get("error_message") else "")
    )


def _result_from_google(
    query: str,
    match_type: str,
    payload: dict[str, Any],
    used_fallback: bool,
    message: str,
) -> GeocodeResult:
    location = payload.get("geometry", {}).get("location", {})
    latitude = location.get("lat")
    longitude = location.get("lng")
    if latitude is None or longitude is None:
        raise ValueError("Invalid Google geocoding payload")

    return GeocodeResult(
        success=True,
        latitude=float(latitude),
        longitude=float(longitude),
        query_used=query,
        display_name=payload.get("formatted_address"),
        match_type=match_type,
        used_fallback=used_fallback,
        message=message,
        attempted_queries=(query,),
    )


def _nominatim_search(query: str, limit: int = 1) -> list[dict[str, Any]]:
    """Query Nominatim and return raw JSON results."""
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
        "countrycodes": "in",
        "addressdetails": 1,
        "accept-language": "en",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=12)
    response.raise_for_status()
    results = response.json()
    return results if isinstance(results, list) else []


def _result_from_nominatim(
    query: str,
    match_type: str,
    payload: dict[str, Any],
    used_fallback: bool,
    message: str,
) -> GeocodeResult:
    return GeocodeResult(
        success=True,
        latitude=float(payload["lat"]),
        longitude=float(payload["lon"]),
        query_used=query,
        display_name=payload.get("display_name"),
        match_type=match_type,
        used_fallback=used_fallback,
        message=message,
        attempted_queries=(query,),
    )


def _known_city_fallback(city: str, attempted_queries: list[str]) -> GeocodeResult:
    supported = get_supported_city(city)
    if not supported:
        return GeocodeResult(
            success=False,
            query_used=attempted_queries[-1] if attempted_queries else _clean_input(city),
            match_type="none",
            used_fallback=False,
            message="We could not find that location.",
            error=(
                "Unable to locate the entered city or address. "
                "Please enter a supported city such as Chennai, Bangalore, Hyderabad, "
                "Mumbai, Delhi, or Pune, or try a simpler address."
            ),
            attempted_queries=tuple(attempted_queries),
        )

    label = supported["label"]
    query_used = attempted_queries[-1] if attempted_queries else supported["queries"][0]
    return GeocodeResult(
        success=True,
        latitude=float(supported["latitude"]),
        longitude=float(supported["longitude"]),
        query_used=query_used,
        display_name=f"{label}, {COUNTRY_SUFFIX}",
        match_type="known_city",
        used_fallback=True,
        message=(
            f"Exact address could not be verified. Using the center of {label} "
            "as a fallback location."
        ),
        attempted_queries=tuple(attempted_queries),
    )


@functools.lru_cache(maxsize=256)
def resolve_location(city: str, address: str | None = None) -> GeocodeResult:
    """
    Resolve a location using address+city, then city-only, then known-city fallback.

    Supports city-only searches when address is blank.
    """
    cleaned_city = _clean_input(city)
    cleaned_address = _clean_input(address)
    attempted_queries: list[str] = []

    if not cleaned_city:
        return GeocodeResult(
            success=False,
            query_used="",
            match_type="none",
            message="City is required.",
            error="Please enter a city name to search for nearby pharmacies.",
            attempted_queries=tuple(),
        )

    queries = build_geocode_queries(cleaned_city, cleaned_address or None)
    if not queries:
        return GeocodeResult(
            success=False,
            query_used="",
            match_type="none",
            message="No valid search query could be built.",
            error="Please check the city and address fields and try again.",
            attempted_queries=tuple(),
        )

    address_failed = False
    last_error: str | None = None

    for query, match_type in queries:
        attempted_queries.append(query)
        sources = [("nominatim", _nominatim_search)]
        if _get_google_api_key():
            sources.insert(0, ("google", _google_geocode_search))

        for source, search_fn in sources:
            try:
                results = search_fn(query)
                if results:
                    used_fallback = match_type == "city" and bool(cleaned_address)
                    if match_type == "address" and cleaned_address:
                        message = f"Location matched using address query: {query}"
                    elif match_type == "city" and cleaned_address and address_failed:
                        message = (
                            f"The address '{cleaned_address}' could not be found. "
                            f"Using city-level location from query: {query}"
                        )
                        used_fallback = True
                    elif match_type == "city":
                        message = f"City-level location matched using query: {query}"
                    else:
                        message = f"Location matched using query: {query}"

                    if source == "google":
                        return _result_from_google(
                            query=query,
                            match_type=match_type,
                            payload=results[0],
                            used_fallback=used_fallback,
                            message=message,
                        )
                    return _result_from_nominatim(
                        query=query,
                        match_type=match_type,
                        payload=results[0],
                        used_fallback=used_fallback,
                        message=message,
                    )

                if source == "nominatim" and match_type == "address":
                    address_failed = True
            except requests.RequestException as exc:
                last_error = str(exc)
                if source == "google":
                    continue
                if match_type == "address":
                    address_failed = True
                time.sleep(0.4)
                continue

            if source == "nominatim":
                time.sleep(0.5)

    fallback = _known_city_fallback(cleaned_city, attempted_queries)
    if fallback.success:
        if cleaned_address:
            fallback.message = (
                f"The address '{cleaned_address}' appears invalid or too specific. "
                f"{fallback.message}"
            )
        elif last_error:
            fallback.message = (
                f"Online geocoding is temporarily unavailable. {fallback.message}"
            )
        return fallback

    fallback.error = (
        last_error
        or "Unable to locate the entered city or address. "
        "Try a supported city (Chennai, Bangalore, Hyderabad, Mumbai, Delhi, Pune) "
        "or leave the address blank for a city-wide search."
    )
    return fallback


def geocode_location(city: str, address: str = "") -> tuple[float, float] | None:
    """Backward-compatible helper returning coordinates or None."""
    result = resolve_location(city, address or None)
    if result.success and result.latitude is not None and result.longitude is not None:
        return result.latitude, result.longitude
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def _parse_overpass_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pharmacies: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("brand") or "Unnamed Pharmacy"
        lat = element.get("lat")
        lon = element.get("lon")

        if lat is None or lon is None:
            center = element.get("center")
            if center:
                lat = center.get("lat")
                lon = center.get("lon")

        if lat is None or lon is None:
            continue

        pharmacies.append(
            {
                "pharmacy_name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "address": tags.get("addr:full")
                or ", ".join(
                    filter(
                        None,
                        [
                            tags.get("addr:street"),
                            tags.get("addr:city"),
                            tags.get("addr:postcode"),
                        ],
                    )
                ),
            }
        )
    return pharmacies


def find_nearby_pharmacies_osm(
    latitude: float,
    longitude: float,
    radius_km: float = 5.0,
) -> list[dict[str, Any]]:
    """Find nearby pharmacies using the Overpass API."""
    radius_m = int(radius_km * 1000)
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
      way["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
      relation["amenity"="pharmacy"](around:{radius_m},{latitude},{longitude});
    );
    out center;
    """

    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        pharmacies = _parse_overpass_elements(data.get("elements", []))
    except Exception:
        pharmacies = []

    for pharmacy in pharmacies:
        pharmacy["distance_km"] = round(
            _haversine_km(latitude, longitude, pharmacy["latitude"], pharmacy["longitude"]),
            2,
        )

    pharmacies.sort(key=lambda p: p["distance_km"])
    return pharmacies


def find_nearby_pharmacies_from_dataset(
    latitude: float,
    longitude: float,
    inventory_df: pd.DataFrame,
    radius_km: float = 10.0,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Fallback: derive nearby pharmacies from historical inventory coordinates."""
    unique = inventory_df.drop_duplicates(subset=["pharmacy_name", "latitude", "longitude"])
    results: list[dict[str, Any]] = []

    for _, row in unique.iterrows():
        distance = _haversine_km(latitude, longitude, row["latitude"], row["longitude"])
        if distance <= radius_km:
            results.append(
                {
                    "pharmacy_name": row["pharmacy_name"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "distance_km": round(distance, 2),
                    "address": "From inventory dataset",
                }
            )

    results.sort(key=lambda p: p["distance_km"])
    return results[:limit]


def discover_nearby_pharmacies(
    latitude: float,
    longitude: float,
    inventory_df: pd.DataFrame,
    radius_km: float = 5.0,
) -> list[dict[str, Any]]:
    """Discover pharmacies via OSM with dataset fallback."""
    pharmacies = find_nearby_pharmacies_osm(latitude, longitude, radius_km=radius_km)
    if pharmacies:
        return pharmacies

    time.sleep(0.5)
    return find_nearby_pharmacies_from_dataset(
        latitude, longitude, inventory_df, radius_km=max(radius_km, 10.0)
    )


def enrich_pharmacies_with_inventory(
    pharmacies: list[dict[str, Any]],
    inventory_df: pd.DataFrame,
    medicine_name: str,
) -> pd.DataFrame:
    """Attach latest inventory records for each pharmacy and medicine."""
    canonical_medicine = canonicalize_medicine_name(medicine_name)
    med_df = inventory_df[inventory_df["medicine_name"].str.lower() == canonical_medicine.lower()]
    if med_df.empty:
        med_df = inventory_df.copy()

    latest = (
        med_df.sort_values("last_reported", ascending=False)
        .groupby("pharmacy_name", as_index=False)
        .first()
    )

    rows: list[dict[str, Any]] = []
    for pharmacy in pharmacies:
        name = pharmacy["pharmacy_name"]
        market_keywords = [part for part in re.split(r"\W+", name) if part]
        match = latest[latest["pharmacy_name"].str.contains("|".join(map(re.escape, market_keywords)), case=False, na=False)]
        if match.empty:
            fallback = latest[latest["pharmacy_name"].str.lower().str.contains(name.split()[0].lower(), na=False)]
            match = fallback if not fallback.empty else latest

        if not match.empty:
            record = match.iloc[0].to_dict()
        else:
            record = {
                "pharmacy_name": name,
                "medicine_name": canonical_medicine,
                "quantity": 0,
                "latitude": pharmacy["latitude"],
                "longitude": pharmacy["longitude"],
                "last_reported": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "availability_status": "Not Available",
            }

        record.update(pharmacy)
        record["pharmacy_name"] = name
        record["medicine_name"] = canonical_medicine
        rows.append(record)

    return pd.DataFrame(rows)
