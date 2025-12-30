# ===============================================
# API Integration Module
# Fetch solar radiation data from external APIs
# ===============================================

import requests
from typing import Optional, List
from .config import LATITUDE, LONGITUDE, TIMEZONE, API_TIMEOUT


def fetch_openmeteo_solar_forecast(
    year: int, month: int, day: int
) -> Optional[List[float]]:
    """
    Fetch solar radiation forecast from Open-Meteo API.

    Args:
        year: Forecast year
        month: Forecast month
        day: Forecast day

    Returns:
        List of 24 hourly radiation values (W/m²) or None if failed
    """
    date_str = f"{year}-{month:02d}-{day:02d}"
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "shortwave_radiation",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": TIMEZONE,
    }

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        hourly_radiation = data.get("hourly", {}).get("shortwave_radiation", [])
        if len(hourly_radiation) == 24:
            return hourly_radiation
    except Exception as e:
        print(f"Open-Meteo API error: {e}")
    return None


def fetch_nasa_power_solar(year: int, month: int, day: int) -> Optional[List[float]]:
    """
    Fetch solar radiation data from NASA POWER API.
    Returns hourly ALLSKY_SFC_SW_DWN (Surface Shortwave Downward Irradiance) in W/m².

    Note: Historical data only (~1 week delay)

    Args:
        year: Data year
        month: Data month
        day: Data day

    Returns:
        List of 24 hourly radiation values (W/m²) or None if failed
    """
    date_str = f"{year}{month:02d}{day:02d}"

    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": date_str,
        "end": date_str,
        "format": "JSON",
        "time-standard": "LST",
    }

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        hourly_data = (
            data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
        )

        # Extract hourly values
        hourly_radiation = []
        for hour in range(24):
            key = f"{year}{month:02d}{day:02d}{hour:02d}"
            value = hourly_data.get(key, -999)
            hourly_radiation.append(max(0, value) if value != -999 else 0)

        if len(hourly_radiation) == 24 and sum(hourly_radiation) > 0:
            return hourly_radiation
        else:
            return None

    except Exception as e:
        print(f"NASA POWER API error: {e}")
        return None


def fetch_api_forecast(api_name: str, year: int, month: int, day: int) -> tuple:
    """
    Fetch forecast from selected API with fallback.

    Args:
        api_name: "NASA POWER" or "Open-Meteo"
        year, month, day: Forecast date

    Returns:
        Tuple of (hourly_data, actual_api_name, use_openmeteo_weights)
    """
    use_openmeteo_weights = False
    actual_api_name = api_name

    if api_name == "NASA POWER":
        api_preds = fetch_nasa_power_solar(year, month, day)
        if not api_preds:
            # Fallback to Open-Meteo
            api_preds = fetch_openmeteo_solar_forecast(year, month, day)
            actual_api_name = "Open-Meteo (fallback)"
            use_openmeteo_weights = True
    else:
        api_preds = fetch_openmeteo_solar_forecast(year, month, day)
        use_openmeteo_weights = True

    return api_preds, actual_api_name, use_openmeteo_weights
