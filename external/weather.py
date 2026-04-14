import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from config import OPENWEATHERMAP_API_KEY, OPENWEATHERMAP_BASE_URL
from models import WeatherData, WeatherParams

logger = logging.getLogger(__name__)


def format_weather_for_prompt(data: WeatherData) -> str:
    """Format WeatherData into a human-readable string for LLM prompt."""
    if data.error_message:
        return "WEATHER_UNAVAILABLE: Live weather data could not be retrieved for this location."

    parts = []

    if data.temperature is not None and data.conditions:
        parts.append(f"Current temperature: {data.temperature}°C")
        parts.append(f"Conditions: {data.conditions}")

    if data.forecast_summary:
        parts.append(f"\n{data.forecast_summary}")

    return "\n".join(parts) if parts else "Weather data unavailable"


async def get_weather(params: WeatherParams) -> WeatherData:
    """
    Fetch current weather and 5-day forecast from OpenWeatherMap API.

    Returns WeatherData with error_message set on any failure.
    """
    if not OPENWEATHERMAP_API_KEY:
        logger.error("OPENWEATHERMAP_API_KEY not configured")
        return WeatherData(error_message="WEATHER_UNAVAILABLE")

    location_query = f"{params.city},{params.country}" if params.country else params.city

    current_url = f"{OPENWEATHERMAP_BASE_URL}/weather"
    forecast_url = f"{OPENWEATHERMAP_BASE_URL}/forecast"

    common_params = {
        "q": location_query,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }

    timeout = aiohttp.ClientTimeout(total=5)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            current_task = fetch_current_weather(session, current_url, common_params)
            forecast_task = fetch_forecast(session, forecast_url, common_params)

            current_data, forecast_data = await asyncio.gather(
                current_task, forecast_task, return_exceptions=True
            )

            if isinstance(current_data, Exception):
                logger.error(f"Current weather fetch failed: {current_data}")
                return WeatherData(error_message="WEATHER_UNAVAILABLE")

            if isinstance(forecast_data, Exception):
                logger.warning(f"Forecast fetch failed: {forecast_data}")
                forecast_summary = None
            else:
                forecast_summary = format_forecast(forecast_data)

            return WeatherData(
                temperature=current_data.get("temp"),
                conditions=current_data.get("conditions"),
                forecast_summary=forecast_summary
            )

    except Exception as e:
        logger.error(f"Weather API request failed: {e}")
        return WeatherData(error_message="WEATHER_UNAVAILABLE")


async def fetch_current_weather(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, str]
) -> dict[str, Any]:
    """Fetch and parse current weather data."""
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        data = await response.json()

        return {
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "conditions": data["weather"][0]["description"].capitalize(),
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "city": data["name"],
            "country": data["sys"].get("country", "")
        }


async def fetch_forecast(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, str]
) -> list[dict[str, Any]]:
    """Fetch and parse 5-day forecast data."""
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        data = await response.json()
        return data["list"]


def format_forecast(forecast_list: list[dict[str, Any]]) -> str:
    """Format forecast data into a human-readable summary."""
    if not forecast_list:
        return ""

    daily_forecasts = {}

    for item in forecast_list:
        dt = datetime.fromisoformat(item["dt_txt"])
        date_key = dt.strftime("%a %b %d")

        if date_key not in daily_forecasts:
            daily_forecasts[date_key] = {
                "temps": [],
                "conditions": []
            }

        daily_forecasts[date_key]["temps"].append(item["main"]["temp"])
        daily_forecasts[date_key]["conditions"].append(
            item["weather"][0]["description"]
        )

    lines = ["5-day forecast:"]
    for date, info in list(daily_forecasts.items())[:5]:
        min_temp = min(info["temps"])
        max_temp = max(info["temps"])
        most_common_condition = max(
            set(info["conditions"]),
            key=info["conditions"].count
        ).capitalize()
        lines.append(f"- {date}: {min_temp:.0f}-{max_temp:.0f}°C, {most_common_condition}")

    return "\n".join(lines)
