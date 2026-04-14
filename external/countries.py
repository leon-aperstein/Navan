import logging
from typing import Any

import aiohttp

from config import RESTCOUNTRIES_BASE_URL
from models import CountryData, CountryParams

logger = logging.getLogger(__name__)


def format_country_for_prompt(data: CountryData) -> str:
    """Format CountryData into a human-readable string for LLM prompt."""
    if data.error_message:
        return "COUNTRY_UNAVAILABLE: Country information could not be retrieved for this location."

    parts = []

    if data.capital:
        parts.append(f"Capital: {data.capital}")

    if data.region:
        parts.append(f"Region: {data.region}")

    if data.population:
        parts.append(f"Population: {data.population:,}")

    if data.languages:
        parts.append(f"Languages: {data.languages}")

    if data.currencies:
        parts.append(f"Currencies: {data.currencies}")

    if data.timezones:
        parts.append(f"Timezones: {data.timezones}")

    return "\n".join(parts) if parts else "Country data unavailable"


async def get_country_info(params: CountryParams) -> CountryData:
    """
    Fetch country information from RestCountries API.

    Returns CountryData with error_message set on any failure.
    """
    url = f"{RESTCOUNTRIES_BASE_URL}/name/{params.name}"
    query_params = {
        "fields": "name,capital,currencies,languages,population,region,timezones"
    }

    timeout = aiohttp.ClientTimeout(total=5)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=query_params) as response:
                response.raise_for_status()
                data = await response.json()

                if not data or not isinstance(data, list):
                    logger.error(f"Unexpected API response format for country: {params.name}")
                    return CountryData(error_message="COUNTRY_UNAVAILABLE")

                country = data[0]

                return CountryData(
                    capital=format_capital(country.get("capital")),
                    currencies=format_currencies(country.get("currencies")),
                    languages=format_languages(country.get("languages")),
                    population=country.get("population"),
                    region=country.get("region"),
                    timezones=format_timezones(country.get("timezones"))
                )

    except aiohttp.ClientResponseError as e:
        logger.error(f"Country API HTTP error for '{params.name}': {e.status}")
        return CountryData(error_message="COUNTRY_UNAVAILABLE")
    except aiohttp.ClientError as e:
        logger.error(f"Country API request failed for '{params.name}': {e}")
        return CountryData(error_message="COUNTRY_UNAVAILABLE")
    except Exception as e:
        logger.error(f"Unexpected error fetching country '{params.name}': {e}")
        return CountryData(error_message="COUNTRY_UNAVAILABLE")


def format_capital(capital_data: Any) -> str | None:
    """Extract capital city name from API response."""
    if isinstance(capital_data, list) and capital_data:
        return capital_data[0]
    return None


def format_currencies(currencies_data: dict[str, Any] | None) -> str | None:
    """Format currencies dict into readable string."""
    if not currencies_data:
        return None

    currency_list = []
    for code, info in currencies_data.items():
        name = info.get("name", code)
        symbol = info.get("symbol", "")
        if symbol:
            currency_list.append(f"{name} ({symbol})")
        else:
            currency_list.append(name)

    return ", ".join(currency_list) if currency_list else None


def format_languages(languages_data: dict[str, str] | None) -> str | None:
    """Format languages dict into readable string."""
    if not languages_data:
        return None

    return ", ".join(languages_data.values())


def format_timezones(timezones_data: list[str] | None) -> str | None:
    """Format timezones list into readable string."""
    if not timezones_data:
        return None

    return ", ".join(timezones_data)
