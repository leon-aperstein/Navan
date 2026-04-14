from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WeatherParams:
    city: str
    country: str | None = None


@dataclass
class CountryParams:
    name: str


@dataclass
class ExtractionResult:
    weather: WeatherParams | None = None
    country: CountryParams | None = None


@dataclass
class WeatherData:
    temperature: float | None = None
    conditions: str | None = None
    forecast_summary: str | None = None
    error_message: str | None = None


@dataclass
class CountryData:
    capital: str | None = None
    currencies: str | None = None
    languages: str | None = None
    population: int | None = None
    region: str | None = None
    timezones: str | None = None
    error_message: str | None = None


