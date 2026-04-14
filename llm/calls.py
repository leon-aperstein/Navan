from __future__ import annotations
import json
import re
import logging
from typing import AsyncIterator
from llm.client import LLMClient
from llm.prompts.extract import build_extract_messages
from llm.prompts.answer import build_answer_messages
from llm.prompts.resolve_refs import build_resolve_refs_messages
from llm.prompts.update_trip import build_update_trip_messages
from llm.prompts.update_user import build_update_user_messages
from models import ExtractionResult, WeatherParams, CountryParams

logger = logging.getLogger(__name__)

def _debug_step(step_name: str, messages: list[dict], response: str | None):
    """No-op: debug logging disabled."""
    pass


async def call_extract(
    client: LLMClient,
    resolved_messages: list[dict],
    latest_user_message: str
) -> ExtractionResult:
    messages = build_extract_messages(resolved_messages, latest_user_message)
    response = await client.chat_completion(
        messages, temperature=0.1, response_format={"type": "json_object"}
    )
    _debug_step("EXTRACT", messages, response)

    if response is None:
        logger.warning("LLM returned None for extraction call")
        return ExtractionResult()

    def parse_extraction(text: str) -> ExtractionResult | None:
        try:
            data = json.loads(text)
            weather = None
            country = None

            if data.get("weather"):
                weather_dict = data["weather"]
                if isinstance(weather_dict, dict) and "city" in weather_dict:
                    weather = WeatherParams(
                        city=weather_dict["city"],
                        country=weather_dict.get("country")
                    )

            if data.get("country"):
                country_dict = data["country"]
                if isinstance(country_dict, dict) and "name" in country_dict:
                    country = CountryParams(name=country_dict["name"])

            return ExtractionResult(weather=weather, country=country)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Parse error: {e}")
            return None

    result = parse_extraction(response)
    if result is not None:
        return result

    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        result = parse_extraction(json_match.group(0))
        if result is not None:
            return result

    logger.info("First extraction parse failed, retrying with stricter prompt")
    retry_messages = messages + [
        {"role": "assistant", "content": response},
        {"role": "user", "content": "Your response was not valid JSON. Please output ONLY the JSON object, nothing else."}
    ]
    retry_response = await client.chat_completion(retry_messages, temperature=0.1)

    if retry_response is not None:
        result = parse_extraction(retry_response)
        if result is not None:
            return result

    logger.warning("Extraction parsing failed after retry, returning empty result")
    return ExtractionResult()


_ANSWER_FALLBACK = "I'm having trouble processing your request right now. Could you try rephrasing your question?"

_ANSWER_RETRY_PROMPT = (
    "Your response was missing the [RESPONSE]: marker. "
    "Please rewrite your answer starting with [RESPONSE]: followed by the user-facing response only."
)


def _extract_after_marker(text: str) -> str | None:
    """Extract content after [RESPONSE]: marker. Returns None if marker not found."""
    for marker in ("[RESPONSE]:", "[RESPONSE]"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return None


async def call_answer(
    client: LLMClient,
    resolved_messages: list[dict],
    user_data_summary: str,
    trip_summary: str,
    weather_data: str | None,
    country_data: str | None,
    latest_user_message: str
) -> str:
    messages = build_answer_messages(
        resolved_messages,
        user_data_summary,
        trip_summary,
        weather_data,
        country_data,
        latest_user_message
    )
    response = await client.chat_completion(messages, temperature=0.7)
    _debug_step("ANSWER", messages, response)

    if response is None:
        logger.warning("LLM returned None for answer call")
        return _ANSWER_FALLBACK

    # Parse response after [RESPONSE] marker
    result = _extract_after_marker(response)
    if result:
        return result

    # Marker missing — retry once with corrective prompt
    logger.info("Answer missing [RESPONSE]: marker, retrying")
    retry_messages = messages + [
        {"role": "assistant", "content": response},
        {"role": "user", "content": _ANSWER_RETRY_PROMPT}
    ]
    retry_response = await client.chat_completion(retry_messages, temperature=0.7)
    _debug_step("ANSWER_RETRY", retry_messages, retry_response)

    if retry_response is not None:
        result = _extract_after_marker(retry_response)
        if result:
            return result

    logger.warning("Answer retry also missing marker, returning fallback")
    return _ANSWER_FALLBACK


async def call_answer_stream(
    client: LLMClient,
    resolved_messages: list[dict],
    user_data_summary: str,
    trip_summary: str,
    weather_data: str | None,
    country_data: str | None,
    latest_user_message: str
) -> AsyncIterator[str]:
    """
    Streaming version of call_answer.
    Buffers chunks until the [RESPONSE]: marker is found, then yields
    everything after the marker token-by-token.
    """
    messages = build_answer_messages(
        resolved_messages,
        user_data_summary,
        trip_summary,
        weather_data,
        country_data,
        latest_user_message
    )

    buffer = ""
    marker_found = False

    async for chunk in client.chat_completion_stream(messages, temperature=0.7):
        if marker_found:
            yield chunk
        else:
            buffer += chunk
            # Check for marker (with or without colon)
            for marker in ("[RESPONSE]:", "[RESPONSE]"):
                if marker in buffer:
                    # Yield everything after the marker
                    after = buffer.split(marker, 1)[1]
                    if after:
                        yield after
                    marker_found = True
                    buffer = ""
                    break

    # If marker was never found, retry with corrective prompt (streamed)
    if not marker_found and buffer.strip():
        logger.info("Stream answer missing [RESPONSE]: marker, retrying")
        retry_messages = messages + [
            {"role": "assistant", "content": buffer},
            {"role": "user", "content": _ANSWER_RETRY_PROMPT}
        ]

        retry_buffer = ""
        retry_marker_found = False

        async for chunk in client.chat_completion_stream(retry_messages, temperature=0.7):
            if retry_marker_found:
                yield chunk
            else:
                retry_buffer += chunk
                for marker in ("[RESPONSE]:", "[RESPONSE]"):
                    if marker in retry_buffer:
                        after = retry_buffer.split(marker, 1)[1]
                        if after:
                            yield after
                        retry_marker_found = True
                        retry_buffer = ""
                        break

        # Retry succeeded without marker — yield raw retry content
        if not retry_marker_found and retry_buffer.strip():
            logger.warning("Stream answer retry also missing marker, returning fallback")
            yield _ANSWER_FALLBACK


async def call_resolve_refs(
    client: LLMClient,
    recent_messages: list[dict],
    full_history: list[dict]
) -> list[dict]:
    messages = build_resolve_refs_messages(recent_messages, full_history)
    response = await client.chat_completion(messages, temperature=0.1)
    _debug_step("RESOLVE_REFS", messages, response)

    if response is None:
        logger.warning("LLM returned None for resolve_refs call")
        return recent_messages

    try:
        resolved = json.loads(response)
        if isinstance(resolved, list):
            # Normalize role casing — model sometimes returns "USER" instead of "user"
            for msg in resolved:
                if isinstance(msg, dict) and "role" in msg:
                    msg["role"] = msg["role"].lower()
            return resolved
        logger.warning(f"resolve_refs response is not a list: {type(resolved)}")
        return recent_messages
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse resolve_refs response: {e}")
        return recent_messages


async def call_update_trip(
    client: LLMClient,
    current_summary: str,
    latest_exchange: list[dict]
) -> str:
    messages = build_update_trip_messages(current_summary, latest_exchange)
    response = await client.chat_completion(messages, temperature=0.3)
    _debug_step("UPDATE_TRIP", messages, response)

    if response is None:
        logger.warning("LLM returned None for update_trip call")
        return current_summary

    return response.strip()


async def call_update_user(
    client: LLMClient,
    current_summary: str,
    latest_exchange: list[dict]
) -> str:
    messages = build_update_user_messages(current_summary, latest_exchange)
    response = await client.chat_completion(messages, temperature=0.3)
    _debug_step("UPDATE_USER", messages, response)

    if response is None:
        logger.warning("LLM returned None for update_user call")
        return current_summary

    return response.strip()
