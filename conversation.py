from __future__ import annotations
import asyncio
import logging
from llm.client import LLMClient
from typing import AsyncIterator
from llm.calls import (
    call_extract,
    call_answer,
    call_answer_stream,
    call_resolve_refs,
    call_update_trip,
    call_update_user
)
from external.weather import get_weather, format_weather_for_prompt
from external.countries import get_country_info, format_country_for_prompt
from config import MAX_HISTORY_MESSAGES

logger = logging.getLogger(__name__)


class ConversationState:
    """Manages all conversation state including messages, summaries, and background tasks."""

    def __init__(self):
        self.client = LLMClient()
        self.messages: list[dict] = []            # raw conversation history
        self.resolved_messages: list[dict] = []   # messages with resolved references
        self.trip_summary: str = ""                # rolling trip plan
        self.user_data_summary: str = ""           # rolling user preferences
        self.background_tasks: list[asyncio.Task] = []  # pending background tasks

    def reset(self):
        """Clear all state for /reset command."""
        self.messages = []
        self.resolved_messages = []
        self.trip_summary = ""
        self.user_data_summary = ""
        # Cancel any pending background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        self.background_tasks = []

    def add_message(self, role: str, content: str):
        """Add a raw message to history."""
        self.messages.append({"role": role, "content": content})

    def get_recent_messages(self, n: int | None = None) -> list[dict]:
        """Get the last n messages (uses MAX_HISTORY_MESSAGES from config if n is None)."""
        if n is None:
            n = MAX_HISTORY_MESSAGES
        return self.messages[-n:] if len(self.messages) > n else self.messages

    def get_resolved_or_raw(self) -> list[dict]:
        """Return resolved_messages if available, otherwise raw messages."""
        return self.resolved_messages if self.resolved_messages else self.messages

    async def _await_background_tasks(self):
        """Await and clear background tasks from previous turn.

        Handles the case where tasks were created on a different event loop.
        """
        if not self.background_tasks:
            return
        try:
            results = await asyncio.gather(*self.background_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Background task {i} failed: {result}")
        except RuntimeError as e:
            logger.warning(f"Discarding stale background tasks (different event loop): {e}")
        self.background_tasks = []

    async def process_turn(self, user_message: str) -> str:
        """
        Process a user message turn and return the assistant's response.

        Steps:
        1. Await background tasks from previous turn
        2. Add user message to history
        3. Extract parameters for external APIs
        4. Fetch external data in parallel (weather, country)
        5. Generate assistant response
        6. Add assistant response to history
        7. Launch background tasks for next turn
        8. Trim context window if needed
        9. Return assistant response
        """
        # Step 1: Await background tasks from previous turn
        await self._await_background_tasks()

        # Step 2: Add user message to raw history
        self.add_message("user", user_message)

        # Step 3: Call extract to determine what external data is needed
        extraction_result = await call_extract(
            self.client,
            self.get_resolved_or_raw(),
            user_message
        )

        # Step 4: Fetch external data in parallel
        weather_str = None
        country_str = None

        tasks = []
        if extraction_result.weather is not None:
            tasks.append(get_weather(extraction_result.weather))
        else:
            tasks.append(None)

        if extraction_result.country is not None:
            tasks.append(get_country_info(extraction_result.country))
        else:
            tasks.append(None)

        # Gather results, filtering out None tasks
        if any(task is not None for task in tasks):
            actual_tasks = [task for task in tasks if task is not None]
            results = await asyncio.gather(*actual_tasks, return_exceptions=True)

            result_index = 0
            if extraction_result.weather is not None:
                weather_data = results[result_index]
                result_index += 1
                if isinstance(weather_data, Exception):
                    logger.error(f"Weather fetch failed: {weather_data}")
                    weather_str = "WEATHER_UNAVAILABLE: Live weather data could not be retrieved for this location."
                else:
                    weather_str = format_weather_for_prompt(weather_data)

            if extraction_result.country is not None:
                country_data = results[result_index]
                if isinstance(country_data, Exception):
                    logger.error(f"Country fetch failed: {country_data}")
                    country_str = "COUNTRY_UNAVAILABLE: Country information could not be retrieved for this location."
                else:
                    country_str = format_country_for_prompt(country_data)

        # Step 5: Call answer to generate response
        assistant_response = await call_answer(
            self.client,
            self.get_resolved_or_raw(),
            self.user_data_summary,
            self.trip_summary,
            weather_str,
            country_str,
            user_message
        )

        # Step 6: Add assistant response to raw history
        self.add_message("assistant", assistant_response)

        # Step 7: Launch background tasks (non-blocking)
        self.launch_background_tasks(user_message, assistant_response)

        # Step 8: Trim context window if needed
        self.trim_history()

        # Step 9: Return the assistant response
        return assistant_response

    async def process_turn_stream(self, user_message: str) -> AsyncIterator[str]:
        """
        Streaming version of process_turn. Yields response chunks as they arrive.

        Same pipeline as process_turn (await background tasks, extract, fetch data),
        but streams the answer instead of waiting for the full response.
        After streaming completes, stores the full response and launches background tasks.
        """
        # Step 1: Await background tasks from previous turn
        await self._await_background_tasks()

        # Step 2: Add user message to raw history
        self.add_message("user", user_message)

        # Step 3: Call extract to determine what external data is needed
        extraction_result = await call_extract(
            self.client,
            self.get_resolved_or_raw(),
            user_message
        )

        # Step 4: Fetch external data in parallel
        weather_str = None
        country_str = None

        tasks = []
        if extraction_result.weather is not None:
            tasks.append(get_weather(extraction_result.weather))
        else:
            tasks.append(None)

        if extraction_result.country is not None:
            tasks.append(get_country_info(extraction_result.country))
        else:
            tasks.append(None)

        if any(task is not None for task in tasks):
            actual_tasks = [task for task in tasks if task is not None]
            results = await asyncio.gather(*actual_tasks, return_exceptions=True)

            result_index = 0
            if extraction_result.weather is not None:
                weather_data = results[result_index]
                result_index += 1
                if isinstance(weather_data, Exception):
                    logger.error(f"Weather fetch failed: {weather_data}")
                    weather_str = "WEATHER_UNAVAILABLE: Live weather data could not be retrieved for this location."
                else:
                    weather_str = format_weather_for_prompt(weather_data)

            if extraction_result.country is not None:
                country_data = results[result_index]
                if isinstance(country_data, Exception):
                    logger.error(f"Country fetch failed: {country_data}")
                    country_str = "COUNTRY_UNAVAILABLE: Country information could not be retrieved for this location."
                else:
                    country_str = format_country_for_prompt(country_data)

        # Step 5: Stream the answer, collecting full response
        full_response_parts = []
        async for chunk in call_answer_stream(
            self.client,
            self.get_resolved_or_raw(),
            self.user_data_summary,
            self.trip_summary,
            weather_str,
            country_str,
            user_message
        ):
            full_response_parts.append(chunk)
            yield chunk

        # Step 6: Store full response in history
        assistant_response = "".join(full_response_parts).strip()
        self.add_message("assistant", assistant_response)

        # Step 7: Launch background tasks (non-blocking)
        self.launch_background_tasks(user_message, assistant_response)

        # Step 8: Trim context window if needed
        self.trim_history()

    def launch_background_tasks(self, user_message: str, assistant_response: str):
        """
        Launch background tasks for reference resolution and summary updates.

        Creates 3 async tasks:
        1. Reference resolution for the latest exchange
        2. Trip summary update
        3. User data summary update

        Each task is wrapped in error handling to prevent crashes.
        """
        # Get the latest exchange (last 2 messages)
        latest_exchange = self.messages[-2:] if len(self.messages) >= 2 else self.messages

        # Task 1: Reference resolution
        async def resolve_refs_task():
            try:
                resolved = await call_resolve_refs(
                    self.client,
                    latest_exchange,
                    self.messages
                )
                # Extend resolved_messages with the resolved version
                self.resolved_messages.extend(resolved)
                # Trim to match raw messages length
                max_len = min(MAX_HISTORY_MESSAGES, len(self.messages))
                if len(self.resolved_messages) > max_len:
                    self.resolved_messages = self.resolved_messages[-max_len:]
            except Exception as e:
                logger.error(f"Reference resolution task failed: {e}")

        # Task 2: Trip summary update
        async def update_trip_task():
            try:
                updated_summary = await call_update_trip(
                    self.client,
                    self.trip_summary,
                    latest_exchange
                )
                self.trip_summary = updated_summary
            except Exception as e:
                logger.error(f"Trip summary update task failed: {e}")

        # Task 3: User data summary update
        async def update_user_task():
            try:
                updated_summary = await call_update_user(
                    self.client,
                    self.user_data_summary,
                    latest_exchange
                )
                self.user_data_summary = updated_summary
            except Exception as e:
                logger.error(f"User data summary update task failed: {e}")

        # Create and store tasks
        self.background_tasks = [
            asyncio.create_task(resolve_refs_task()),
            asyncio.create_task(update_trip_task()),
            asyncio.create_task(update_user_task())
        ]

    def trim_history(self):
        """Trim message history to MAX_HISTORY_MESSAGES if needed."""
        if len(self.messages) > MAX_HISTORY_MESSAGES:
            self.messages = self.messages[-MAX_HISTORY_MESSAGES:]

        # Also trim resolved_messages to match
        if len(self.resolved_messages) > MAX_HISTORY_MESSAGES:
            self.resolved_messages = self.resolved_messages[-MAX_HISTORY_MESSAGES:]
