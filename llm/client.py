from __future__ import annotations
import logging
from typing import Any, AsyncIterator
from openai import AsyncOpenAI
import config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )
        self.model = config.DEEPSEEK_MODEL

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None
    ) -> str | None:
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)

            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content

            logger.warning("No choices returned in response")
            return None

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None

    async def chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Streaming version of chat_completion. Yields content chunks as they arrive."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            yield "I'm having trouble processing your request right now. Could you try rephrasing your question?"
