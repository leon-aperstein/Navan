# Travel Assistant

A conversational travel planning assistant that showcases advanced prompt engineering and multi-stage LLM orchestration with external data integration.

## Architecture

The system uses a multi-stage LLM pipeline with critical path optimization and background processing:

```
User message arrives
        │
        ▼
┌─────────────────────────────┐
│  LLM Call 1: EXTRACT        │  Input: conversation history (with resolved refs) + latest message
│  Decide if external APIs    │  Output: structured JSON — weather params, country name, or null
│  are needed, extract params │
└─────────────┬───────────────┘
              │
              ▼
     ┌── Need external data? ──┐
     │ YES                     │ NO
     ▼                         │
┌────────────────────┐         │
│ External API calls │         │
│ (parallel):        │         │
│ • Weather API      │         │
│ • Country Info API │         │
└────────┬───────────┘         │
         │                     │
         └──────┬──────────────┘
                ▼
┌─────────────────────────────┐
│  LLM Call 2: ANSWER (CoT)   │  Input: recent messages, user data summary, trip summary,
│  Generate user response     │         external API data (if any)
│  Chain-of-thought reasoning │  Output: natural language response to user
└─────────────┬───────────────┘
              │
              ▼
       Return response to user
              │
              ▼  (background, non-blocking)
┌─────────────────────────────────────────────────────────┐
│  Three parallel LLM calls:                              │
│                                                         │
│  Call 3: REFERENCE RESOLUTION                           │
│  "what's the weather there" → "...there (meant Tokyo)" │
│  Stored as enriched history for next turn               │
│                                                         │
│  Call 4: TRIP SUMMARY UPDATE                            │
│  Update rolling trip plan summary if conversation       │
│  warrants it (e.g., user confirmed an itinerary)        │
│                                                         │
│  Call 5: USER DATA UPDATE                               │
│  Update rolling user profile/preferences summary        │
│  (e.g., user mentioned they're vegetarian, traveling    │
│  with kids, budget is $3000)                            │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Turn Context Maintenance**: Rolling summaries preserve user preferences and trip plans across long conversations
- **Chain-of-Thought Reasoning**: Structured 5-step reasoning framework (UNDERSTAND → CONTEXT → DATA → PLAN → CAVEATS) improves response quality
- **External Data Integration**: Weather forecasts (OpenWeatherMap) and country information (RestCountries) seamlessly blended with LLM knowledge
- **Background Processing**: Reference resolution and summary updates run asynchronously without adding user-facing latency
- **Reference Resolution**: Ambiguous pronouns and deictics ("there", "it", "that place") automatically clarified with parenthetical annotations
- **Streaming Responses**: Token-by-token delivery in the CLI — CoT reasoning is buffered silently, only the user-facing response is streamed
- **Graceful Degradation**: External API failures don't crash the system; the assistant falls back to general knowledge with appropriate caveats

## Setup

### Prerequisites

- Python 3.11 or higher
- API keys:
  - DeepSeek API key (free at [platform.deepseek.com](https://platform.deepseek.com))
  - OpenWeatherMap API key (free at [openweathermap.org](https://openweathermap.org/api))

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Navan - final best"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and add your API keys:
   ```
   DEEPSEEK_API_KEY=your_deepseek_key_here
   OPENWEATHERMAP_API_KEY=your_openweathermap_key_here
   ```

## Usage

Start the CLI:
```bash
python main.py
```

The CLI supports streaming (token-by-token response delivery) and the full multi-stage pipeline.

### Example Interaction

```
You: I'm planning a trip to Tokyo in October with my family
Assistant: Tokyo in October is a wonderful choice! The weather is typically mild...

You: What's the weather like there?
Assistant: [Fetches current Tokyo weather via OpenWeatherMap]
          Currently in Tokyo it's 18°C and partly cloudy...

You: What should we pack?
Assistant: [Uses weather data + family context from user profile]
          For your family trip to Tokyo in October, I'd recommend...
```

### Available Commands

- `quit` - Exit the assistant
- `/reset` - Clear conversation history and start fresh

## Project Structure

```
travel-assistant/
├── main.py                      # Entry point - CLI chat loop with streaming
├── config.py                    # Configuration (LLM endpoint, model, API URLs, keys)
├── conversation.py              # ConversationState class, turn orchestration, streaming
├── models.py                    # Data classes for structured data
├── llm/
│   ├── __init__.py
│   ├── client.py                # LLM client wrapper (OpenAI-compatible for DeepSeek)
│   ├── calls.py                 # Functions that compose prompts + call LLM + parse responses
│   └── prompts/
│       ├── __init__.py
│       ├── extract.py           # Extraction prompt (Call 1)
│       ├── answer.py            # Answer prompt with CoT (Call 2)
│       ├── resolve_refs.py      # Reference resolution prompt (Call 3)
│       ├── update_trip.py       # Trip summary update prompt (Call 4)
│       └── update_user.py       # User data update prompt (Call 5)
├── external/
│   ├── __init__.py
│   ├── weather.py               # OpenWeatherMap API client
│   └── countries.py             # RestCountries API client
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment configuration
├── transcripts/                 # Sample conversation transcripts
│   ├── japan_family_chat.md     # Family trip planning with weather + country data
│   ├── south_america_chat.md    # Multi-turn with follow-ups and references
│   └── travel_europe_then_pivot_israel_chat.md  # Topic pivots and edge cases
├── PROMPT_ENGINEERING.md        # Prompt engineering decision notes
└── README.md                    # This file
```

## Prompt Engineering

This project demonstrates advanced prompt engineering techniques including:

- **Chain-of-thought reasoning** with structured multi-step thinking
- **Data routing** to decide when to use external APIs vs LLM knowledge
- **Reference resolution** to maintain conversation clarity
- **Rolling summaries** for long-term memory without context window overflow
- **Anti-hallucination strategies** to handle uncertainty transparently

For detailed notes on prompt engineering decisions, see [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md).

## Conversation Types Supported

The assistant handles diverse travel queries:

- Destination recommendations
- Packing suggestions
- Local attractions and activities
- Itinerary planning
- Budget advice
- Visa and documentation requirements
- Weather forecasts and seasonal information
- Food and dining recommendations
- Transportation guidance
- Safety tips

## Error Handling

- External API failures degrade gracefully with fallback to general LLM knowledge
- Background task failures are logged but don't impact user-facing responses
- Context window overflow is handled with automatic trimming while preserving key information in summaries
- Malformed LLM outputs are recovered via regex fallback and corrective retry prompts

## Development

Sample conversation transcripts demonstrating various features and edge cases are available in the `transcripts/` directory.

## License

[Add your license here]

## Acknowledgments

- DeepSeek for providing the LLM API
- OpenWeatherMap for weather data
- RestCountries for country information