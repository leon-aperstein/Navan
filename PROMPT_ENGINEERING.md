# Prompt Engineering Decisions

This document outlines key prompt engineering techniques used in the Travel Assistant project.

## Architecture Overview

The system uses a multi-stage LLM pipeline with 5 specialized calls: (1) Extract parameters for external APIs, (2) Generate user response with chain-of-thought reasoning, (3) Resolve ambiguous references in conversation history, (4) Update trip plan summary, (5) Update user profile summary. Calls 1-2 run on the critical path (user-facing); calls 3-5 run asynchronously in the background while the user types their next message.

## Why No Framework (LangChain, etc.)

This project uses direct DeepSeek API calls rather than a framework like LangChain, by design.

**Simplicity and rubric alignment.** The assignment states: *"The technical implementation should be simple, allowing you to focus primarily on prompt engineering and conversational quality."* The entire LLM client is ~70 lines of code. Introducing chains, agents, or memory classes would add abstraction layers that obscure the prompts — the opposite of what the assignment asks to showcase. Every design choice (the `[RESPONSE]:` marker for CoT separation, rolling summaries vs. full history, background reference resolution) is explicit and readable in the code.

**Hand-crafted data routing vs. model-driven tool calling.** The assignment asks to *"implement a decision method for when to use external data vs. LLM knowledge."* Instead of native tool calling, where the model autonomously decides which tools to invoke, I implement an explicit extraction prompt that acts as a transparent decision function. I can encode rules such as "a mere mention of a country does NOT trigger the country-facts lookup" and adjust them deterministically. While tool calling can be steered via tool descriptions, its final choice is still probabilistic; the extraction prompt keeps the decision logic fully inspectable and testable. This is a prompt engineering decision, not a framework decision — you could use tool calling with or without LangChain.

**Debuggability.** With raw API calls, every token sent and received is visible in one place. No framework internals, callbacks, or hidden retries to trace through when something goes wrong.

**Trade-offs.** This approach foregoes LangChain's built-in retry logic, token counting, and memory abstractions. For this project's scope that is acceptable — the retry and fallback logic is hand-written where needed (extract JSON retry, answer marker retry), and conversation memory is handled through rolling summaries. If the project were to grow significantly, migrating to a framework would be straightforward since the prompts are already modularized into separate files.

## Chain-of-Thought Reasoning (Answer Prompt)

The answer prompt implements a 5-step CoT framework that structures the LLM's thinking before generating responses:

1. **UNDERSTAND**: Clarify what the user is really asking and identify query type
2. **CONTEXT**: Review known user preferences and trip plan from summaries
3. **DATA**: Evaluate available external data (weather, country info) for relevance and gaps
4. **PLAN**: Outline key points to cover and optimal response structure
5. **CAVEATS**: Identify uncertainties, assumptions, or time-sensitive facts requiring qualification

The `[RESPONSE]:` marker acts as a clear separator between internal reasoning and user-facing output. This approach improves answer quality by forcing systematic consideration of context, data availability, and uncertainty. The reasoning remains invisible to users while ensuring responses are grounded, personalized, and appropriately cautious about factual claims.

## Data Routing (Extract Prompt)

The extract prompt solves the critical question: when should we fetch live data versus rely on LLM knowledge? It uses explicit decision criteria embedded in the system prompt: fetch weather for current conditions/forecasts/packing advice, fetch country data for visa/currency/language questions, fetch neither for recommendations/attractions/general planning.

Few-shot examples demonstrate the decision logic across diverse query types, including context-aware extraction (e.g., "What's the weather there?" after Tokyo was mentioned). Strict JSON output format (`{"weather": {...} | null, "country": {...} | null}`) enables reliable parsing and prevents LLM creativity from breaking downstream logic. The prompt receives resolved messages (not raw), so references like "there" are already clarified from prior turns.

## Reference Resolution

Follow-up questions often contain ambiguous references: "What's the weather there?" or "How much does it cost?" The reference resolution prompt adds parenthetical annotations like "there (meaning: Tokyo)" or "it (meaning: Tokyo Skytree admission)" to disambiguate these references for future turns.

This is critical because the LLM only sees a limited window of recent messages (not the full conversation history). Without resolution, a message like "What's the weather there?" becomes meaningless once the earlier message mentioning "Tokyo" has scrolled out of the context window. By annotating "there (meaning: Tokyo)" at the time the reference is fresh, the resolved version carries the full meaning forward even after the original context is gone. The prompt explicitly instructs preservation of exact message structure — only annotations are added, never rephrasing. This prevents summary drift and maintains conversation fidelity. The task runs in the background since resolved messages aren't needed until the next turn.

## Rolling Summaries (Trip + User)

Why free text over structured data? Travel planning involves fluid, open-ended information that resists rigid schemas. A user might mention "I prefer quieter neighborhoods" or "We'll have jetlag the first day"—statements that don't fit neatly into predefined fields but are valuable for personalization.

Both trip and user summary prompts use an incremental update pattern with explicit "return unchanged" instructions. This prevents the common LLM failure mode of hallucinating changes or adding speculation. The prompts clearly distinguish between exploratory conversation (no update) and actual decisions (update required). Summaries act as compressed long-term memory, persisting key information even as older messages fall out of the context window during longer conversations.

## Error Handling & Anti-Hallucination Strategy

Handling confused responses and hallucinations is a multi-layered problem. Post-hoc hallucination detection (e.g., a second LLM call to fact-check the first) is unreliable and adds latency, so this system focuses on **prevention through prompt design**, **structural recovery in the pipeline**, and **graceful degradation when data is unavailable**.

### Layer 1: Prevention (Prompt-Level)

The answer prompt includes explicit anti-hallucination instructions that steer the LLM away from confident fabrication:

- **Never invent specific facts** (prices, opening hours, phone numbers, visa fees) — the prompt forbids making up precise data the model isn't sure about
- **Prefer honesty over confidence** — phrases like "I'd recommend checking..." or "Prices vary, but typically..." are encouraged instead of fabricated specifics
- **Caveat time-sensitive information** — visa rules, COVID restrictions, and travel regulations are flagged as potentially outdated
- **Source transparency** — the prompt requires the LLM to signal whether information comes from live data ("The current forecast shows...") or general knowledge ("Typically this time of year..."). This lets the user calibrate how much to trust each claim
- **Chain-of-thought reasoning** — the 5-step CoT framework (UNDERSTAND → CONTEXT → DATA → PLAN → CAVEATS) forces the model to explicitly consider what data it has, what gaps exist, and what assumptions it's making *before* generating the response. Step 5 (CAVEATS) specifically asks: "Am I uncertain about any claims? Should I caveat?"

### Layer 2: Structural Recovery (Pipeline-Level)

Each LLM call in the pipeline has parsing and fallback logic to recover from malformed or confused model output:

- **Extract (JSON parsing)**: If the model returns invalid JSON, the system tries regex extraction from the response. If that also fails, it retries with a corrective prompt ("Your response was not valid JSON. Please output ONLY the JSON object."). If all parsing fails, it returns an empty result (no API calls made) — the conversation continues normally, just without external data.
- **Answer ([RESPONSE]: marker)**: If the model omits the CoT `[RESPONSE]:` marker, the system retries with a corrective prompt asking the model to rewrite its answer starting with the marker. This prevents internal chain-of-thought reasoning from leaking to the user. If the retry also fails, a friendly fallback message is shown instead of exposing raw reasoning. If the model returns None, the same fallback is used.
- **Resolve references (JSON array)**: If parsing fails, the system falls back to the original unresolved messages. The conversation continues with slightly less context clarity, but doesn't break.
- **Update summaries (trip/user)**: If the LLM returns None, the current summary is preserved unchanged. Background task failures are caught and logged but never affect the user-facing response.

### Layer 3: Graceful Degradation (External Data)

When external APIs fail, the system doesn't pass raw errors to the LLM. Instead, it uses standardized markers (`WEATHER_UNAVAILABLE`, `COUNTRY_UNAVAILABLE`) that the answer prompt is trained to handle:

- **Weather unavailable**: The LLM falls back to general seasonal/climate knowledge and signals this to the user: "I don't have the latest forecast, but typically this time of year..."
- **Country data unavailable**: The LLM uses general knowledge with appropriate uncertainty: "From what I know, the currency is... but I'd recommend double-checking on the official embassy site."
- **Forecast time limitation**: Weather forecasts only cover ~5 days. For trips further out, the prompt instructs combining available data with seasonal patterns while noting the constraint.

This three-layer approach ensures the system degrades gracefully at every level — from individual LLM output parsing to external service failures — while maintaining a natural, trustworthy conversation with the user.

## Background Processing

Three post-processing tasks run asynchronously after responding to the user: reference resolution, trip summary update, and user data summary update. This architecture leverages natural conversation latency—users typically take several seconds to type their next message, which is plenty of time for background LLM calls to complete.

The benefit: richer context and better personalization without adding perceived latency. Reference resolution makes subsequent extractions more accurate. Summary updates enable long-term memory and personalization that improves over the conversation. Failed background tasks degrade gracefully (previous state preserved), so the critical user-facing path remains reliable even if enrichment tasks fail.
