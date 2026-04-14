import asyncio
import sys
import logging
from conversation import ConversationState

# Suppress all logging output in CLI mode
logging.disable(logging.CRITICAL)


async def main():
    """Main CLI loop for the Travel Assistant."""
    state = ConversationState()

    # Print welcome message
    print("\n  Travel Assistant")
    print("  Your AI-powered travel planning companion")
    print("  Type your message, '/reset' to start over, or 'quit' to exit.\n")

    while True:
        try:
            # Use run_in_executor for non-blocking input
            user_input = await asyncio.get_running_loop().run_in_executor(
                None, lambda: input("You: ")
            )
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Happy travels!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("\nGoodbye! Happy travels!")
            break

        if user_input.lower() == '/reset':
            state.reset()
            print("\n  Conversation reset. Let's start fresh!\n")
            continue

        print("\nAssistant: ", end="", flush=True)

        try:
            async for chunk in state.process_turn_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception:
            print(f"\nI apologize, but I encountered an error processing your request. Please try again.\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye! Happy travels!")
        sys.exit(0)
