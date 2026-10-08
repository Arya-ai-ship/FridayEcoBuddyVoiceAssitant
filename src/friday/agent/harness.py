"""Build Friday's MAF harness agent with only the four Friday tools (Req 5.1, 5.4, 5.14, 5.15).

``build_harness_agent`` calls ``create_harness_agent`` with Friday's rules and persona as
the only instructions, the four Friday function tools, the Dataset-inventory context
provider, and the Friday middleware. Every harness capability is disabled: no todo, mode,
file memory, web search, tool auto-approval, or compaction, and no skills, file access,
shell, background agents, or looping. The ``DEFAULT_HARNESS_INSTRUCTIONS`` therefore never
reach the model (spike item 2).

The function-invocation iteration and consecutive-error limits are set on the chat client
in ``llm/bedrock.py`` (spike item 3), not here. Friday's own ``MAX_TOOL_CALLS`` termination
in the function middleware is the real budget; those client limits are only a backstop.

``history_provider`` is Friday's own ``InMemoryHistoryProvider`` so ``run_turn`` can append
the fixed outcome text to the Session history after a short-circuit (spike item 6).
"""

from agent_framework import (
    Agent,
    BaseChatClient,  # the LLM port; BedrockChatClient and the fake both subclass it
    InMemoryHistoryProvider,
    create_harness_agent,
)

from friday.agent.middleware import (
    DatasetInventoryProvider,
    FridayMiddleware,
    IndicatorMapProvider,
    current_turn,
)
from friday.agent.prompts import FRIDAY_PERSONA, FRIDAY_RULES
from friday.agent.tools import ToolRegistry
from friday.data.indicators import IndicatorMap

HARNESS_NAME = "Friday"
"""The harness agent's name; the model identifies itself as Friday (Req 4.1)."""


def build_harness_agent(
    chat_client: BaseChatClient,
    tools: ToolRegistry,
    middleware: FridayMiddleware,
    history: InMemoryHistoryProvider,
    indicators: IndicatorMap,
) -> Agent:
    """Construct the Friday harness agent (one instance, shared by all Sessions).

    ``history`` is Friday's own provider so ``run_turn`` can append messages to a Session.
    ``indicators`` is passed to ``IndicatorMapProvider`` so the model always knows every
    supported indicator, its aliases, series ID, and transformation.
    Every opt-in capability is left off; only the four Friday tools and Friday's own
    instructions are exposed to the model.
    """
    return create_harness_agent(
        client=chat_client,
        name=HARNESS_NAME,
        harness_instructions=FRIDAY_RULES,
        agent_instructions=FRIDAY_PERSONA,
        tools=tools.function_tools(current_turn),
        history_provider=history,
        context_providers=[IndicatorMapProvider(indicators), DatasetInventoryProvider()],
        middleware=[middleware.function, middleware.chat],
        disable_todo=True,
        disable_mode=True,
        disable_file_memory=True,
        disable_web_search=True,
        disable_tool_auto_approval=True,
        disable_compaction=True,
    )
