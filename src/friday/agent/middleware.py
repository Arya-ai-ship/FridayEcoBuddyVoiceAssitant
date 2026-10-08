"""Friday's MAF middleware: the function and chat wrappers plus the inventory provider.

``run_turn`` (``agent/loop.py``) sets a :class:`TurnContext` in ``_CURRENT_TURN`` before
starting the harness run. Asyncio tasks copy the context, so the function middleware, the
chat middleware, and the context provider all see the right turn even when several
Sessions run at once (design: Turn context and middleware).

- :class:`FunctionGuard` counts every tool call, enforces the ``MAX_TOOL_CALLS`` budget
  and the no-data short-circuit, validates arguments, and brackets each Tool with
  ``status(tool_start)`` / display / ``status(tool_done|tool_error)`` events, speaking the
  start line concurrently with the Tool (Req 4.4, 4.5, 4.11, 5.5, 5.8, 5.9, 5.11, 8.6, 10.10).
- :class:`ChatGuard` bounds each model call with ``asyncio.wait_for(LLM_TIMEOUT_S)``, maps
  any failure through the injected ``map_llm_error`` into ``TurnContext.llm_error``, and
  counts unknown tool names that MAF answered without reaching the function middleware
  (Req 5.7, 5.8, 5.12).
- :class:`DatasetInventoryProvider` adds the per-run Dataset inventory before each model
  call (Req 8.4).

This module never imports ``aws_errors``, botocore, or the MAF Bedrock package: the
``map_llm_error`` callable is injected by ``wiring.py``.
"""

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from agent_framework import (
    ChatContext,
    ChatMiddleware,
    ChatResponse,
    ContextProvider,
    FunctionInvocationContext,
    FunctionMiddleware,
    Message,
    MiddlewareTermination,
    SessionContext,
)

from friday.agent import narration
from friday.agent.prompts import dataset_inventory, indicator_catalogue
from friday.agent.templates import OUTCOME_TEXTS, TOOL_NAMES, ToolName
from friday.agent.tool_runs import ToolOutcome, to_json
from friday.agent.tools import ToolRegistry
from friday.constants import LLM_TIMEOUT_S, MAX_STATUS_WORDS, MAX_TOOL_CALLS
from friday.data.indicators import IndicatorMap
from friday.errors import AwsCredentialError, FridayError, ToolError, TTSError
from friday.events import AudioErrorKind, Event, Phase, TurnEvents, audio_error_kind
from friday.ports import TTSClient
from friday.session import Session

_log: Final = logging.getLogger(__name__)

MapLlmError = Callable[[BaseException], FridayError]
"""Maps a model-call failure to a ``FridayError`` (injected from ``llm/bedrock.py``)."""

StopReason = Literal["tool_limit", "no_data"]
"""Why the function middleware terminated the run early."""

_NO_DATA_TOOLS: Final = frozenset({"describe_data", "fill_missing", "plot_data"})
"""Analysis tools that need at least one Dataset in the Session (Req 8.6)."""

_INVENTORY_SOURCE: Final = "friday_inventory"
"""Context-provider source id for the per-run Dataset inventory."""

_CATALOGUE_SOURCE: Final = "friday_indicator_catalogue"
"""Context-provider source id for the supported-indicator catalogue."""


# --- Turn context -----------------------------------------------------------


@dataclass
class TurnContext:
    """Mutable state of one user turn, shared by the middleware and context provider."""

    session: Session
    events: "TurnEvents"
    queue: "asyncio.Queue[Event]"
    tts_timeout_s: float
    calls: int = 0
    outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])
    stop: StopReason | None = None
    llm_error: FridayError | None = None
    status_words: int = 0
    """Words in the status lines spoken so far this turn (part of the 60-word limit)."""


_CURRENT_TURN: "contextvars.ContextVar[TurnContext]" = contextvars.ContextVar("friday_turn")


def set_current_turn(turn: TurnContext) -> contextvars.Token[TurnContext]:
    """Bind ``turn`` as the running turn; the token restores the previous binding."""
    return _CURRENT_TURN.set(turn)


def reset_current_turn(token: contextvars.Token[TurnContext]) -> None:
    """Undo a :func:`set_current_turn` binding."""
    _CURRENT_TURN.reset(token)


def current_turn() -> TurnContext:
    """Return the running turn; raises ``LookupError`` when none is active."""
    return _CURRENT_TURN.get()


def tool_limit_result(tool: str) -> str:
    """The tool result for a call rejected by the ``MAX_TOOL_CALLS`` budget (Req 5.9)."""
    return to_json({"ok": False, "tool": tool, "error": {"kind": "tool_limit"}})


def _stop_reply(stop: StopReason) -> ChatResponse:
    """A text-only model reply carrying the fixed stop-outcome text (ends MAF's loop)."""
    return ChatResponse(messages=[Message(role="assistant", contents=[OUTCOME_TEXTS[stop]])])


def _no_data_result(tool: str) -> str:
    """The tool result for an analysis call with no Dataset loaded (Req 8.6)."""
    return to_json({"ok": False, "tool": tool, "error": {"kind": "no_data"}})


def _as_mapping(arguments: object) -> dict[str, Any]:
    """Normalize a function-call's arguments (a Pydantic model or mapping) to a dict."""
    dump = getattr(arguments, "model_dump", None)
    if callable(dump):
        return cast(dict[str, Any], dump())
    return dict(cast("dict[str, Any]", arguments)) if arguments else {}


# --- Function middleware -----------------------------------------------------


class FunctionGuard(FunctionMiddleware):
    """Counts, budgets, validates, and brackets every Tool call with status events."""

    def __init__(self, tools: ToolRegistry, tts: TTSClient) -> None:
        """Use ``tools`` to validate and ``tts`` to speak start lines concurrently."""
        self._tools = tools
        self._tts = tts

    async def process(
        self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        turn = current_turn()
        raw_name = context.function.name
        turn.calls += 1

        # 1. Budget: the 9th call ends the turn without running the Tool (Req 5.9).
        if turn.calls > MAX_TOOL_CALLS:
            context.result = tool_limit_result(raw_name)
            turn.stop = "tool_limit"
            raise MiddlewareTermination()

        # 2. No-data short-circuit for analysis tools (Req 8.6).
        if raw_name in _NO_DATA_TOOLS and not turn.session.datasets:
            context.result = _no_data_result(raw_name)
            turn.stop = "no_data"
            raise MiddlewareTermination()

        # Only the four Friday names reach function middleware (spike item 4).
        name: ToolName = cast(ToolName, raw_name)

        # 3. Validate; a bad call becomes a named-field error with no status event (Req 5.8).
        raw_args = _as_mapping(context.arguments)
        try:
            args = self._tools.validate(name, raw_args)
        except ToolError as exc:
            failure = ToolOutcome.failure(name, exc)
            context.result = failure.result_json
            turn.outcomes.append(failure)
            return

        # 4. Speak the start line while the Tool runs. tool_start is queued as soon as its
        # audio is ready, and always before the Tool's display event (Req 4.4).
        start_text = narration.start_line(name, raw_args, self._tools.indicator_name(args))
        start = asyncio.create_task(self._emit_status(turn, "tool_start", name, start_text))
        recorded = len(turn.outcomes)
        try:
            # 5. Run the Tool (its body appends the outcome to turn.outcomes).
            await call_next()
        except BaseException:
            start.cancel()
            raise
        await start
        if len(turn.outcomes) <= recorded:
            return
        outcome = turn.outcomes[recorded]

        # 6. Display event, then the spoken done/error status line (Req 4.5, 4.11).
        display = outcome.display_event(turn.events)
        if display is not None:
            await turn.queue.put(display)
        if outcome.ok:
            await self._emit_status(turn, "tool_done", name, narration.done_line(name))
        else:
            line = narration.error_line(name, outcome.indicator)
            await self._emit_status(turn, "tool_error", name, line)

    async def _emit_status(self, turn: TurnContext, phase: Phase, tool: str, text: str) -> None:
        """Queue a status line, spoken while the turn's status budget lasts (Req 4.4, 4.10).

        A line that would push the turn past ``MAX_STATUS_WORDS`` (only possible with many
        tool calls in one turn) is still shown in the chat but carries no audio.
        """
        words = len(text.split())
        audio: bytes | None = None
        audio_error: AudioErrorKind | None = None
        if turn.status_words + words <= MAX_STATUS_WORDS:
            turn.status_words += words
            audio, audio_error = await speak(self._tts, text, turn.tts_timeout_s)
        event = turn.events.status(phase, tool, text, audio=audio, audio_error=audio_error)
        await turn.queue.put(event)


async def speak(
    tts: TTSClient, text: str, timeout_s: float
) -> tuple[bytes | None, AudioErrorKind | None]:
    """Synthesize ``text``; a failure becomes an ``audio_error`` kind, never an exception.

    The text is still shown without audio when synthesis fails (Req 4.8).
    """
    try:
        return await tts.synthesize(text, timeout_s), None
    except (TTSError, AwsCredentialError) as exc:
        return None, audio_error_kind(exc)
    except Exception as exc:
        _log.warning("Speech synthesis raised %s", type(exc).__name__)
        return None, "tts_failed"


# --- Chat middleware ---------------------------------------------------------


class ChatGuard(ChatMiddleware):
    """Bounds each model call and maps its failures to ``TurnContext.llm_error``."""

    def __init__(self, map_llm_error: MapLlmError, timeout_s: float = LLM_TIMEOUT_S) -> None:
        """Use ``map_llm_error`` to classify a failure; ``timeout_s`` per model call."""
        self._map = map_llm_error
        self._timeout_s = timeout_s

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        turn = current_turn()
        if turn.stop is not None:
            # The previous response used up the budget with unknown tool names, which MAF
            # answers itself. End the turn here instead of calling the model again (Req 5.9).
            # (MiddlewareTermination raised from chat middleware does not stop MAF's
            # function loop; a text-only reply does.)
            context.result = _stop_reply(turn.stop)
            return
        try:
            await asyncio.wait_for(call_next(), self._timeout_s)
        except TimeoutError as exc:
            turn.llm_error = self._map(exc)
            raise turn.llm_error from exc
        except FridayError:
            raise
        except Exception as exc:
            turn.llm_error = self._map(exc)
            raise turn.llm_error from exc
        self._count_unknown_names(turn, context)
        if turn.calls > MAX_TOOL_CALLS:
            # Known calls in this batch now hit the budget in the function middleware;
            # if there are none, the next model call is short-circuited above.
            turn.stop = "tool_limit"

    def _count_unknown_names(self, turn: TurnContext, context: ChatContext) -> None:
        """Count unknown-tool-name calls MAF answered without hitting function middleware.

        Unknown names never reach the function middleware (spike item 4); MAF returns an
        ``Error: Requested function "<name>" not found.`` tool result. Each such call still
        counts toward the budget (Req 5.8).
        """
        result = context.result
        messages = getattr(result, "messages", None) or ()
        for message in messages:
            for content in getattr(message, "contents", None) or ():
                if getattr(content, "type", None) != "function_call":
                    continue
                name = getattr(content, "name", None)
                if isinstance(name, str) and name not in self._tool_names:
                    turn.calls += 1

    # The four Friday tool names (single source: templates.TOOL_NAMES).
    _tool_names: Final[frozenset[str]] = frozenset(TOOL_NAMES)


# --- Context provider --------------------------------------------------------


class IndicatorMapProvider(ContextProvider):
    """Adds the supported-indicator catalogue to the instructions before each model call.

    The catalogue lists every fetchable indicator with its canonical name, aliases, FRED
    series ID, and transformation, so the model knows exactly what it can pull without
    guessing or hallucinating indicator names.
    """

    def __init__(self, indicators: IndicatorMap) -> None:
        super().__init__(_CATALOGUE_SOURCE)
        self._indicators = indicators

    async def before_run(
        self,
        *,
        agent: object,
        session: object,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject the indicator catalogue before every model call."""
        context.extend_instructions(self.source_id, indicator_catalogue(self._indicators))


class DatasetInventoryProvider(ContextProvider):
    """Adds the per-run Dataset inventory to the instructions before each model call."""

    def __init__(self) -> None:
        """Register under a fixed source id."""
        super().__init__(_INVENTORY_SOURCE)

    async def before_run(
        self,
        *,
        agent: object,
        session: object,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Append the inventory of the running turn's Session (Req 8.4)."""
        turn = current_turn()
        context.extend_instructions(self.source_id, dataset_inventory(turn.session))


# --- Facade ------------------------------------------------------------------


@dataclass(frozen=True)
class FridayMiddleware:
    """The function and chat middleware pair handed to ``build_harness_agent``."""

    function: FunctionGuard
    chat: ChatGuard

    @classmethod
    def build(
        cls, tools: ToolRegistry, tts: TTSClient, map_llm_error: MapLlmError
    ) -> "FridayMiddleware":
        """Construct the middleware pair from its collaborators."""
        return cls(FunctionGuard(tools, tts), ChatGuard(map_llm_error))
