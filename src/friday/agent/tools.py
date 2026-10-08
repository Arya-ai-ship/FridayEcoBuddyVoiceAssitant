"""The Tool registry: the four Friday Tools the harness exposes to the LLM (Req 5.4).

Argument models, specs, and validation live in ``agent.tool_args``; the Tool bodies and
``ToolOutcome`` live in ``agent.tool_runs``. This module is the registry facade the Agent,
the harness builder, and the Friday middleware use.

Coupling with the turn (task 12.1). ``function_tools(current_turn)`` takes a zero-argument
getter that returns the running turn as a ``ToolTurn`` (in production it reads the
``TurnContext`` contextvar). Each function tool body calls ``current_turn()`` to find the
Friday Session, runs the Tool through ``ToolRegistry.run``, appends the ``ToolOutcome`` to
``turn.outcomes`` (so the function middleware reads ``turn.outcomes[-1]`` after
``call_next()`` to queue the display and status events), and returns the outcome's JSON
result to MAF. That is the whole contract: the registry never imports the middleware.

No exception escapes a Tool: ``run`` returns any ``ToolError`` as its own result and any
other ``Exception`` as a generic ``tool_failed`` result whose message names only the Tool
(Req 5.11). ``asyncio.CancelledError`` (a ``BaseException``) still propagates, so a turn
timeout can cancel a running Tool.
"""

import logging
from collections.abc import Callable
from typing import Final, Protocol

from agent_framework import FunctionTool

from friday.agent.templates import ToolName
from friday.agent.tool_args import (
    DescribeArgs,
    FetchArgs,
    FillArgs,
    PlotArgs,
    ToolArgs,
    ToolSpec,
    tool_specs,
    validate_tool_args,
)
from friday.agent.tool_runs import (
    ToolOutcome,
    fetch_indicator,
    run_describe,
    run_fetch,
    run_fill,
    run_plot,
)
from friday.data.indicators import IndicatorMap
from friday.data.plot import ChartRenderer
from friday.errors import ToolError, ToolFailedError
from friday.ports import FredSource
from friday.session import Session

_log: Final = logging.getLogger(__name__)


class ToolTurn(Protocol):
    """What a function tool needs from the running turn (``TurnContext`` satisfies it)."""

    @property
    def session(self) -> Session:
        """The Friday Session the turn belongs to."""
        ...

    @property
    def outcomes(self) -> list[ToolOutcome]:
        """Outcomes of the turn's Tool calls, in call order; each body appends its own."""
        ...


CurrentTurn = Callable[[], ToolTurn]
"""Returns the running turn; raises (e.g. ``LookupError``) when no turn is active."""


class ToolRegistry:
    """The Fetch, Stats, Fill, and Plot Tools."""

    def __init__(self, fred: FredSource, renderer: ChartRenderer, indicators: IndicatorMap) -> None:
        """Use ``fred`` for observations, ``renderer`` for charts, ``indicators`` to resolve."""
        self._fred = fred
        self._renderer = renderer
        self._indicators = indicators

    @staticmethod
    def specs() -> list[ToolSpec]:
        """The four Tool specs, from the argument models (Req 5.4)."""
        return tool_specs()

    @staticmethod
    def validate(name: str, raw_args: object) -> ToolArgs:
        """Validate a Tool call; raises ``ToolValidationError`` (Req 5.8, 10.10, 11.8)."""
        return validate_tool_args(name, raw_args)

    async def run(self, session: Session, args: ToolArgs) -> ToolOutcome:
        """Run one validated Tool call; never raises an ``Exception`` (Req 5.11).

        On success the Session gains the new Dataset or chart; on failure it is unchanged
        and the outcome carries ``{"ok": false, "tool", "error"}``.
        """
        try:
            return await self._dispatch(session, args)
        except ToolError as exc:
            return ToolOutcome.failure(args.tool, exc, indicator=self.indicator_name(args))
        except Exception as exc:
            # Only the type is logged: the text may quote inputs or internals.
            _log.warning("Tool %s raised %s", args.tool, type(exc).__name__)
            failed = ToolFailedError(args.tool)
            return ToolOutcome.failure(args.tool, failed, indicator=self.indicator_name(args))

    async def _dispatch(self, session: Session, args: ToolArgs) -> ToolOutcome:
        match args:
            case FetchArgs():
                return await run_fetch(session, args, self._fred, self._indicators)
            case DescribeArgs():
                return await run_describe(session, args)
            case FillArgs():
                return await run_fill(session, args)
            case PlotArgs():
                return await run_plot(session, args, self._renderer)

    def indicator_name(self, args: ToolArgs) -> str | None:
        """The canonical Indicator a fetch names (for its status lines), else ``None``."""
        return fetch_indicator(args, self._indicators) if isinstance(args, FetchArgs) else None

    def function_tools(self, current_turn: CurrentTurn) -> list[FunctionTool]:
        """One MAF function tool per spec, bound to the turn ``current_turn`` returns.

        Each tool's ``input_model`` is the spec's flattened JSON schema, and approval is
        never required. Its body is ``call_tool``, so it returns the JSON result string.
        """
        return [self._function_tool(spec, current_turn) for spec in self.specs()]

    def _function_tool(self, spec: ToolSpec, current_turn: CurrentTurn) -> FunctionTool:
        name = spec.name

        async def body(**arguments: object) -> str:
            outcome = await self.call_tool(name, arguments, current_turn)
            return outcome.result_json

        return FunctionTool(
            name=name,
            description=spec.description,
            func=body,
            input_model=spec.input_schema,
            approval_mode="never_require",
        )

    async def call_tool(
        self, name: ToolName, raw_args: object, current_turn: CurrentTurn
    ) -> ToolOutcome:
        """A function tool body: validate, run on the turn's Session, record the outcome.

        The middleware has already validated ``raw_args``; validating again here yields
        the typed arguments (an invalid call still becomes a failed outcome). The outcome
        is appended to ``turn.outcomes``. Without a running turn the call fails with
        ``tool_failed`` and nothing is recorded. Never raises an ``Exception``.
        """
        try:
            turn = current_turn()
        except Exception as exc:
            _log.warning("Tool %s called outside a turn: %s", name, type(exc).__name__)
            return ToolOutcome.failure(name, ToolFailedError(name))
        try:
            args = self.validate(name, raw_args)
        except ToolError as exc:
            outcome = ToolOutcome.failure(name, exc)
        except Exception as exc:
            _log.warning("Tool %s arguments raised %s", name, type(exc).__name__)
            outcome = ToolOutcome.failure(name, ToolFailedError(name))
        else:
            outcome = await self.run(turn.session, args)
        turn.outcomes.append(outcome)
        return outcome
