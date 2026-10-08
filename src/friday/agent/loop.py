"""The ``Agent`` facade: drive one MAF harness turn and stream Friday's NDJSON events.

``run_turn`` snapshots the Session's ``AgentSession``, runs the harness as a task while
yielding queued events, and ends every turn with exactly one ``final`` event (design:
``run_turn`` algorithm). On a credential or model failure it rolls the conversation
history back to the snapshot (the harness persists history after each model call); Datasets
and charts created earlier in the turn stay. A ``tool_limit``/``no_data`` stop emits the
fixed outcome text. After every successful or stopped turn the stored history is completed
so each function call has exactly one result (Converse requires it). Otherwise the model's
reply is parsed, guarded, given a next-step offer, synthesized, and emitted as
``final(outcome=ok)``.

MAF is imported as ``maf`` so ``maf.Agent`` never collides with Friday's ``Agent``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final, Literal

import agent_framework as maf

from friday.agent import narration
from friday.agent.middleware import (
    TurnContext,
    reset_current_turn,
    set_current_turn,
    speak,
    tool_limit_result,
)
from friday.agent.narration import grounded_numbers
from friday.agent.templates import OUTCOME_TEXTS
from friday.constants import TTS_TIMEOUT_S
from friday.errors import AwsCredentialError
from friday.events import AudioErrorKind, Event, TurnEvents
from friday.ports import TTSClient
from friday.redact import Redactor
from friday.session import Session

_log: Final = logging.getLogger(__name__)

_QUEUE_SENTINEL: Final = object()
"""Marks the end of the event queue once the harness task finishes."""

SpokenOutcome = Literal["llm_unavailable", "tool_limit", "no_data"]
"""Fixed outcomes whose text is also spoken (``aws_credentials`` never is, Req 4.8)."""


def snapshot_state(agent_session: maf.AgentSession) -> dict[str, Any]:
    """A JSON-serializable copy of the Session's conversation state (spike item 6)."""
    return agent_session.to_dict()


def restore_state(agent_session: maf.AgentSession, snapshot: dict[str, Any]) -> None:
    """Reset ``agent_session`` to a prior :func:`snapshot_state` (spike item 6)."""
    agent_session.state = maf.AgentSession.from_dict(snapshot).state


class Agent:
    """Friday's facade over one shared MAF harness agent."""

    def __init__(
        self,
        harness: maf.Agent,
        tts: TTSClient,
        history: maf.InMemoryHistoryProvider,
        redactor: Redactor,
    ) -> None:
        """Use ``harness`` for the tool loop, ``tts`` for Friday's reply, ``history`` to append."""
        self._harness = harness
        self._tts = tts
        self._history = history
        self._redactor = redactor

    async def run_turn(self, session: Session, user_text: str) -> AsyncIterator[Event]:
        """Run one user turn and yield its events, ending with exactly one ``final``."""
        events = TurnEvents(session.session_id)
        queue: asyncio.Queue[Event] = asyncio.Queue()
        turn = TurnContext(session=session, events=events, queue=queue, tts_timeout_s=TTS_TIMEOUT_S)
        snapshot = snapshot_state(session.agent_session)
        token = set_current_turn(turn)
        try:
            async for event in self._drive(session, user_text, turn, snapshot):
                yield event
        finally:
            reset_current_turn(token)

    async def _drive(
        self,
        session: Session,
        user_text: str,
        turn: TurnContext,
        snapshot: dict[str, Any],
    ) -> AsyncIterator[Event]:
        task = asyncio.create_task(self._run_harness(session, user_text, turn))
        error: BaseException | None = None

        async def watch() -> None:
            nonlocal error
            try:
                await task
            except BaseException as exc:  # noqa: BLE001 - recorded and classified below
                error = exc
            finally:
                await turn.queue.put(_QUEUE_SENTINEL)  # type: ignore[arg-type]

        watcher = asyncio.create_task(watch())
        try:
            while True:
                item = await turn.queue.get()
                if item is _QUEUE_SENTINEL:
                    break
                yield item
        finally:
            await watcher

        async for event in self._finish(session, turn, snapshot, task, error):
            yield event

    async def _run_harness(self, session: Session, user_text: str, turn: TurnContext) -> Any:
        """Run the harness; the middleware fills ``turn`` and the event queue."""
        return await self._harness.run(user_text, session=session.agent_session)

    async def _finish(
        self,
        session: Session,
        turn: TurnContext,
        snapshot: dict[str, Any],
        task: "asyncio.Task[Any]",
        error: BaseException | None,
    ) -> AsyncIterator[Event]:
        if error is not None:
            yield await self._handle_error(session, turn, snapshot, error)
            return
        if turn.stop is not None:
            response = task.result()
            yield await self._handle_stop(session, turn, response)
            return
        async for event in self._handle_success(session, turn, task.result()):
            yield event

    async def _handle_error(
        self,
        session: Session,
        turn: TurnContext,
        snapshot: dict[str, Any],
        error: BaseException,
    ) -> Event:
        """Roll history back and emit the credential or unavailable outcome (Req 5.7, 5.12)."""
        restore_state(session.agent_session, snapshot)
        classified = turn.llm_error or error
        if isinstance(classified, AwsCredentialError):
            text = OUTCOME_TEXTS["aws_credentials"]
            return turn.events.final("aws_credentials", text, "")  # never spoken (Req 4.8)
        _log.warning("Turn failed: %s", type(classified).__name__)
        return await self._fixed_outcome(turn, "llm_unavailable")

    async def _handle_stop(self, session: Session, turn: TurnContext, response: Any) -> Event:
        """Emit the fixed stop-outcome text and complete the history (Req 5.9, 8.6)."""
        outcome = turn.stop
        if outcome is None:  # defensive: _finish only calls this when stop is set
            raise RuntimeError("stop outcome missing")
        await self._complete_history(session, response, stop_text=OUTCOME_TEXTS[outcome])
        return await self._fixed_outcome(turn, outcome)

    async def _fixed_outcome(self, turn: TurnContext, outcome: SpokenOutcome) -> Event:
        """A ``final`` event with a fixed outcome text, spoken as well as shown (design)."""
        text = OUTCOME_TEXTS[outcome]
        audio, audio_error = await self._synthesize(text)
        return turn.events.final(outcome, text, text, audio=audio, audio_error=audio_error)

    async def _handle_success(
        self, session: Session, turn: TurnContext, response: Any
    ) -> AsyncIterator[Event]:
        """Parse, guard, offer, synthesize, and emit ``final(outcome=ok)`` (Req 5.10, 8.1-8.5)."""
        stored = await self._complete_history(session, response, stop_text=None)
        results = [*_tool_result_texts(stored), *(o.result_json for o in turn.outcomes)]
        display, spoken = narration.parse_reply(response.text or "")
        offer = narration.next_step_offer(turn.outcomes, session)
        offer_words = len(offer.split()) if offer else 0
        display, spoken = narration.guard(
            display, spoken, grounded_numbers(session, results), turn.status_words + offer_words
        )
        if offer:
            display = f"{display}\n\n{offer}" if display else offer
            spoken = f"{spoken} {offer}".strip()
        audio, audio_error = await self._synthesize(spoken)
        yield turn.events.final("ok", display, spoken, audio=audio, audio_error=audio_error)

    async def _synthesize(self, spoken: str) -> tuple[bytes | None, AudioErrorKind | None]:
        """Synthesize ``spoken`` within the TTS timeout; map a failure to an audio error."""
        if not spoken.strip():
            return None, None
        return await speak(self._tts, spoken, TTS_TIMEOUT_S)

    async def _complete_history(
        self, session: Session, response: Any, *, stop_text: str | None
    ) -> list[Any]:
        """Keep the stored history valid for the next Converse request; return it.

        Converse needs every function call followed by exactly one result. When the function
        middleware terminates a run, the harness does not store the last batch's results
        (spike item 5), so they are taken from the run response; a call with no result
        there gets a ``tool_limit`` result. Results already stored are never added again.
        A stopped turn then ends with its fixed outcome text, unless the run already
        stored it (the chat middleware's short-circuit reply). Returns the stored messages
        afterwards (empty if the history could not be read).
        """
        agent_session = session.agent_session
        state = agent_session.state.setdefault(self._history.source_id, {})
        try:
            stored = await self._history.get_messages(agent_session.session_id, state=state)
            additions: list[maf.Message] = []
            results = _results_for_dangling_calls(stored, response)
            if results:
                additions.append(maf.Message(role="tool", contents=results))
            if stop_text is not None and (additions or not _ends_with_text(stored, stop_text)):
                additions.append(maf.Message(role="assistant", contents=[stop_text]))
            if additions:
                await self._history.save_messages(agent_session.session_id, additions, state=state)
            return [*stored, *additions]
        except Exception as exc:  # the turn's final event must still be sent
            _log.warning("Could not complete the Session history: %s", type(exc).__name__)
            return []


def _dangling_calls(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """``(call_id, tool name)`` of stored function calls that have no stored result."""
    calls: dict[str, str] = {}
    answered: set[str] = set()
    for message in messages:
        for content in getattr(message, "contents", None) or ():
            call_id = getattr(content, "call_id", None)
            if not isinstance(call_id, str):
                continue
            kind = getattr(content, "type", None)
            if kind == "function_call":
                calls.setdefault(call_id, str(getattr(content, "name", "")))
            elif kind == "function_result":
                answered.add(call_id)
    return [(call_id, name) for call_id, name in calls.items() if call_id not in answered]


def _results_for_dangling_calls(stored: Sequence[Any], response: Any) -> list[maf.Content]:
    """One result per dangling call: the run response's result, else a ``tool_limit`` result."""
    dangling = _dangling_calls(stored)
    if not dangling:
        return []
    available: dict[str, maf.Content] = {}
    for message in getattr(response, "messages", None) or ():
        for content in getattr(message, "contents", None) or ():
            call_id = getattr(content, "call_id", None)
            if getattr(content, "type", None) == "function_result" and isinstance(call_id, str):
                available.setdefault(call_id, content)
    return [
        available[call_id]
        if call_id in available
        else maf.Content.from_function_result(call_id, result=tool_limit_result(name))
        for call_id, name in dangling
    ]


def _tool_result_texts(messages: Sequence[Any]) -> list[str]:
    """The text of every function result stored in the Session history."""
    texts: list[str] = []
    for message in messages:
        for content in getattr(message, "contents", None) or ():
            result = getattr(content, "result", None)
            if getattr(content, "type", None) == "function_result" and result is not None:
                texts.append(result if isinstance(result, str) else str(result))
    return texts


def _ends_with_text(messages: Sequence[Any], text: str) -> bool:
    """Whether the last stored message is an assistant message with exactly ``text``."""
    if not messages:
        return False
    last = messages[-1]
    return getattr(last, "role", None) == "assistant" and (last.text or "").strip() == text
