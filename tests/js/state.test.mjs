// Exhaustive state x event transition table for the Assistant_State reducer
// (Req 2.1, 2.7-2.12, 3.1, 3.2, 3.9, 3.10).

import assert from "node:assert/strict";
import { test } from "node:test";

import { reduce, STATES, INITIAL_STATE } from "../../src/friday/static/js/state.js";

const EVENTS = [
  "session_start",
  "mic_granted",
  "mic_denied",
  "mic_stop",
  "submit",
  "clip_started",
  "playback_drained",
  "response_no_audio",
  "failure",
];

// Expected next state per (state, event) with the default ctx (no pending, empty queue).
// `null` means "unchanged" (no-op).
const TABLE = {
  idle: {
    session_start: "idle",
    mic_granted: "listening",
    mic_denied: "idle",
    mic_stop: null,
    submit: "working",
    clip_started: "speaking",
    playback_drained: "idle",
    response_no_audio: "idle",
    failure: "idle",
  },
  listening: {
    session_start: "idle",
    mic_granted: null,
    mic_denied: "idle",
    mic_stop: "working",
    submit: "working",
    clip_started: "speaking",
    playback_drained: "idle",
    response_no_audio: "idle",
    failure: "idle",
  },
  working: {
    session_start: "idle",
    mic_granted: null, // mic clicks are no-ops while working (Req 3.9)
    mic_denied: "idle",
    mic_stop: null,
    submit: null, // already working
    clip_started: "speaking",
    playback_drained: "idle",
    response_no_audio: "idle",
    failure: "idle",
  },
  speaking: {
    session_start: "idle",
    mic_granted: "listening", // mic click stops playback and listens (Req 3.1)
    mic_denied: "idle",
    mic_stop: null,
    submit: "working",
    clip_started: "speaking",
    playback_drained: "idle",
    response_no_audio: "idle",
    failure: "idle",
  },
};

test("initial state is idle (Req 2.12)", () => {
  assert.equal(INITIAL_STATE, "idle");
});

test("exhaustive transition table (default ctx)", () => {
  for (const state of STATES) {
    for (const type of EVENTS) {
      const expected = TABLE[state][type];
      const next = reduce(state, { type }, { pendingRequests: 0, queueLength: 0 });
      const want = expected === null ? state : expected;
      assert.equal(
        next,
        want,
        `reduce(${state}, ${type}) => ${next}, expected ${want}`,
      );
    }
  }
});

test("unknown events are no-ops", () => {
  for (const state of STATES) {
    assert.equal(reduce(state, { type: "nonsense" }), state);
  }
});

test("drain with a pending request stays working, not idle (Req 2.8)", () => {
  assert.equal(reduce("working", { type: "playback_drained" }, { pendingRequests: 1 }), "working");
  assert.equal(
    reduce("speaking", { type: "response_no_audio" }, { pendingRequests: 2 }),
    "working",
  );
});

test("drain with clips still queued stays speaking", () => {
  assert.equal(reduce("speaking", { type: "playback_drained" }, { queueLength: 1 }), "speaking");
});

test("drain with nothing pending or queued goes idle (Req 2.8)", () => {
  assert.equal(
    reduce("speaking", { type: "playback_drained" }, { pendingRequests: 0, queueLength: 0 }),
    "idle",
  );
});
