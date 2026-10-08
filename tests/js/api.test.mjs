// Tests for api.js: NDJSON streaming and the 60 s first-event / idle timers (Req 1.8).
// Uses a fake fetch returning a ReadableStream and injected timer functions.

import assert from "node:assert/strict";
import { test } from "node:test";

import { streamChat, transcribe } from "../../src/friday/static/js/api.js";
import { CLIENT_TIMEOUT_S } from "../../src/friday/static/js/constants.js";

const TIMEOUT_MS = CLIENT_TIMEOUT_S * 1000;
const enc = new TextEncoder();

// A fake controllable timer set. Call `fire(id)` to trip a scheduled timer.
function fakeTimers() {
  let nextId = 1;
  const pending = new Map();
  return {
    setTimer(fn, ms) {
      const id = nextId++;
      pending.set(id, { fn, ms });
      return id;
    },
    clearTimer(id) {
      pending.delete(id);
    },
    fire(id) {
      const t = pending.get(id);
      pending.delete(id);
      if (t) t.fn();
    },
    get active() {
      return [...pending.keys()];
    },
  };
}

// A fake Response whose body streams the given NDJSON lines.
function ndjsonResponse(lines, { ok = true } = {}) {
  let i = 0;
  return {
    ok,
    body: {
      getReader() {
        return {
          async read() {
            if (i < lines.length) {
              const chunk = enc.encode(lines[i] + "\n");
              i += 1;
              return { value: chunk, done: false };
            }
            return { value: undefined, done: true };
          },
        };
      },
    },
  };
}

test("parses NDJSON events in order and sends the session header", async () => {
  const events = [];
  const seen = {};
  const fetchImpl = async (url, opts) => {
    seen.url = url;
    seen.header = opts.headers["X-Friday-Session"];
    seen.body = JSON.parse(opts.body);
    return ndjsonResponse([
      JSON.stringify({ type: "status", seq: 1 }),
      JSON.stringify({ type: "final", seq: 2, outcome: "ok" }),
    ]);
  };
  const timers = fakeTimers();
  await streamChat(
    { sessionId: "sess-1", text: "hi", onEvent: (e) => events.push(e), onFailure: () => {} },
    { fetchImpl, setTimer: timers.setTimer, clearTimer: timers.clearTimer },
  );
  assert.equal(seen.url, "/api/chat");
  assert.equal(seen.header, "sess-1");
  assert.deepEqual(seen.body, { text: "hi" });
  assert.deepEqual(
    events.map((e) => e.type),
    ["status", "final"],
  );
});

test("the first-event timer fails the stream when it fires (Req 1.8)", async () => {
  let failure = null;
  const timers = fakeTimers();
  // A fetch that only settles when its abort signal fires, mirroring the browser: the
  // timeout calls controller.abort(), which rejects the pending fetch with AbortError.
  const fetchImpl = (_url, opts) =>
    new Promise((_resolve, reject) => {
      opts.signal.addEventListener("abort", () => {
        const err = new Error("aborted");
        err.name = "AbortError";
        reject(err);
      });
    });
  const promise = streamChat(
    { sessionId: "s", text: "x", onEvent: () => {}, onFailure: (m) => (failure = m) },
    { fetchImpl, setTimer: timers.setTimer, clearTimer: timers.clearTimer },
  );
  // One timer armed for the first event, at the 60 s budget.
  assert.equal(timers.active.length, 1);
  timers.fire(timers.active[0]);
  await promise;
  assert.equal(failure, "timeout");
});

test("the idle timer is re-armed after each event", async () => {
  const armedMs = [];
  const timers = fakeTimers();
  const wrapSet = (fn, ms) => {
    armedMs.push(ms);
    return timers.setTimer(fn, ms);
  };
  const fetchImpl = async () =>
    ndjsonResponse([
      JSON.stringify({ type: "status", seq: 1 }),
      JSON.stringify({ type: "status", seq: 2 }),
      JSON.stringify({ type: "final", seq: 3, outcome: "ok" }),
    ]);
  await streamChat(
    { sessionId: "s", text: "x", onEvent: () => {}, onFailure: () => {} },
    { fetchImpl, setTimer: wrapSet, clearTimer: timers.clearTimer },
  );
  // First-event timer + one re-arm per event, all at the 60 s budget.
  assert.ok(armedMs.length >= 4);
  assert.ok(armedMs.every((ms) => ms === TIMEOUT_MS));
});

test("a non-ok response fails with 'error'", async () => {
  let failure = null;
  const timers = fakeTimers();
  const fetchImpl = async () => ndjsonResponse([], { ok: false });
  await streamChat(
    { sessionId: "s", text: "x", onEvent: () => {}, onFailure: (m) => (failure = m) },
    { fetchImpl, setTimer: timers.setTimer, clearTimer: timers.clearTimer },
  );
  assert.equal(failure, "error");
});

test("transcribe returns the text on a 200", async () => {
  const fetchImpl = async () => ({ ok: true, json: async () => ({ text: "pull inflation" }) });
  const result = await transcribe({ sessionId: "s", pcm: new Uint8Array([1]) }, fetchImpl);
  assert.deepEqual(result, { ok: true, text: "pull inflation" });
});

test("transcribe surfaces the error code on a 4xx", async () => {
  const fetchImpl = async () => ({
    ok: false,
    json: async () => ({ error: { code: "stt_empty", message: "nothing heard" } }),
  });
  const result = await transcribe({ sessionId: "s", pcm: new Uint8Array([1]) }, fetchImpl);
  assert.equal(result.ok, false);
  assert.equal(result.code, "stt_empty");
});
