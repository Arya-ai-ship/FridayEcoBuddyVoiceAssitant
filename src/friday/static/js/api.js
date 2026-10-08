// Backend transport: NDJSON chat stream, transcribe upload, session-end beacon
// (Req 1.8, 3.2, 3.6, 12.5).
//
// `streamChat` POSTs to /api/chat with the X-Friday-Session header and reads the
// application/x-ndjson response line by line, invoking `onEvent` for each parsed event.
// Two 60 s timers guard it: one until the first event arrives, then an idle timer reset
// after every event. On abort/timeout/error it calls `onFailure` once and stops (Req 1.8).
//
// Timers and `fetch` are injected (defaults bind to the real globals) so tests drive them
// with fakes (task 17.11).

import { CLIENT_TIMEOUT_S } from "./constants.js";

const CHAT_URL = "/api/chat";
const TRANSCRIBE_URL = "/api/transcribe";
const SESSION_END_URL = "/api/session/end";
const SESSION_HEADER = "X-Friday-Session";
const TIMEOUT_MS = CLIENT_TIMEOUT_S * 1000;

/**
 * @typedef {Object} StreamDeps
 * @property {typeof fetch} [fetchImpl]
 * @property {(fn: () => void, ms: number) => any} [setTimer]
 * @property {(id: any) => void} [clearTimer]
 */

/**
 * Stream a chat turn. Resolves when the stream ends or fails; never rejects.
 * @param {Object} opts
 * @param {string} opts.sessionId
 * @param {string} opts.text
 * @param {(event: object) => void} opts.onEvent
 * @param {(message: string) => void} opts.onFailure
 * @param {StreamDeps} [deps]
 * @returns {Promise<void>}
 */
export async function streamChat({ sessionId, text, onEvent, onFailure }, deps = {}) {
  const fetchImpl = deps.fetchImpl ?? globalThis.fetch.bind(globalThis);
  const setTimer = deps.setTimer ?? globalThis.setTimeout.bind(globalThis);
  const clearTimer = deps.clearTimer ?? globalThis.clearTimeout.bind(globalThis);

  const controller = new AbortController();
  let settled = false;
  let timer = null;

  const fail = (message) => {
    if (settled) return;
    settled = true;
    if (timer !== null) clearTimer(timer);
    controller.abort();
    onFailure(message);
  };
  const arm = () => {
    if (timer !== null) clearTimer(timer);
    timer = setTimer(() => fail("timeout"), TIMEOUT_MS);
  };

  arm(); // first-event timer (Req 1.8)
  try {
    const response = await fetchImpl(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json", [SESSION_HEADER]: sessionId },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      fail("error");
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line.length === 0) continue;
        arm(); // idle timer between events (Req 1.8)
        let event;
        try {
          event = JSON.parse(line);
        } catch {
          fail("error");
          return;
        }
        if (!settled) onEvent(event);
      }
    }
    if (!settled) {
      settled = true;
      if (timer !== null) clearTimer(timer);
    }
  } catch (err) {
    // AbortError from a timeout is already handled by fail(); ignore the double.
    if (!settled) fail(err && err.name === "AbortError" ? "timeout" : "error");
  }
}

/**
 * Upload recorded PCM16 audio for transcription.
 * @param {Object} opts
 * @param {string} opts.sessionId
 * @param {ArrayBuffer|Uint8Array} opts.pcm
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<{ok: true, text: string} | {ok: false, code: string, message: string}>}
 */
export async function transcribe({ sessionId, pcm }, fetchImpl = globalThis.fetch) {
  try {
    const response = await fetchImpl(TRANSCRIBE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        [SESSION_HEADER]: sessionId,
      },
      body: pcm,
    });
    const payload = await response.json().catch(() => null);
    if (response.ok && payload && typeof payload.text === "string") {
      return { ok: true, text: payload.text };
    }
    const error = (payload && payload.error) || {};
    return {
      ok: false,
      code: error.code || "stt_failed",
      message: error.message || "",
    };
  } catch {
    return { ok: false, code: "stt_failed", message: "" };
  }
}

/**
 * Discard the Session on page hide, using a beacon so it survives unload (Req 12.5).
 * @param {string} sessionId
 * @param {Navigator} [nav]
 */
export function endSession(sessionId, nav = globalThis.navigator) {
  if (nav && typeof nav.sendBeacon === "function") {
    nav.sendBeacon(SESSION_END_URL, sessionId);
  }
}
