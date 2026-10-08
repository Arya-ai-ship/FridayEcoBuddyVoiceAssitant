// Pure Assistant_State reducer (Req 2.x, 3.1, 3.2, 3.9, 3.10).
//
// `reduce(state, event, ctx) -> nextState` implements the design's state diagram. It is
// a pure function: no DOM, no timers, no I/O. The DOM layer (app.js) owns side effects
// and reflects the returned state onto the orb. `ctx` carries the two counters the
// transitions depend on, read AFTER the triggering side effect has been applied:
//   - pendingRequests: in-flight Backend requests (chat or transcribe)
//   - queueLength: audio clips queued or playing
//
// States: "idle" | "listening" | "working" | "speaking".
// Events (string `type`):
//   session_start          a new Session begins                        -> idle (2.12)
//   mic_granted            mic click accepted, recording started       (2.9, 3.1)
//   mic_denied             mic permission denied / no device           -> idle (2.11, 3.5)
//   mic_stop               mic click while listening, or 60 s auto     -> working (2.7, 3.2, 3.8)
//   submit                 typed message submitted                     -> working (2.7, 3.10)
//   clip_started           first/any audio clip began playing          -> speaking (2.10)
//   playback_drained       the audio queue emptied                     (2.8)
//   response_no_audio      a Backend response carried no Spoken_Text    (2.8, 2.11)
//   failure                Backend error or STT/TTS failure            -> idle (2.11, 3.6)
//
// Mic clicks while `working` are no-ops (Req 3.9); app.js must not emit mic events then.

/** @typedef {"idle"|"listening"|"working"|"speaking"} AssistantState */

/** The four valid states, exported so app.js and tests share one source. */
export const STATES = Object.freeze(["idle", "listening", "working", "speaking"]);

/** The state a Session begins in (Req 2.12). */
export const INITIAL_STATE = "idle";

/**
 * Compute the next Assistant_State.
 * @param {AssistantState} state current state
 * @param {{type: string}} event the triggering event
 * @param {{pendingRequests?: number, queueLength?: number}} [ctx] post-effect counters
 * @returns {AssistantState} the next state (unchanged for a no-op)
 */
export function reduce(state, event, ctx = {}) {
  const pending = ctx.pendingRequests ?? 0;
  const queued = ctx.queueLength ?? 0;
  const type = event && event.type;

  switch (type) {
    case "session_start":
      // A fresh Session always resets to idle (Req 2.12).
      return "idle";

    case "mic_granted":
      // Starting capture from idle or speaking enters listening (Req 2.9, 3.1).
      // Ignored elsewhere: a grant can only follow a click allowed in those states.
      return state === "idle" || state === "speaking" ? "listening" : state;

    case "mic_denied":
      // Denied mic drops to idle (Req 2.11, 3.5).
      return "idle";

    case "mic_stop":
      // Stopping capture (click or 60 s auto-stop) sends the audio and works (Req 2.7, 3.2, 3.8).
      return state === "listening" ? "working" : state;

    case "submit":
      // A typed submit works from any non-listening/working state and from listening
      // (the recording is discarded by app.js first) (Req 2.7, 3.10).
      return state === "working" ? state : "working";

    case "clip_started":
      // The first clip of the turn starting means Friday is speaking (Req 2.10).
      return "speaking";

    case "playback_drained":
    case "response_no_audio":
      // Idle only once nothing is playing/queued and no request is in flight (Req 2.8).
      if (queued > 0) return "speaking";
      return pending > 0 ? "working" : "idle";

    case "failure":
      // Any failure returns to idle within the DOM frame (Req 2.11, 3.6).
      return "idle";

    default:
      return state;
  }
}
