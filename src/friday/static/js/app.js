// Application wiring: connects input, mic, API, audio, chat, and the state reducer, and
// reflects the Assistant_State onto the Voice_Orb (Req 1.x, 2.x, 3.x, 4.x).
//
// This module owns the side effects; the pure modules it drives are tested on their own.
// A per-page session UUID is sent on every request. The audio context is unlocked inside
// the first Enter submit and the first Mic_Button click so Friday's later async audio can
// play in Chrome and Safari.

import { endSession, streamChat, transcribe } from "./api.js";
import { AudioPlayer } from "./audio.js";
import { ChatView } from "./chat.js";
import { validateSubmission } from "./input.js";
import { MicRecorder } from "./mic.js";
import { INITIAL_STATE, reduce } from "./state.js";

const MIC_REQUIRED =
  "Boss, microphone access is required for voice input. You can still type your request.";
const STT_RETRY = "Sorry Boss, I didn't catch that — please repeat or type it.";
const AWS_CREDENTIALS =
  "Boss, the AWS credentials have expired or lack access. " +
  "Please refresh the AWS values in .env and restart the Backend.";
const BACKEND_FAILURE = "Sorry Boss, something went wrong reaching the Backend. Please try again.";
const TOO_LONG = "That message is too long, Boss — please keep it under 2000 characters.";
const NO_SPEECH = "I didn't hear anything. Tap the mic and try again, or type your request.";
const LISTENING = "Listening… I'll stop when you pause, or tap the mic to finish.";
const MIC_LABEL_START = "Start voice input";
const MIC_LABEL_STOP = "Stop voice input";

function makeAudioContext() {
  const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
  return Ctor ? new Ctor() : null;
}

class App {
  constructor(doc = globalThis.document) {
    this._doc = doc;
    this._sessionId = globalThis.crypto.randomUUID();
    this._state = INITIAL_STATE;
    this._pending = 0;
    this._level = 0;
    this._unlocked = false;
    this._micStarting = false;

    this._log = doc.getElementById("chat-log");
    this._orb = doc.getElementById("voice-orb");
    this._input = doc.getElementById("chat-input");
    this._form = doc.getElementById("composer");
    this._sendButton = doc.getElementById("send-button");
    this._micButton = doc.getElementById("mic-button");
    this._notice = doc.getElementById("notice");

    this._chat = new ChatView(this._log);
    this._audio = new AudioPlayer(makeAudioContext, {
      onStarted: () => this._dispatch({ type: "clip_started" }),
      onDrained: () => this._dispatch({ type: "playback_drained" }),
      onLevel: (level) => this._setLevel(level),
      onAudioError: () => this._showNotice("Audio unavailable — showing the text instead."),
    });
    this._mic = new MicRecorder({
      onLevel: (level) => this._setLevel(level),
      onAutoStop: (reason) => this._onAutoStop(reason),
    });
  }

  start() {
    this._input.addEventListener("keydown", (e) => this._onKeydown(e));
    this._form.addEventListener("submit", (e) => {
      e.preventDefault(); // never navigate; the Send button and Enter share _submitTyped
      this._submitTyped();
    });
    this._micButton.addEventListener("click", () => this._onMicClick());
    this._doc.defaultView?.addEventListener("pagehide", () => endSession(this._sessionId));
    this._render();
  }

  // --- State ---------------------------------------------------------------

  get _queueLength() {
    return this._state === "speaking" ? 1 : 0;
  }

  _dispatch(event) {
    this._state = reduce(this._state, event, {
      pendingRequests: this._pending,
      queueLength: this._queueLength,
    });
    this._render();
  }

  _render() {
    const listening = this._state === "listening";
    this._orb.dataset.state = this._state;
    this._orb.setAttribute("aria-label", `Friday is ${this._state}`);
    this._micButton.setAttribute("aria-pressed", String(listening));
    this._micButton.setAttribute("aria-label", listening ? MIC_LABEL_STOP : MIC_LABEL_START);
    this._micButton.disabled = this._state === "working";
    this._sendButton.disabled = this._pending > 0;
  }

  _setLevel(level) {
    this._level = Math.max(0, Math.min(1, level));
    globalThis.requestAnimationFrame(() => {
      this._orb.style.setProperty("--level", String(this._level));
    });
  }

  _showNotice(text) {
    this._notice.textContent = text;
  }

  _clearNotice() {
    this._notice.textContent = "";
  }

  // --- Typed input ---------------------------------------------------------

  _onKeydown(event) {
    // Enter sends, Shift+Enter adds a line. Ignore Enter while an IME is composing
    // (Safari reports keyCode 229 for those), so accepting a candidate never sends.
    if (event.key !== "Enter" || event.shiftKey) return;
    if (event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    this._submitTyped();
  }

  _submitTyped() {
    const waiting = this._pending > 0;
    const result = validateSubmission(this._input.value, waiting);
    if (result.action === "ignore") return;
    if (result.action === "too_long") {
      this._showNotice(TOO_LONG);
      return;
    }
    this._unlock();
    if (this._state === "listening") this._mic.discard();
    this._input.value = "";
    this._clearNotice();
    this._chat.addUserMessage(result.text);
    this._dispatch({ type: "submit" });
    void this._sendChat(result.text);
  }

  // --- Microphone ----------------------------------------------------------

  async _onMicClick() {
    if (this._state === "working") return; // no-op while working (Req 3.9)
    this._unlock();
    if (this._state === "listening") {
      await this._stopRecording();
      return;
    }
    if (this._micStarting) return; // permission prompt / setup already in progress
    this._audio.stop(); // stop any playback before listening (Req 3.1)
    this._micStarting = true;
    try {
      // Called synchronously from the click so Safari keeps the user gesture.
      await this._mic.start();
      this._showNotice(LISTENING);
      this._dispatch({ type: "mic_granted" });
    } catch (err) {
      globalThis.console?.warn("Friday: microphone start failed", err);
      this._showNotice(MIC_REQUIRED);
      this._dispatch({ type: "mic_denied" });
    } finally {
      this._micStarting = false;
    }
  }

  _onAutoStop(reason) {
    if (reason === "no_speech") {
      // Nothing was said: don't send silence to Transcribe.
      this._mic.discard();
      this._showNotice(NO_SPEECH);
      this._dispatch({ type: "failure" });
      return;
    }
    void this._stopRecording();
  }

  async _stopRecording() {
    if (!this._mic.recording) return;
    this._clearNotice();
    const pcm = this._mic.stop();
    this._dispatch({ type: "mic_stop" });
    this._pending += 1;
    try {
      const result = await transcribe({ sessionId: this._sessionId, pcm });
      if (!result.ok) {
        this._showNotice(result.code === "aws_credentials" ? AWS_CREDENTIALS : STT_RETRY);
        this._pending -= 1;
        this._dispatch({ type: "failure" });
        return;
      }
      this._chat.addUserMessage(result.text);
      await this._runChat(result.text);
    } catch (err) {
      globalThis.console?.warn("Friday: transcription failed", err);
      this._showNotice(STT_RETRY);
      this._pending -= 1;
      this._dispatch({ type: "failure" });
    }
  }

  // --- Chat stream ---------------------------------------------------------

  async _sendChat(text) {
    this._pending += 1;
    this._render(); // disable Send while the request is in flight
    await this._runChat(text);
  }

  async _runChat(text) {
    let failed = false;
    await streamChat({
      sessionId: this._sessionId,
      text,
      onEvent: (event) => this._onEvent(event),
      onFailure: () => {
        failed = true;
        this._showNotice(BACKEND_FAILURE);
      },
    });
    this._pending -= 1;
    if (failed) {
      this._dispatch({ type: "failure" });
    } else if (this._queueLength === 0) {
      this._dispatch({ type: "response_no_audio" });
    }
  }

  _onEvent(event) {
    this._chat.handleEvent(event);
    const audio = event.audio && event.audio.b64;
    if (audio) {
      this._audio.enqueue(base64ToBuffer(audio));
    } else if (event.type === "dataset_preview") {
      // Hold the preview until its tool_start clip plays (Req 4.4). chat.js already
      // rendered it; the hold here keeps the orb/audio ordering consistent.
      this._audio.holdUntilClip(event);
    }
    if (event.type === "final" && event.outcome === "aws_credentials") {
      this._showNotice(AWS_CREDENTIALS);
    }
  }

  _unlock() {
    if (this._unlocked) return;
    this._unlocked = true;
    void this._audio.unlock();
  }
}

function base64ToBuffer(b64) {
  const binary = globalThis.atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

const app = new App();
app.start();

export { App };
