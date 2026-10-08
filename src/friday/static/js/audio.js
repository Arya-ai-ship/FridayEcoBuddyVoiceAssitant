// Audio playback: one AudioContext, a FIFO clip queue, one clip at a time through an
// AnalyserNode, with the dataset_preview hold and autoplay-unlock rules
// (Req 2.5, 2.10, 4.4, 4.8, 4.9).
//
// The context is created lazily and only resumed inside a user gesture via `unlock()`
// (Req from design: autoplay). Clips are decoded with `decodeAudioData` and played in
// arrival order, never overlapping. `onStarted(level)` fires when a clip begins (drives
// `speaking`), `onDrained` fires when the queue empties, and `onLevel(0..1)` is polled
// while a clip plays so the orb can glow with the playback level (Req 2.5).
//
// Preview hold (Req 4.4): a payload enqueued with `holdUntilClip()` is released only when
// the next clip actually starts playing, or at once if that clip failed to play, so the
// spoken "Fetching…" status is heard before its Preview_Table appears.
//
// Audio failure (Req 4.8): if a clip cannot decode or the context stays suspended, the
// clip is dropped, `onAudioError` fires (the UI shows "Audio unavailable" and keeps the
// text), and the queue keeps moving. The AudioContext factory is injected for tests.

/**
 * @typedef {Object} AudioCallbacks
 * @property {() => void} [onStarted]      a clip began playing (Req 2.10)
 * @property {() => void} [onDrained]      the queue emptied (Req 2.8)
 * @property {(level: number) => void} [onLevel]  playback level 0..1 (Req 2.5)
 * @property {() => void} [onAudioError]   a clip could not be played (Req 4.8)
 * @property {(payload: object) => void} [onRelease]  a held preview was released (Req 4.4)
 */

export class AudioPlayer {
  /**
   * @param {() => (AudioContext|null)} createContext factory for a (possibly null) context
   * @param {AudioCallbacks} [callbacks]
   */
  constructor(createContext, callbacks = {}) {
    this._createContext = createContext;
    this._cb = callbacks;
    /** @type {AudioContext|null} */
    this._ctx = null;
    this._analyser = null;
    /** @type {Array<{data: ArrayBuffer}>} */
    this._queue = [];
    /** @type {Array<{payload: object}>} */
    this._held = [];
    this._playing = false;
    this._current = null; // the active AudioBufferSourceNode
    this._levelTimer = null;
  }

  /** Create the context if needed and resume it inside a user gesture (autoplay unlock). */
  async unlock() {
    const ctx = this._ensureContext();
    if (ctx && ctx.state === "suspended" && typeof ctx.resume === "function") {
      await ctx.resume();
    }
  }

  /** Enqueue an MP3 clip (ArrayBuffer) for playback in arrival order (Req 4.9). */
  enqueue(data) {
    this._queue.push({ data });
    if (!this._playing) this._advance();
  }

  /** Hold a display payload until the next clip starts, or release now if none will play. */
  holdUntilClip(payload) {
    this._held.push({ payload });
  }

  /** Stop all playback and clear the queue (mic interrupt, Req 3.1). */
  stop() {
    this._queue = [];
    this._releaseHeld();
    this._stopLevelPolling();
    if (this._current) {
      try {
        this._current.onended = null;
        this._current.stop();
      } catch {
        // Already stopped.
      }
      this._current = null;
    }
    this._playing = false;
  }

  _ensureContext() {
    if (this._ctx === null) {
      this._ctx = this._createContext();
      if (this._ctx && typeof this._ctx.createAnalyser === "function") {
        this._analyser = this._ctx.createAnalyser();
        this._analyser.fftSize = 256;
        this._analyser.connect(this._ctx.destination);
      }
    }
    return this._ctx;
  }

  _releaseHeld() {
    const held = this._held;
    this._held = [];
    for (const item of held) {
      if (this._cb.onRelease) this._cb.onRelease(item.payload);
    }
  }

  _fail() {
    // The clip cannot play: release any held preview, report, and keep the queue moving.
    this._releaseHeld();
    if (this._cb.onAudioError) this._cb.onAudioError();
  }

  async _advance() {
    const next = this._queue.shift();
    if (!next) {
      this._playing = false;
      if (this._cb.onDrained) this._cb.onDrained();
      return;
    }
    this._playing = true;
    const ctx = this._ensureContext();
    if (!ctx || ctx.state === "suspended") {
      // No usable context (autoplay still blocked): drop this clip, keep going (Req 4.8).
      this._fail();
      this._advance();
      return;
    }
    let buffer;
    try {
      buffer = await ctx.decodeAudioData(next.data);
    } catch {
      this._fail();
      this._advance();
      return;
    }
    this._play(ctx, buffer);
  }

  _play(ctx, buffer) {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this._analyser ?? ctx.destination);
    this._current = source;
    source.onended = () => {
      if (this._current !== source) return; // superseded by stop()
      this._current = null;
      this._stopLevelPolling();
      this._advance();
    };
    source.start();
    // The clip has started: release the held preview and signal `speaking` (Req 4.4, 2.10).
    this._releaseHeld();
    if (this._cb.onStarted) this._cb.onStarted();
    this._startLevelPolling();
  }

  _startLevelPolling() {
    if (!this._analyser || !this._cb.onLevel) return;
    const data = new Uint8Array(this._analyser.frequencyBinCount);
    const poll = () => {
      if (!this._current) return;
      this._analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 1) sum += data[i];
      this._cb.onLevel(data.length ? sum / data.length / 255 : 0);
      this._levelTimer = globalThis.requestAnimationFrame
        ? globalThis.requestAnimationFrame(poll)
        : null;
    };
    poll();
  }

  _stopLevelPolling() {
    if (this._levelTimer !== null && globalThis.cancelAnimationFrame) {
      globalThis.cancelAnimationFrame(this._levelTimer);
    }
    this._levelTimer = null;
  }
}
