// Microphone capture: getUserMedia mono -> analyser -> PCM worklet -> in-memory Int16
// buffer, with voice-assistant style end-of-speech detection, an analyser level for the
// orb, and discard-on-typed-submit (Req 2.3, 2.5, 3.1, 3.2, 3.5, 3.8, 3.10).
//
// `MicRecorder.start()` must be called directly from the click handler: it creates and
// resumes the AudioContext and calls getUserMedia before its first `await`, so Safari
// and Chrome still see the user gesture (otherwise the context stays suspended and the
// worklet records silence). The graph is source -> analyser -> worklet -> muted gain ->
// destination; keeping every node on the path to the destination guarantees the audio
// thread actually pulls the mic samples (WebKit does not process unconnected nodes).
//
// Recording stops automatically, like a modern voice assistant:
//   - MIC_SILENCE_S of quiet after speech was heard (end of utterance),
//   - MIC_NO_SPEECH_S with no speech at all,
//   - the MIC_MAX_S hard cap.
// `stop()` returns the concatenated PCM16 bytes; `discard()` throws them away.

import { MIC_MAX_S, MIC_NO_SPEECH_S, MIC_SILENCE_S, STT_SAMPLE_RATE_HZ } from "./constants.js";

// audioWorklet.addModule resolves relative URLs against the document, not this module,
// so build an absolute URL from this module's own location (/static/js/).
const WORKLET_URL = new URL("./pcm-worklet.js", import.meta.url).href;
const PROCESSOR = "pcm-downsampler";

// Normalized peak levels (0-1). Speech must rise above SPEECH_LEVEL to count as started;
// the utterance ends after MIC_SILENCE_S below SILENCE_LEVEL. The gap between the two
// keeps breath noise and room hum from holding the recording open.
const SPEECH_LEVEL = 0.06;
const SILENCE_LEVEL = 0.03;

export class MicRecorder {
  /**
   * @param {Object} [opts]
   * @param {() => AudioContext} [opts.createContext]
   * @param {MediaDevices} [opts.mediaDevices]
   * @param {(level: number) => void} [opts.onLevel]
   * @param {(reason: "silence"|"no_speech"|"max") => void} [opts.onAutoStop]
   */
  constructor(opts = {}) {
    this._createContext =
      opts.createContext ?? (() => new (globalThis.AudioContext || globalThis.webkitAudioContext)());
    this._mediaDevices = opts.mediaDevices ?? globalThis.navigator?.mediaDevices;
    this._onLevel = opts.onLevel;
    this._onAutoStop = opts.onAutoStop;
    this._ctx = null;
    this._stream = null;
    this._source = null;
    this._node = null;
    this._sink = null;
    this._analyser = null;
    this._raf = null;
    this._capTimer = null;
    this._startedAt = 0;
    this._heardSpeech = false;
    this._silenceSince = null;
    /** @type {Int16Array[]} */
    this._chunks = [];
    this._recording = false;
  }

  get recording() {
    return this._recording;
  }

  /**
   * Request the mic and start recording. Rejects if permission is denied, no device is
   * available, or audio setup fails (Req 3.5); everything is released on failure.
   * @returns {Promise<void>}
   */
  async start() {
    if (!this._mediaDevices || typeof this._mediaDevices.getUserMedia !== "function") {
      throw new Error("no_microphone");
    }
    // Synchronous, inside the user gesture: create + resume the context, ask for the mic.
    const ctx = this._createContext();
    this._ctx = ctx;
    const resumed = ctx.state === "suspended" ? ctx.resume() : Promise.resolve();
    const media = this._mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    try {
      this._stream = await media;
      await resumed;
      await ctx.audioWorklet.addModule(WORKLET_URL);
      if (ctx.state === "suspended") await ctx.resume();
      this._wire(ctx);
    } catch (err) {
      this._teardown();
      throw err;
    }
    this._chunks = [];
    this._startedAt = globalThis.performance.now();
    this._heardSpeech = false;
    this._silenceSince = null;
    this._recording = true;
    this._pollLevel();
    // rAF pauses in background tabs; the hard cap must still fire there.
    this._capTimer = globalThis.setTimeout(() => {
      if (this._recording && this._onAutoStop) this._onAutoStop("max");
    }, MIC_MAX_S * 1000);
  }

  /** Build source -> analyser -> worklet -> muted gain -> destination. */
  _wire(ctx) {
    this._source = ctx.createMediaStreamSource(this._stream);
    this._analyser = ctx.createAnalyser();
    this._analyser.fftSize = 1024;
    this._node = new AudioWorkletNode(ctx, PROCESSOR, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this._node.port.onmessage = (event) => {
      if (this._recording) this._chunks.push(new Int16Array(event.data));
    };
    this._sink = ctx.createGain();
    this._sink.gain.value = 0; // keep the graph pulled without echoing the mic
    this._source.connect(this._analyser);
    this._analyser.connect(this._node);
    this._node.connect(this._sink);
    this._sink.connect(ctx.destination);
  }

  /**
   * Stop recording and return the mono PCM16 little-endian bytes captured so far.
   * @returns {Uint8Array}
   */
  stop() {
    const pcm = this._collect();
    this._teardown();
    return pcm;
  }

  /** Discard the recording and release the mic without returning audio (Req 3.10). */
  discard() {
    this._chunks = [];
    this._teardown();
  }

  /** Sample rate of the produced PCM (always 16 kHz for Transcribe). */
  get sampleRate() {
    return STT_SAMPLE_RATE_HZ;
  }

  _collect() {
    const total = this._chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Int16Array(total);
    let offset = 0;
    for (const chunk of this._chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return new Uint8Array(merged.buffer);
  }

  _pollLevel() {
    const data = new Uint8Array(this._analyser.fftSize);
    const tick = () => {
      if (!this._recording) return;
      this._analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i += 1) {
        peak = Math.max(peak, Math.abs(data[i] - 128));
      }
      const level = peak / 128;
      if (this._onLevel) this._onLevel(level);
      const reason = this._autoStopReason(level, globalThis.performance.now());
      if (reason) {
        if (this._onAutoStop) this._onAutoStop(reason);
        return;
      }
      this._raf = globalThis.requestAnimationFrame(tick);
    };
    tick();
  }

  /**
   * Decide whether to end the recording now (Req 3.8). Returns the reason or `null`.
   * @param {number} level current normalized peak level
   * @param {number} now performance.now() in ms
   */
  _autoStopReason(level, now) {
    if (now - this._startedAt >= MIC_MAX_S * 1000) return "max";
    if (level >= SPEECH_LEVEL) {
      this._heardSpeech = true;
      this._silenceSince = null;
      return null;
    }
    if (!this._heardSpeech) {
      return now - this._startedAt >= MIC_NO_SPEECH_S * 1000 ? "no_speech" : null;
    }
    if (level >= SILENCE_LEVEL) {
      this._silenceSince = null;
      return null;
    }
    if (this._silenceSince === null) this._silenceSince = now;
    return now - this._silenceSince >= MIC_SILENCE_S * 1000 ? "silence" : null;
  }

  _teardown() {
    this._recording = false;
    this._heardSpeech = false;
    this._silenceSince = null;
    if (this._raf !== null && globalThis.cancelAnimationFrame) {
      globalThis.cancelAnimationFrame(this._raf);
    }
    this._raf = null;
    if (this._capTimer !== null) {
      globalThis.clearTimeout(this._capTimer);
      this._capTimer = null;
    }
    if (this._node) this._node.port.onmessage = null;
    for (const node of [this._source, this._analyser, this._node, this._sink]) {
      try {
        node?.disconnect();
      } catch {
        // Already disconnected.
      }
    }
    this._source = null;
    this._analyser = null;
    this._node = null;
    this._sink = null;
    if (this._stream) {
      for (const track of this._stream.getTracks()) track.stop();
      this._stream = null;
    }
    if (this._ctx && typeof this._ctx.close === "function") {
      this._ctx.close().catch(() => {});
    }
    this._ctx = null;
  }
}
