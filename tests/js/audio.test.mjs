// Tests for the audio clip queue (Req 4.4, 4.8, 4.9) with a fake AudioContext.
// Covers in-order non-overlapping playback, holding a preview until the status clip
// starts, autoplay unlock resuming a suspended context, and a clip arriving while the
// context is still suspended reporting an audio failure without blocking the queue.

import assert from "node:assert/strict";
import { test } from "node:test";

import { AudioPlayer } from "../../src/friday/static/js/audio.js";

// --- Fakes -----------------------------------------------------------------

class FakeSource {
  constructor(ctx) {
    this._ctx = ctx;
    this.buffer = null;
    this.onended = null;
    this.started = false;
  }
  connect() {}
  start() {
    this.started = true;
    this._ctx.playing = this;
  }
  stop() {
    this._ctx.playing = null;
  }
  // Test helper: simulate the clip finishing.
  finish() {
    const cb = this.onended;
    this._ctx.playing = null;
    if (cb) cb();
  }
}

class FakeAnalyser {
  constructor() {
    this.fftSize = 256;
    this.frequencyBinCount = 128;
  }
  connect() {}
  getByteFrequencyData() {}
}

class FakeContext {
  constructor({ state = "running", failDecode = false } = {}) {
    this.state = state;
    this.destination = {};
    this.playing = null;
    this.resumed = 0;
    this._failDecode = failDecode;
    this.sources = [];
  }
  createAnalyser() {
    return new FakeAnalyser();
  }
  createBufferSource() {
    const s = new FakeSource(this);
    this.sources.push(s);
    return s;
  }
  async decodeAudioData() {
    if (this._failDecode) throw new Error("bad audio");
    return { duration: 1 };
  }
  async resume() {
    this.resumed += 1;
    this.state = "running";
  }
}

const clip = (n) => new Uint8Array([n]).buffer;

// --- Tests -----------------------------------------------------------------

test("clips play one at a time, in order", async () => {
  const ctx = new FakeContext();
  const started = [];
  const player = new AudioPlayer(() => ctx, { onStarted: () => started.push(ctx.playing) });
  player.enqueue(clip(1));
  await Promise.resolve();
  await Promise.resolve();
  // First clip is playing; a second does not start until the first ends.
  player.enqueue(clip(2));
  await Promise.resolve();
  assert.equal(ctx.sources.filter((s) => s.started).length, 1);
  ctx.playing.finish();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(ctx.sources.filter((s) => s.started).length, 2);
  assert.equal(started.length, 2);
});

test("onDrained fires when the queue empties", async () => {
  const ctx = new FakeContext();
  let drained = 0;
  const player = new AudioPlayer(() => ctx, { onDrained: () => (drained += 1) });
  player.enqueue(clip(1));
  await Promise.resolve();
  await Promise.resolve();
  ctx.playing.finish();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(drained, 1);
});

test("a held preview is released only when its status clip starts (Req 4.4)", async () => {
  const ctx = new FakeContext();
  const released = [];
  const player = new AudioPlayer(() => ctx, { onRelease: (p) => released.push(p) });
  player.holdUntilClip({ kind: "preview" });
  assert.equal(released.length, 0); // held before any clip plays
  player.enqueue(clip(1));
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(released, [{ kind: "preview" }]);
});

test("unlock resumes a suspended context (autoplay)", async () => {
  const ctx = new FakeContext({ state: "suspended" });
  const player = new AudioPlayer(() => ctx);
  await player.unlock();
  assert.equal(ctx.resumed, 1);
  assert.equal(ctx.state, "running");
});

test("a clip while suspended reports failure, releases the preview, keeps going (Req 4.8)", async () => {
  const ctx = new FakeContext({ state: "suspended" });
  let errors = 0;
  const released = [];
  const player = new AudioPlayer(() => ctx, {
    onAudioError: () => (errors += 1),
    onRelease: (p) => released.push(p),
    onDrained: () => released.push("drained"),
  });
  player.holdUntilClip({ kind: "preview" });
  player.enqueue(clip(1));
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(errors, 1);
  assert.deepEqual(released, [{ kind: "preview" }, "drained"]);
});

test("a clip that fails to decode is skipped and reported (Req 4.8)", async () => {
  const ctx = new FakeContext({ failDecode: true });
  let errors = 0;
  const player = new AudioPlayer(() => ctx, { onAudioError: () => (errors += 1) });
  player.enqueue(clip(1));
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(errors, 1);
});

test("stop clears the queue and current clip", async () => {
  const ctx = new FakeContext();
  const player = new AudioPlayer(() => ctx);
  player.enqueue(clip(1));
  player.enqueue(clip(2));
  await Promise.resolve();
  await Promise.resolve();
  player.stop();
  assert.equal(ctx.playing, null);
});
