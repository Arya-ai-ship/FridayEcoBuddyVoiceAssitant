// AudioWorklet processor: downsample the mic's float32 mono input to 16 kHz Int16 PCM
// and post each block to the main thread (Req 3.1, 3.3).
//
// The worklet runs on the audio thread. It carries a fractional read position so the
// decimation ratio (contextRate / 16000) stays accurate across 128-sample render quanta,
// and converts each picked sample to little-endian Int16. No buffering of the whole take
// happens here; mic.js accumulates the posted chunks in memory.

const TARGET_RATE = 16000;

class PcmDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ratio = sampleRate / TARGET_RATE; // sampleRate is the global context rate
    this._pos = 0; // fractional read index into the running input stream
  }

  /**
   * @param {Float32Array[][]} inputs
   * @returns {boolean} keep the processor alive
   */
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    const out = [];
    // Pick samples at the fractional ratio within this 128-sample quantum.
    while (this._pos < channel.length) {
      const sample = channel[Math.floor(this._pos)];
      const clamped = Math.max(-1, Math.min(1, sample));
      out.push(clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff);
      this._pos += this._ratio;
    }
    this._pos -= channel.length; // carry the fraction into the next quantum

    if (out.length > 0) {
      const pcm = new Int16Array(out.length);
      for (let i = 0; i < out.length; i += 1) pcm[i] = out[i] | 0;
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-downsampler", PcmDownsampler);
