// Browser-facing limits. Mirrors friday/constants.py (same names, same units);
// tests/test_parity.py checks that every value here matches. Keep each entry a
// plain numeric literal so the parity test can read it.

/** Maximum chat message length after trimming (Req 1.2, 1.7). */
export const MAX_TEXT_CHARS = 2000;

/** Seconds to wait for the first stream event and between events (Req 1.8). */
export const CLIENT_TIMEOUT_S = 60;

/** Recording auto-stops after this many seconds (Req 3.8). */
export const MIC_MAX_S = 60;

/** Recording auto-stops after this many seconds of silence following speech (Req 3.8). */
export const MIC_SILENCE_S = 2;

/** Recording auto-stops when no speech is heard at all within this many seconds. */
export const MIC_NO_SPEECH_S = 8;

/** PCM16 mono sample rate the recorder produces for Transcribe (Req 3.3). */
export const STT_SAMPLE_RATE_HZ = 16000;
