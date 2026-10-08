"""Limits and timeouts for the Backend (single source of truth).

Browser-facing values are mirrored in ``static/js/constants.js`` and checked by
``tests/test_parity.py``. Never hardcode these numbers anywhere else.
"""

from typing import Final

# --- Chat input ------------------------------------------------------------
MAX_TEXT_CHARS: Final = 2000
"""Maximum chat message length after trimming (Req 1.2, 1.7)."""

MAX_CHAT_BODY_BYTES: Final = 16 * 1024
"""Maximum ``POST /api/chat`` body size (16 KiB)."""

# --- Agent loop ------------------------------------------------------------
MAX_TOOL_CALLS: Final = 8
"""Tool calls processed per user turn; the 9th ends the turn (Req 5.9)."""

LLM_TIMEOUT_S: Final = 60.0
"""Per-request Bedrock timeout in seconds (Req 5.7)."""

LLM_CONNECT_TIMEOUT_S: Final = 5.0
"""Connect timeout for Bedrock clients in seconds."""

MAX_ECHO_CHARS: Final = 40
"""Longest argument value quoted back to the LLM in a tool validation error."""

AWS_MAX_ATTEMPTS: Final = 1
"""Total attempts per AWS SDK call (botocore ``total_max_attempts``); 1 means no retries."""

# --- Narration -------------------------------------------------------------
MAX_SPOKEN_WORDS: Final = 60
"""Maximum Spoken_Text words per Friday response (Req 4.10)."""

MAX_DONE_WORDS: Final = 10
"""Maximum words in a tool completion status line (Req 4.5)."""

MAX_SPOKEN_SENTENCES: Final = 2
"""Sentences the LLM is told to write in its ``<spoken>`` part (design: System prompt)."""

MAX_STATUS_WORDS: Final = 34
"""Status-line words spoken per turn. Later lines are shown but not spoken, so status lines
plus the longest offer (26 words) stay within ``MAX_SPOKEN_WORDS`` (Req 4.10)."""

# --- Speech ----------------------------------------------------------------
TTS_TIMEOUT_S: Final = 15.0
"""Polly synthesis timeout in seconds (Req 4.8)."""

TTS_CONNECT_TIMEOUT_S: Final = 5.0
"""Connect timeout for the Polly client in seconds (inside the TTS budget)."""

STT_TIMEOUT_S: Final = 30.0
"""Transcribe timeout in seconds (Req 3.3, 3.6)."""

STT_SAMPLE_RATE_HZ: Final = 16000
"""PCM16 mono sample rate sent to Transcribe."""

STT_CHUNK_MS: Final = 100
"""Audio chunk length streamed to Transcribe, in milliseconds."""

STT_PACE_FACTOR: Final = 4.0
"""Streaming speed relative to real time (60 s of audio takes about 15 s)."""

MIC_MAX_S: Final = 60
"""Recording auto-stops after this many seconds (Req 3.8)."""

MIC_SILENCE_S: Final = 2
"""Recording auto-stops after this many seconds of silence following speech (Req 3.8)."""

MIC_NO_SPEECH_S: Final = 8
"""Recording auto-stops when no speech is heard at all within this many seconds."""

MAX_AUDIO_BYTES: Final = 2_500_000
"""Maximum ``POST /api/transcribe`` body size (about 78 s of 16 kHz PCM16)."""

# --- Browser ---------------------------------------------------------------
CLIENT_TIMEOUT_S: Final = 60
"""Browser timeout to the first stream event and between events (Req 1.8)."""

# --- Data ------------------------------------------------------------------
FRED_TIMEOUT_S: Final = 15.0
"""FRED API request timeout in seconds (Req 6.10)."""

MAX_INDICATOR_CHARS: Final = 100
"""Maximum length of the ``indicator`` tool argument."""

YOY_LAG_ROWS: Final = 12
"""Rows dropped by the YoY_Transformation for monthly data (Req 6.6, 6.7)."""

PREVIEW_ROWS: Final = 10
"""Rows shown in a Preview_Table (Req 7.1)."""

PREVIEW_DECIMALS: Final = 4
"""Maximum decimals for Preview_Table display values."""

STATS_DECIMALS: Final = 2
"""Decimals for non-count Descriptive_Statistics display values (Req 9.4)."""

MAX_PLOT_SERIES: Final = 5
"""Maximum Datasets per chart (Req 11.1, 11.8)."""

MAX_TITLE_CHARS: Final = 100
"""Maximum chart title length (Req 11.4, 11.8)."""

# --- Sessions, redaction, server -------------------------------------------
SESSION_MAX_IDLE_S: Final = 7200.0
"""Idle Sessions are evicted after this many seconds."""

REDACT_MIN_FRAGMENT: Final = 8
"""Shortest Credential fragment the Redactor masks (Req 5.13, 12.4)."""

DEFAULT_PORT: Final = 8000
"""Backend port when ``FRIDAY_PORT`` is unset or blank (Req 13.3)."""

MIN_PORT: Final = 1
MAX_PORT: Final = 65535
"""Valid ``FRIDAY_PORT`` range (Req 13.5)."""

BIND_HOST: Final = "127.0.0.1"
"""The only address the Backend binds to (Req 13.2)."""

GRACEFUL_SHUTDOWN_S: Final = 3
"""Uvicorn graceful shutdown timeout; keeps exit well under 5 s (Req 13.8)."""
