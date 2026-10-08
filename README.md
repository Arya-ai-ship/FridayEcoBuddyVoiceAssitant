# Friday — Your next-gen eco-buddy

Friday is a local, single-user proof-of-concept chat and voice assistant for pulling and
analyzing US economic data from FRED. You type or speak requests like "please pull
inflation" and Friday fetches the series, shows a preview table with a CSV download, and
can run descriptive statistics, fill missing values, and plot one or more series. It speaks
back short status lines and findings.

The Agent runs on the Microsoft Agent Framework harness with only Friday's four tools
(fetch, stats, fill, plot). GPT-5.6 Terra on Amazon Bedrock (Converse API) orchestrates the
conversation, Amazon Transcribe handles speech-to-text, and Amazon Polly handles
text-to-speech. The Backend holds all credentials and proxies every external call; the
browser never sees a credential and never calls an AWS or FRED host directly.

## Prerequisites

- **Python ≥ 3.12** (developed on 3.14).
- **[uv](https://docs.astral.sh/uv/)** for dependency management and running.
- **Node.js** (only for the JavaScript tests, `node --test`).
- **AWS credentials** with access to Bedrock, Transcribe, and Polly in one region. The
  `--check` command verifies each. The IAM permissions it needs are:
  - `sts:GetCallerIdentity`
  - `bedrock:InvokeModel` / `bedrock:InvokeModelWithResponseStream` on the
    `us.openai.gpt-5.6-terra` cross-Region inference profile **and** on the destination
    foundation models the profile routes to
  - `polly:SynthesizeSpeech`
  - `transcribe:StartStreamTranscription`
- **A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)** (free).

## Setup

Copy the example environment file and fill in your values:

```sh
cp .env.example .env
```

`.env` is git-ignored and is read only at startup. Required variables: `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `FRED_API_KEY`, `BEDROCK_MODEL_ID`. For temporary
(sandbox) credentials, also set `AWS_SESSION_TOKEN`. The suggested model is
`us.openai.gpt-5.6-terra` in `us-east-2`. Optional settings (voice, engine, language,
Indicator_Map path, port) have defaults shown in `.env.example`.

## Start command

```sh
./start.sh
```

`start.sh` runs `uv sync --frozen` and then launches Friday. The first run installs
dependencies.

## Local URL

Friday prints its address at startup, by default <http://127.0.0.1:8000>. Open it in a
browser. The Backend binds to `127.0.0.1` only.

## Refreshing expired AWS credentials

Sandbox credentials expire. When they do, requests show a message asking you to refresh
them. Update the `AWS_*` values (including `AWS_SESSION_TOKEN` for temporary credentials) in
`.env`, stop the Backend with Ctrl+C, and run `./start.sh` again. Friday never reads or
edits `.env` for you.

## Verifying credentials (`--check`)

Before a full session, verify every dependency:

```sh
uv run friday --check   # or: make verify-aws
```

It prints an `OK`/`FAIL` line per check (AWS identity, Bedrock, Polly, Transcribe, FRED) and
exits 0 only when all pass. Failure reasons are classified and redacted — no credential
value is ever printed. A model ID without an inference-profile prefix fails the Bedrock
check with the classified Bedrock error.

## `make` targets

| Target | Runs |
|---|---|
| `make run` | `./start.sh` |
| `make fmt` | `ruff format` and `ruff check --fix` |
| `make test` | `pytest` and `node --test tests/js` |
| `make check` | `ruff check`, `ruff format --check`, `pyright`, then `make test` |
| `make verify-aws` | `uv run friday --check` |

## Manual walkthrough

With valid credentials and `./start.sh` running, open the Local_URL and try:

1. Click the mic and say "pull inflation". You should hear a spoken "Fetching…" status
   before the preview table appears, then see the preview with a CSV download link and hear
   a spoken next-step offer.
2. Say or type "run the stats" to get a statistics table and spoken findings.
3. Fetch a second indicator, then say "plot both" to get a multi-series chart with a legend.

Voice goes in and out throughout, and the orb changes state (idle, listening, working,
speaking).

## Tests

All tests run offline with fakes, mocks, and stubs — no live AWS calls:

```sh
make check
```

An opt-in live suite exercises the real services and runs only when enabled:

```sh
FRIDAY_LIVE_TESTS=1 uv run pytest tests/live
```

## Notes

- Sessions are in-memory and are lost when the server stops.
- The full session history is sent to the model each turn (acceptable for a POC; tool
  results are compact summaries).
- `agent-framework-bedrock` is a pre-release; its live tool-calling behavior is confirmed by
  the `--check` command and the opt-in live suite.
