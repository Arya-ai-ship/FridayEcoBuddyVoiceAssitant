# Friday — Your next-gen eco-buddy

Friday is a local, single-user proof-of-concept chat and voice assistant for pulling and
analyzing US economic data from FRED. You type or speak requests like "pull inflation" and
Friday fetches the series, shows a preview table with a CSV download, and can run
descriptive statistics, fill missing values, and plot one or more series. It speaks back
short status lines and findings through Amazon Polly.

Friday acts as a **co-economist**: it explains indicators in plain language with everyday
analogies, holds a witty multi-turn conversation, and reasons over the data it already has
before reaching for a tool. Ask "where is the missing value?" and it reads the gap date
from the fetch result — no extra tool call needed.

The Agent runs on the Microsoft Agent Framework harness with only Friday's four tools
(fetch, stats, fill, plot). GPT-5.6 Terra on Amazon Bedrock (Converse API) orchestrates the
conversation, Amazon Transcribe handles speech-to-text, and Amazon Polly handles
text-to-speech. The Backend holds all credentials and proxies every external call; the
browser never sees a credential and never calls an AWS or FRED host directly.

## What Friday can fetch

Friday ships with 11 indicators out of the box. Each entry lists its canonical name,
common aliases, and FRED series ID. GDP and all others are available immediately — just ask
by name or alias.

| Indicator | Aliases | Series | Notes |
|---|---|---|---|
| cpi | consumer price index, cpi-u, headline cpi | CPIAUCSL | monthly |
| inflation | inflation rate, cpi inflation, yoy inflation | CPIAUCSL | year-over-year |
| core cpi | core consumer price index, cpi less food and energy | CPILFESL | monthly |
| unemployment rate | unemployment, jobless rate | UNRATE | monthly |
| nonfarm payrolls | payrolls, nfp, total nonfarm employment | PAYEMS | monthly |
| fed funds rate | federal funds rate, fed funds, effective federal funds rate | FEDFUNDS | monthly |
| 10-year treasury yield | 10y yield, ten-year treasury yield, … | GS10 | monthly |
| housing prices | home prices, house prices, case-shiller | CSUSHPINSA | monthly |
| housing starts | new housing starts, housing units started | HOUST | monthly |
| industrial production | industrial production index, ip index | INDPRO | monthly |
| gdp | gross domestic product, real gdp | GDPC1 | quarterly, real chained 2017$ |

To use a custom indicator map, set `INDICATOR_MAP_PATH` in `.env` to a JSON file following
the same schema as `src/friday/data/default_indicators.json`.

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

## UI

The interface is a modern dark-theme chat layout:

- **Sidebar** — brand mark, voice orb (idle / listening / working / speaking), and a usage
  hint. Collapses to a top bar on narrow screens.
- **Chat window** — alternating message bubbles with avatar initials (B for Boss, F for
  Friday). Preview tables, stats tables, CSV download links, and charts appear inline in
  Friday's message block.
- **Composer** — auto-expanding textarea pinned to the bottom. Send with Enter or the send
  button; Shift+Enter inserts a newline. The mic button activates voice input.

## How the conversation works

Friday uses the full conversation history to decide what to offer next. After a successful
fetch it asks whether to run statistics or plot the data (and notes any missing values,
naming the exact gap dates). After stats, fill, or plot it offers the remaining useful
steps — and never re-offers something it already did on that series earlier in the thread.

Friday reasons from context first. If you ask a question whose answer is already in the
fetch result or dataset inventory (indicator name, date range, missing-value dates, row
count), it answers directly without calling a tool. It calls a tool only to fetch new data,
compute statistics, fill gaps, or draw a chart.

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
value is ever printed.

## `make` targets

| Target | Runs |
|---|---|
| `make run` | `./start.sh` |
| `make fmt` | `ruff format` and `ruff check --fix` |
| `make test` | `pytest` and `node --test tests/js` |
| `make check` | `ruff check`, `ruff format --check`, `pyright`, then `make test` |
| `make verify-aws` | `uv run friday --check` |

## Manual walkthrough

With valid credentials and `./start.sh` running, open the Local URL and try:

1. **Voice**: click the mic and say "pull inflation". The orb pulses while Friday works,
   you hear a spoken "Fetching…" status, then the preview table appears with a CSV link.
   Friday asks whether to run stats or plot the data.
2. **Ask about the data**: type "where is the missing value?" — Friday reads the gap date
   from the fetch result without calling a tool.
3. **Stats**: say or type "run the stats" to get a statistics table and spoken key findings.
4. **Multi-series**: fetch a second indicator (e.g. "pull unemployment"), then say "plot
   both" to get a multi-series chart with a legend.
5. **GDP**: ask "pull real GDP" to fetch the quarterly GDPC1 series.

Voice goes in and out throughout; the orb changes state (idle → listening → working → speaking).

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
- GDP is quarterly; all other default indicators are monthly. Mixed-frequency series plot
  correctly but fill-missing interprets gaps based on observation count, not calendar
  cadence.
- `agent-framework-bedrock` is a pre-release; its live tool-calling behavior is confirmed by
  the `--check` command and the opt-in live suite.
