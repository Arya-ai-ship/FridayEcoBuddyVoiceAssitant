# Implementation Plan: Friday Voice Data Assistant

## Overview

Friday is built in Python 3.12+ (FastAPI, numpy, matplotlib, managed with `uv`) with a build-free ES-module frontend, following the design and `.kiro/steering/engineering-standards.md`.

The order keeps live AWS out of the critical path, because the sandbox token is currently expired:

1. Scaffolding and engineering guard-rails.
2. Pure domain modules (config, redaction, indicators, datasets, fetch math, stats, fill, plot), each with Hypothesis property tests.
3. A Microsoft Agent Framework (MAF) spike, then ports, session, tools, narration, and the Agent facade on the MAF harness, tested with a scripted fake MAF chat client and fake TTS/STT/FRED clients.
4. Adapters (FRED, the `BedrockChatClient` factory, Polly, Transcribe), tested only with mocks and stubs.
5. Wiring, HTTP server, CLI, and `--check`.
6. Frontend.
7. Live verification once the user refreshes `.env`.

Every task finishes with `make check` passing (ruff lint + format check, pyright strict, pytest, `node --test`). No task reads `.env` or copies its values anywhere. Tests use `tmp_path` `.env` files and obviously fake placeholder values.

## Tasks

- [x] 1. Scaffold the project and engineering guard-rails
  - [x] 1.1 Create `pyproject.toml` with the `src/` layout and pinned dependencies
    - `requires-python = ">=3.12"`, package `friday` under `src/friday/` with `__init__.py`, console script `friday = "friday.cli:main"`, and a minimal `src/friday/cli.py` `main()` stub
    - Runtime deps pinned with `==`: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `httpx`, `boto3`, `numpy`, `matplotlib`, `aws-sdk-transcribe-streaming[awscrt]`. Dev deps: `pytest`, `pytest-asyncio`, `hypothesis`, `pytest-cov`, `ruff`, `pyright`, `pre-commit`
    - Dev-only typing stubs pinned with `==`: `boto3-stubs[bedrock-runtime,polly,sts]` (or the `types-boto3` equivalents) and `botocore-stubs`
    - Create `.python-version` pinned to the local 3.14 minor version so `uv` runs are reproducible
    - Run `uv sync` on the local Python 3.14 and confirm every pin installs (numpy/matplotlib wheels included). If `aws-sdk-transcribe-streaming[awscrt]` fails to install or import, replace it with the pinned `amazon-transcribe` fallback and record the choice in a comment in `pyproject.toml`
    - Add `[tool.ruff]` (`target-version="py312"`, `line-length=100`, `lint.select=["E","F","I","B","UP","SIM","S"]`, `tests/**` ignores `S101`), `[tool.pyright]` (`typeCheckingMode="strict"`, `include=["src"]`), `[tool.pytest.ini_options]` (`asyncio_mode`, `testpaths`, `--cov=friday --cov-report=term-missing`), and `[tool.coverage]`
    - Keep strict pyright for all of `src/`, but downgrade `reportMissingTypeStubs` and `reportUnknownMemberType` to warnings only for the adapter modules (`llm/`, `speech/`, `data/plot.py`, `check.py`), via `executionEnvironments` or a per-file `# pyright:` comment, so `make check` passes without weakening the domain layer
    - Keep `uv.lock` tracked in the repo (the agent does not create commits; the user commits)
    - _Requirements: 13.1_

  - [x] 1.2 Create `Makefile`, `start.sh`, `.env.example`, README skeleton, and pre-commit config
    - `Makefile` targets `run`, `fmt`, `test` (`uv run pytest && node --test tests/js`), `check` (ruff check, ruff format --check, pyright, test), `verify-aws` (`uv run friday --check`)
    - `start.sh` exactly as in the design (`uv sync --frozen` with the "Friday: dependency installation failed" message and exit 1, then `exec uv run --no-sync friday "$@"`). Make it executable
    - `.env.example`: every variable from Req 12.1 marked REQUIRED/OPTIONAL, placeholder values only, defaults shown for optional settings, `openai.gpt-5.6-terra` and `us-east-2` as suggested values, and a note that `AWS_SESSION_TOKEN` is required for temporary (sandbox) credentials
    - `.gitignore` already exists and is complete. Verify `.gitignore` ignores `.env` (via `git check-ignore`) without reading `.env`
    - `README.md` skeleton with headings for prerequisites, setup, Start_Command, Local_URL, credential refresh, `--check`, and `make` targets
    - Optional `.pre-commit-config.yaml` (ruff, ruff-format, detect-secrets, revs pinned) and `.secrets.baseline`; `pre-commit install` works because the git repo is already initialized
    - _Requirements: 12.8, 12.9, 12.12, 12.13, 13.1, 13.6, 13.7_

  - [x] 1.3 Implement `constants.py`, `style.py`, and `errors.py`
    - `constants.py`: all limits and timeouts listed in the design as `Final` values
    - `style.py`: UI tokens (`--bg`, `--text`, `--cyan`, `--violet`, reduced-motion state colors), chart style (background, text, grid, size, dpi), and the five-color Neon_Accent palette
    - `errors.py`: the `FridayError(kind)` hierarchy exactly as in the design, with `kind` strings matching the wire codes
    - _Requirements: 4.10, 5.9, 11.4, 14.3, 14.7, 14.8_

  - [x] 1.4 Add architecture, parity, and smoke tests
    - `tests/test_architecture.py`: parse `src/friday` imports with `ast` and fail on forbidden layer edges (per the design's layering table), on `os.environ`/dotenv use outside `config.py`, and on `print(` outside `cli.py`
    - Create `src/friday/static/js/constants.js` (browser-facing limits) and `src/friday/static/styles.css` containing the `:root` tokens
    - `tests/test_parity.py`: `constants.js` values and `styles.css` `:root` tokens match `constants.py` and `style.py`
    - `tests/test_smoke.py`: `.env.example` lists every variable with required/optional markers, contains only placeholders, shows the suggested Model_ID and region and the session-token note; `.gitignore` contains `.env`
    - _Requirements: 12.8, 12.9, 12.12, 12.13_

  - [x] 1.5 Write style contrast tests
    - `tests/test_style.py`: computed WCAG relative luminance of page and chart backgrounds ≤ 0.05, text contrast ≥ 4.5:1 for UI and chart text tokens, five pairwise-distinct palette colors including a cyan and a violet
    - _Requirements: 14.3, 14.4, 14.7, 14.8_

- [x] 2. Implement configuration and redaction
  - [x] 2.1 Implement `config.py`
    - Frozen `Settings`, `REQUIRED`, `read_environment(dotenv_path)` (`dotenv_values()` overlaid with `os.environ`, never mutating `os.environ`, missing file allowed), pure `load_settings(env)` (blank = unset, one `ConfigError` listing all missing names, defaults for optional settings), `parse_port(raw)`, `secret_values(settings)`
    - _Requirements: 12.1, 12.2, 12.3, 12.6, 12.10, 13.3, 13.5_

  - [x] 2.2 Write property test for Settings merge and required-variable reporting
    - **Property 25: Settings merge and required-variable reporting**
    - **Validates: Requirements 12.2, 12.6**

  - [x] 2.3 Write property test for port parsing
    - **Property 26: Port parsing**
    - **Validates: Requirements 13.3, 13.5**

  - [x] 2.4 Implement `redact.py`
    - `Redactor(secrets, min_fragment=8)` masking every maximal span covered by an 8-character window of any Credential, plus `(AKIA|ASIA)[A-Z0-9]{16}` and `api_key=` query values; `RedactingFilter` rewriting `record.msg`/`args`
    - _Requirements: 5.13, 12.4_

  - [x] 2.5 Write property test for redaction
    - **Property 24: Redaction removes credentials in full and in part**
    - Generator `secret_texts()` defined in the test module (20-char key ID, 40-char secret, long session token, 32-char FRED key shapes, all random)
    - **Validates: Requirements 5.13, 12.4**

  - [x] 2.6 Write unit tests for configuration
    - `tests/test_config.py`: optional defaults, no `.env` file present, process env wins over `.env`, missing-variable message contains no Credential value
    - _Requirements: 12.3, 12.6, 12.10_

- [x] 3. Implement the Indicator_Map and date parsing
  - [x] 3.1 Implement `data/indicators.py` and `data/default_indicators.json`
    - Frozen `IndicatorEntry`/`IndicatorMap`, `resolve()` with `strip().casefold()` matching and `UnknownIndicator(names)`, a loader that rejects missing/unreadable/invalid JSON, empty `series_id` (error names the Indicator), and normalized name/alias collisions (error names the path and cause)
    - Default map with exactly the 10 Glossary rows (inflation uses `"yoy"`)
    - _Requirements: 6.1, 6.2, 6.9, 6.14, 12.7_

  - [x] 3.2 Write property test for indicator resolution
    - **Property 1: Indicator resolution is case- and whitespace-insensitive and complete**
    - **Validates: Requirements 6.2, 6.9**

  - [x] 3.3 Write unit tests for Indicator_Map loading
    - Malformed files (missing, unreadable, bad JSON, empty `series_id`, alias collision), and the packaged default map has exactly the 10 Glossary rows with the listed series IDs
    - _Requirements: 6.1, 6.14, 12.7_

  - [x] 3.4 Implement `data/dates.py`
    - `parse_date_range(start, end)` with strict `YYYY-MM-DD` and real-calendar checks, start ≤ end, raising `DateRangeError` that names the bad input. Shared by fetch and plot
    - _Requirements: 6.12, 11.9_

- [x] 4. Implement Dataset, CSV, and fetch transformations
  - [x] 4.1 Implement `data/dataset.py` and shared test strategies
    - Frozen `Dataset` with read-only float64 `values` (NaN = Missing), `preview_rows(ds, n=PREVIEW_ROWS)` (ISO dates, ≤ 4 decimals for display, `""` for Missing), `to_csv` (header `date,<value_column>`, `repr(float)`, empty field for Missing), `parse_csv`
    - `tests/strategies.py` with the shared `datasets()` and `fred_observations()` generators described in the design (edge cases: all-missing, single non-missing, leading/trailing missing, extreme floats)
    - _Requirements: 6.5, 7.1, 7.2, 7.3, 7.5, 7.6_

  - [x] 4.2 Write property test for preview rows
    - **Property 8: Preview shows the first rows in display format**
    - **Validates: Requirements 7.1, 7.2, 7.3**

  - [x] 4.3 Write property test for CSV round-trip
    - **Property 9: CSV export round-trips**
    - **Validates: Requirements 7.5, 7.6**

  - [x] 4.4 Implement `data/fetch.py`
    - `build_dataset(observations, entry, start, end)` (filter to bounds, sort ascending, `.` → Missing, value column named after the Indicator) and `apply_yoy(values)` (row-position formula, Missing on missing/zero denominator, drop first 12 rows, `RangeTooShortForYoY` when nothing remains)
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 6.13_

  - [x] 4.5 Write property test for Dataset construction
    - **Property 2: Dataset construction filters, sorts, and maps placeholders**
    - **Validates: Requirements 6.3, 6.4, 6.5**

  - [x] 4.6 Write property test for the YoY transformation
    - **Property 3: YoY transformation matches its formula and drops 12 rows**
    - **Validates: Requirements 6.6, 6.7, 6.13**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement statistics and fill
  - [x] 6.1 Implement `data/stats.py`
    - Frozen `Stats`; `describe(ds)` with numpy over non-missing values (sample std, None when count < 2; all non-count stats None when count is 0; first/last non-missing dates; linear percentiles); `format_stats_rows(stats)` producing the 11 labeled rows (integers for counts, 2 decimals otherwise, ISO dates, "undefined")
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6, 9.7, 9.8, 9.9_

  - [x] 6.2 Write property test for statistics against a reference model
    - **Property 10: Descriptive statistics match a reference model and are deterministic**
    - **Validates: Requirements 9.1, 9.2, 9.6, 9.8, 9.9**

  - [x] 6.3 Write property test for statistics ordering and counts
    - **Property 11: Descriptive statistics are ordered and counts are complete**
    - **Validates: Requirements 9.7**

  - [x] 6.4 Write property test for stats table formatting
    - **Property 12: Stats table formatting**
    - **Validates: Requirements 9.4**

  - [x] 6.5 Implement `data/fill.py`
    - `fill(ds, method="forward_fill") -> (Dataset, filled, unfilled)` for `forward_fill` and `linear_interpolation` (calendar-day weights), edges left Missing and counted as unfilled, new Dataset with `derived_from` and `fill_method` set and the same `value_column`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.11_

  - [x] 6.6 Write property test for fill semantics
    - **Property 13: Fill matches reference semantics**
    - **Validates: Requirements 10.3, 10.4, 10.5**

  - [x] 6.7 Write property test for fill structure and gap accounting
    - **Property 14: Fill preserves structure and accounts for every gap**
    - **Validates: Requirements 10.1, 10.6, 10.7**

  - [x] 6.8 Write property test for fill idempotence
    - **Property 15: Fill is idempotent**
    - **Validates: Requirements 10.8, 10.11**

- [x] 7. Implement the chart spec and renderer
  - [x] 7.1 Implement `build_chart_spec` in `data/plot.py`
    - Frozen `ChartSpec`; validate 1–5 distinct known IDs and a title ≤ 100 chars, apply `parse_date_range`, filter each series to the range, reject any Dataset with no non-missing point in range (naming it), assign palette colors by index, set x/y labels, title, legend flag, and alt text
    - _Requirements: 11.1, 11.2, 11.3, 11.5, 11.6, 11.7, 11.8, 11.9, 14.8_

  - [x] 7.2 Implement `ChartRenderer` in `data/plot.py`
    - OO matplotlib `Figure` on Agg under an instance `threading.Lock`, fixed style from `style.py` (dark background, light text, grid, 10×5.5 in at 110 dpi, PNG), NaN gaps, legend only for 2+ series; any exception becomes `ChartRenderError`
    - _Requirements: 11.4, 11.7, 11.10, 14.7, 14.8_

  - [x] 7.3 Write property test for chart specs
    - **Property 16: Chart spec reflects the requested series**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.6, 11.7, 14.8**

  - [x] 7.4 Write unit tests for the renderer
    - Returns valid PNG bytes for 1 and 5 series, leaves input Datasets unchanged, maps an injected matplotlib failure to `ChartRenderError`
    - _Requirements: 11.4, 11.10_

- [x] 8. Checkpoint - Domain layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Spike the MAF harness, then implement ports, session, events, and test fakes
  - [x] 9.0 Spike: add Microsoft Agent Framework and confirm its API on Python 3.14
    - Add `agent-framework-core==1.20.0` and `agent-framework-bedrock==1.0.0b261002` to `pyproject.toml` runtime dependencies (exact `==` pins), run `uv sync` on the local Python 3.14, and keep `uv.lock` updated. Confirm `uv sync --frozen` (used by `start.sh`) accepts the pinned pre-release without a global pre-release flag. Report, rather than loosen, any conflict with existing pins (pydantic, httpx, boto3, OpenTelemetry)
    - In a throwaway script outside `src/` and `tests/` (deleted afterwards; no credentials, no network), confirm imports and record the exact names in the design's "Items the spike must confirm" list: `create_harness_agent`, `Agent`, `AgentSession`, the function-tool type, function and chat middleware types, the context-provider hook, the chat-client base class to subclass for a fake, and the Bedrock module path (`agent_framework.amazon` or `agent_framework_bedrock`)
    - Using a minimal fake chat client, confirm: each disable flag (`disable_todo`, `disable_mode`, `disable_file_memory`, `disable_web_search`, `disable_tool_auto_approval`, `disable_compaction`) removes its tools and instructions from the request; setting `harness_instructions` replaces `DEFAULT_HARNESS_INSTRUCTIONS`; the function-invocation iteration limit and consecutive-error limit options; whether unknown tool names reach function middleware; whether MAF validates arguments before function middleware; how function middleware returns a result without calling the tool and terminates the run; `AgentSession` serialize/restore and appending a plain assistant message
    - Confirm the timeout mechanism (prebuilt boto3 client or botocore `Config` accepted by `BedrockChatClient`, else chat-middleware `asyncio.wait_for` only), and that `BedrockChatClient` built with explicit values reads no environment variable and no `.env` file (test with conflicting `BEDROCK_*`/`AWS_*` values in a monkeypatched environment, never the real `.env`)
    - Confirm OpenTelemetry with no exporter configured opens no network connection (socket guard), and note any MAF setup call or `OTEL_*` variable Friday must avoid
    - Record findings in `design.md` (replace each "to verify" marker with the confirmed API or the chosen fallback). If a finding invalidates part of the design, stop and ask the user before continuing
    - _Requirements: 5.2, 5.7, 5.14, 5.15, 12.4, 12.11_

  - [x] 9.1 Implement `ports.py`
    - `TTSClient`, `STTClient`, `FredSource` Protocols (no SDK imports). There is no `llm/base.py`: the LLM port is the MAF chat-client base abstraction, and `ToolSpec` lives in `agent/tools.py` (task 10.1)
    - _Requirements: 5.1, 5.4_

  - [x] 9.2 Implement `session.py`
    - `Session` (`agent_session: AgentSession`, insertion-ordered datasets, charts, FRED cache, counter, `asyncio.Lock`, `next_id(prefix)`, most-recent Dataset lookup) and `SessionStore(new_agent_session)` (`get_or_create` with UUID validation that creates the `AgentSession` through the injected factory, `end`, `evict_idle`)
    - _Requirements: 5.1, 6.8, 8.4_

  - [x] 9.3 Implement `events.py`
    - Event models for `status`, `dataset_preview`, `stats_table`, `chart`, `final` with per-turn `seq`, and an NDJSON encoder that passes every line through the `Redactor`
    - _Requirements: 4.4, 4.8, 7.1, 7.4, 7.9, 9.4, 11.6, 12.4_

  - [x] 9.4 Implement `tests/fakes.py` and the test network guard
    - `ScriptedChatClient`: a subclass of the MAF chat-client base class (as recorded in task 9.0) that replays scripted model responses (function calls, one or several per response, and final text), records the messages, instructions, and tool list of every model call, and can raise a credential error (bare or wrapped as the `__cause__` of a MAF-style exception), another error, or hang past the timeout at any call
    - `FakeTTS` (configurable failure/timeout/credential error), `FakeSTT`, and `FakeFred` (records calls, injectable `http_error`/`timeout`/`no_observations`)
    - `tests/conftest.py`: an autouse fixture (skipped for `tests/live/`) that fails any socket connection to a non-loopback address
    - _Requirements: 5.1, 5.7, 5.12, 6.10_

  - [x] 9.5 Update `tests/test_architecture.py` for MAF
    - `LAYER_OF`: remove `llm.base`, `llm.mantle`, and `llm.converse`; add `llm.bedrock` (adapters), `agent.harness` and `agent.middleware` (application)
    - `ALLOWED_THIRD_PARTY`: allow `agent_framework` in the application layer; adapters and composition stay unrestricted
    - Add a dotted-prefix rule so the MAF Bedrock module (path recorded in task 9.0) may be imported only by `llm.bedrock` and `wiring`, even if it is a submodule of `agent_framework`
    - Extend the checker self-tests: `agent.loop` importing `agent_framework` is allowed; `agent.loop` importing the MAF Bedrock module, `agent.middleware` importing `friday.aws_errors` or `botocore`, and `data.fetch` importing `agent_framework` are flagged
    - If the steering file `.kiro/steering/engineering-standards.md` still names `llm/base.py`, propose the one-line wording change to the user instead of editing it
    - _Requirements: 5.14, 12.4_

  - [x] 9.6 Update the suggested Model_ID to `us.openai.gpt-5.6-terra`
    - `.env.example`: `BEDROCK_MODEL_ID=us.openai.gpt-5.6-terra` with the comment "Suggested value: us.openai.gpt-5.6-terra" and a note that it is the US cross-Region inference profile used by the Bedrock runtime Converse API (no Mantle wording); region stays `us-east-2`
    - `tests/test_smoke.py` `test_suggested_model_id_and_region`: expect the new value and comment
    - `README.md`: replace any mention of the old ID or the Mantle endpoint with the new ID and Converse
    - Keep the placeholder-only and session-token rules intact; do not read `.env`
    - _Requirements: 12.8, 12.12_

- [x] 10. Implement the tool registry
  - [x] 10.1 Implement tool argument models, `specs()`, and `validate()` in `agent/tools.py`
    - Frozen Pydantic models with `extra="forbid"` for the four tools, limits from `constants.py`; `specs()` from `model_json_schema()`; `validate()` raises `ToolValidationError` naming the unknown tool or bad field, including "invalid method 'x'; supported: forward_fill, linear_interpolation"
    - _Requirements: 5.4, 5.8, 10.10, 11.4, 11.8_

  - [x] 10.2 Implement `ToolRegistry.run` and `function_tools()` for all four tools
    - `function_tools()` wraps each spec as a MAF function tool (schema from the arg model, approval never required) whose body calls `ToolRegistry.run` for the current turn's Session and never lets a Tool exception escape
    - Fetch (resolve → dates → per-Session FRED cache → `build_dataset` → optional YoY → store), Stats (read-only, defaults to most recent Dataset), Fill (stores a new Dataset), Plot (spec → `ChartRenderer` in `asyncio.to_thread` → stores chart); `ToolOutcome` with the JSON tool result and the display event payload; any exception becomes an `{ok: false, tool, error}` result with no Session change
    - _Requirements: 5.5, 5.11, 6.2, 6.8, 6.10, 6.11, 6.12, 6.15, 8.4, 9.1, 9.3, 9.10, 10.1, 10.2, 10.9, 11.1, 11.8, 11.9, 11.10_

  - [x] 10.3 Write property test for date validation before FRED
    - **Property 4: Invalid dates are rejected before any FRED call**
    - **Validates: Requirements 6.12**

  - [x] 10.4 Write property test for Session changes
    - **Property 5: Tool calls change the Session only through successful fetch/fill**
    - **Validates: Requirements 5.5, 5.11, 6.8, 6.11, 7.7, 9.3, 9.10, 10.1, 10.9, 11.4**

  - [x] 10.5 Write property test for most-recent Dataset defaults
    - **Property 6: Omitted dataset references resolve to the most recent Dataset**
    - **Validates: Requirements 8.4**

  - [x] 10.6 Write property test for invalid plot arguments
    - **Property 17: Invalid plot arguments produce errors and no chart**
    - **Validates: Requirements 11.8, 11.9**

  - [x] 10.7 Write unit tests for the registry
    - Exactly four tool specs with the documented schemas; validation messages name the field; FRED cache hit makes no second call; one FRED series per fetch with the mapped series ID
    - _Requirements: 5.4, 5.8, 6.15, 10.10_

- [x] 11. Implement templates, narration, and prompts
  - [x] 11.1 Implement `agent/templates.py`
    - Every user-facing string: start/done/error status lines, next-step offer pieces, and the four fixed outcome texts (`llm_unavailable`, `aws_credentials`, `tool_limit`, `no_data`), plus the STT, mic, and CSV-unavailable messages the Backend returns
    - _Requirements: 3.6, 4.1, 4.4, 4.5, 4.11, 5.7, 5.9, 5.12, 8.1, 8.2, 8.3, 8.6_

  - [x] 11.2 Implement status lines, `parse_reply`, and `next_step_offer` in `agent/narration.py`
    - `start_line`, `done_line` (≤ 10 words), `error_line` (names the failed action), `parse_reply` (`<display>`/`<spoken>` with 2-sentence fallback), `next_step_offer` rules from the design
    - _Requirements: 4.4, 4.5, 4.11, 8.1, 8.2, 8.3, 8.5_

  - [x] 11.3 Implement `guard` and `grounded_numbers` in `agent/narration.py`
    - Add "Boss" to display text when missing, drop spoken sentences with ungrounded numbers (0–2-decimal roundings and ISO date parts allowed), strip tables and data rows, truncate at sentence boundaries to keep the response ≤ 60 spoken words, and drop extra `done` lines from Spoken_Text when status lines plus offer alone exceed the limit
    - _Requirements: 4.1, 4.2, 4.3, 4.10, 9.5_

  - [x] 11.4 Write property test for next-step offers
    - **Property 18: Next-step offer follows the turn's outcomes**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.5**

  - [x] 11.5 Write property test for the narration guard
    - **Property 19: Narration guard enforces persona, grounding, and length**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.10, 9.5**

  - [x] 11.6 Implement `agent/prompts.py`
    - `FRIDAY_RULES` (passed as `harness_instructions`: numbers-only-from-tool-results rule, four-tools rule, `<display>`/`<spoken>` format, no next-step offers, relay supported Indicator names on `unknown_indicator`, offer retry on FRED failures), `FRIDAY_PERSONA` (passed as `agent_instructions`: Friday persona and the "Boss" rule), and a pure `dataset_inventory(session_view)` text builder with the most recent Dataset marked (used by the context provider in task 12.1)
    - _Requirements: 4.1, 4.2, 4.3, 5.6, 6.9, 6.10, 8.4_

- [x] 12. Implement the Agent on the MAF harness
  - [x] 12.1 Implement `agent/middleware.py`
    - `TurnContext` (Friday Session, event queue, call counter, outcomes, `stop`, `llm_error`) carried in a `contextvars.ContextVar` (or run-level middleware/kwargs if task 9.0 confirmed them)
    - `FridayMiddleware(tools, tts, map_llm_error)` function middleware: count every call; on the 9th, return a `tool_limit` error result without running the Tool, set `stop`, and terminate the run; `no_data` short-circuit for stats/fill/plot with no Datasets; `ToolRegistry.validate` with named-field errors and no status event; `status(tool_start)` with start-line TTS concurrent with the Tool; `next(...)`; display event; `status(tool_done|tool_error)`
    - Chat middleware: `asyncio.wait_for(..., LLM_TIMEOUT_S)` per model call, exceptions mapped with the injected `map_llm_error` and recorded in `TurnContext.llm_error`, and the unknown-tool-name counting fallback if task 9.0 showed unknown names skip function middleware
    - `DatasetInventoryProvider` context provider adding `dataset_inventory(...)` instructions before each run
    - No imports of `aws_errors`, botocore, or the MAF Bedrock package
    - _Requirements: 4.4, 4.5, 4.6, 4.11, 5.3, 5.5, 5.7, 5.8, 5.9, 5.11, 5.12, 8.4, 8.6, 10.10_

  - [x] 12.2 Implement `agent/harness.py`
    - `build_harness_agent(chat_client, tools, middleware)` calling `create_harness_agent` with `name="Friday"`, `harness_instructions=FRIDAY_RULES`, `agent_instructions=FRIDAY_PERSONA`, `tools=tools.function_tools()`, the inventory context provider, the Friday middleware, `disable_todo`, `disable_mode`, `disable_file_memory`, `disable_web_search`, `disable_tool_auto_approval`, and `disable_compaction` all `True`, no token limits, no skills/file access/shell/background agents/looping, iteration limit `MAX_TOOL_CALLS + 1`, and a consecutive-error limit of at least `MAX_TOOL_CALLS`
    - _Requirements: 5.1, 5.4, 5.6, 5.14, 5.15_

  - [x] 12.3 Implement the `Agent` facade `run_turn` in `agent/loop.py`
    - Snapshot the `AgentSession` state; run the harness as a task while yielding queued events; restore the snapshot on `AwsCredentialError` (`final(outcome="aws_credentials")`, no TTS call) and on `LLMUnavailableError`, timeout, or any other harness error (`final(outcome="llm_unavailable")`); for `tool_limit`/`no_data`, emit the fixed outcome text and append it to the `AgentSession` as an assistant message; otherwise `parse_reply` → `guard` → `next_step_offer` → TTS with 15 s timeout (`audio_error` on failure) → `final(outcome="ok")`; `final` always last
    - _Requirements: 4.4, 4.5, 4.6, 4.7, 4.8, 4.11, 5.1, 5.3, 5.5, 5.7, 5.9, 5.10, 5.11, 5.12, 8.6_

  - [x] 12.4 Write property test for no-data analysis requests
    - **Property 7: Analysis requests with no data run no Tool**
    - **Validates: Requirements 8.6**

  - [x] 12.5 Write property test for status event bracketing
    - **Property 20: Status events bracket every tool execution**
    - Generator `llm_scripts()` defined in the test module, replayed by `ScriptedChatClient`
    - **Validates: Requirements 4.4, 4.5, 4.11**

  - [x] 12.6 Write property test for the tool-call budget
    - **Property 21: Tool-call budget counts every request, including rejected ones**
    - **Validates: Requirements 5.8, 5.9, 10.10**

  - [x] 12.7 Write property test for full-history LLM calls
    - **Property 22: The LLM sees the full Session history**
    - **Validates: Requirements 5.1, 5.3, 5.4, 5.6, 5.10, 5.15**

  - [x] 12.8 Write property test for history rollback
    - **Property 23: LLM failures roll back the turn's history**
    - **Validates: Requirements 5.7, 5.12**

  - [x] 12.9 Write unit tests for the Agent
    - `tests/test_agent.py`: "please pull inflation" happy path with the full event sequence and offer; fixed outcome texts; system-prompt rules on every call; `aws_credentials` makes no TTS call; TTS failure and timeout set `audio_error` and still send text; a chat-client call that hangs past 60 s yields `llm_unavailable`
    - _Requirements: 4.4, 4.6, 4.8, 5.6, 5.7, 5.9, 5.12, 8.1, 8.6_

  - [x] 12.10 Write unit tests for the harness composition
    - `tests/test_harness.py`: every disable flag is passed and no opt-in capability is configured; the scripted client sees exactly the four Friday tools and no harness tools or `DEFAULT_HARNESS_INSTRUCTIONS`; the iteration and consecutive-error limits are set; a full fake turn makes no non-loopback network connection (telemetry included)
    - _Requirements: 5.4, 5.14, 5.15, 12.4_

- [x] 13. Checkpoint - Agent works end-to-end with fakes
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement adapters (verified with mocks only, no live calls)
  - [x] 14.1 Implement `aws_errors.py`
    - Pure `classify_aws_error(exc) -> ErrorKind` for botocore `ClientError` codes, HTTP 401/403 body `code`/`__type`, smithy exception names, and timeouts (`asyncio.TimeoutError`, botocore read/connect timeouts) (`CREDENTIAL`, `TIMEOUT`, `OTHER`), with redacted messages
    - Walk the exception chain (`exc`, `__cause__`, `__context__`, with a visited set against cycles) so a botocore error wrapped inside MAF exceptions is still found
    - _Requirements: 5.12, 5.13_

  - [x] 14.2 Implement `data/fred.py` `FredClient`
    - `httpx.AsyncClient` from the Container, `series/observations` with `file_type=json`, optional bounds, 15 s timeout, `FredError` kinds `http_error`/`timeout`/`no_observations`; `api_key` never appears in errors or logs
    - _Requirements: 6.2, 6.10, 6.15, 12.4_

  - [x] 14.3 Implement `llm/bedrock.py`
    - `make_chat_client(settings) -> BedrockChatClient` built only from explicit Settings values (model ID passed through unchanged, region, access key, secret, session token when set), never pointed at an env file; if task 9.0 showed the client consults the default AWS credential chain or accepts a prebuilt client, pass a `boto3.Session`/`bedrock-runtime` client built from the explicit credentials with `Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 1})`
    - `map_llm_error(exc) -> FridayError` using `classify_aws_error`: `CREDENTIAL` → `AwsCredentialError`, `TIMEOUT`/`OTHER` → `LLMUnavailableError`, redacted messages
    - The only module importing the MAF Bedrock package (besides `wiring.py`)
    - _Requirements: 5.2, 5.7, 5.12, 5.13, 5.14, 12.11_

  - [x] 14.4 Write a Converse contract test for the scripted fake
    - `tests/test_llm_bedrock_contract.py`: drive the real `BedrockChatClient` (from `make_chat_client` with placeholder credentials) against `botocore.stub.Stubber` Converse responses containing `toolUse` and text, through the harness with the four Friday tools; assert the tool calls and results round-trip in the same MAF message shapes `ScriptedChatClient` produces, and that the request carries the configured model ID and the four `toolSpec` entries. If the client cannot accept a stubbed boto3 client, record that in the test module and cover the shapes with the spike's findings instead
    - _Requirements: 5.1, 5.3, 5.14, 12.11_

  - [x] 14.5 Implement `speech/tts.py` `PollyTTS`
    - `synthesize_speech` with configured `VoiceId`/`Engine`, `OutputFormat="mp3"`, in `asyncio.to_thread` under `asyncio.wait_for(15)`; map to `TTSError(tts_failed|tts_timeout)` or `AwsCredentialError`
    - _Requirements: 4.6, 4.8, 5.12, 12.11_

  - [x] 14.6 Implement `speech/stt.py` `TranscribeSTT`
    - Streaming client with explicit static credentials from Settings, configured language code, PCM 16 kHz, 100 ms chunks paced at 4× real time, non-partial results joined, `asyncio.wait_for(30)`; `STTError(stt_failed|stt_timeout|stt_empty)` or `AwsCredentialError`; audio held only in memory. Use the `amazon-transcribe` fallback if task 1.1 selected it, keeping the same `STTClient` interface
    - _Requirements: 3.3, 3.6, 5.12, 12.11_

  - [x] 14.7 Write unit tests for AWS error classification and the Bedrock factory
    - `tests/test_llm_bedrock.py`: `classify_aws_error` table (five credential codes, timeouts, other), each bare and wrapped one and two levels deep via `__cause__` and `__context__`, plus a cyclic chain; `map_llm_error` mapping; `make_chat_client` passes the Settings model ID unchanged (including an ID without an inference-profile prefix), region, and credentials with and without a session token, and ignores conflicting `BEDROCK_*`/`AWS_*` values set with `monkeypatch` (placeholder values only)
    - _Requirements: 5.2, 5.12, 5.13, 12.11_

  - [x] 14.8 Write unit tests for the FRED client
    - `tests/test_fred.py`: three failure kinds, query parameters, no `api_key` in error text, using `httpx.MockTransport`
    - _Requirements: 6.10, 6.15, 12.4_

  - [x] 14.9 Write unit tests for speech adapters
    - `tests/test_speech.py`: Polly called with configured voice/engine (Stubber); Transcribe language code, empty transcript, and timeout mapping with a faked stream client
    - _Requirements: 3.3, 3.6, 4.6, 4.8_

- [x] 15. Implement wiring, HTTP server, CLI, and credential check
  - [x] 15.1 Implement `wiring.py`
    - Frozen `Container`, `build_container(settings, *, chat_client=None, tts=None, stt=None, fred=None)`: the `BedrockChatClient` from `make_chat_client` unless overridden, `FridayMiddleware` with `map_llm_error`, the harness agent from `build_harness_agent`, `SessionStore(harness.create_session)`, the `Agent` facade, one `Redactor`, `ChartRenderer`, shared `httpx.AsyncClient`, boto3 clients, Indicator_Map, and an async `close()`. No routing by model-ID prefix
    - _Requirements: 5.2, 5.14, 6.1, 12.11_

  - [x] 15.2 Implement `server.py` `create_app(container, on_ready)`
    - Routes from the design's HTTP table (`/`, `/static/*`, `/api/chat` NDJSON streaming under the per-Session lock, `/api/transcribe`, `/api/datasets/{id}.csv` with the attachment filename, `/api/session/end`); `TrustedHostMiddleware`, CSP and security headers, body-size limits; redacted JSON error bodies; lifespan calls `on_ready` and closes the Container
    - Origin check: state-changing `POST` routes (`/api/chat`, `/api/transcribe`, `/api/session/end`) reject a present `Origin` header that is not the Local_URL origin (`http://127.0.0.1:<port>` or `http://localhost:<port>`) with `403 forbidden_origin`
    - `/api/chat` requires `Content-Type: application/json` and the `X-Friday-Session` header
    - Register no CORS middleware
    - _Requirements: 1.7, 3.3, 3.6, 5.12, 7.4, 7.5, 7.8, 12.4, 12.5, 13.4, 14.5_

  - [x] 15.3 Implement `check.py`
    - `run_checks(container) -> list[CheckResult]`: STS `GetCallerIdentity`, one tiny Converse request ("Reply with OK.", no tools) sent directly through the Container's `BedrockChatClient` (not the harness) with the 60 s limit and `map_llm_error`, Polly `"Check."`, Transcribe 1 s of silence, FRED `UNRATE` with `limit=1`; each result classified and redacted
    - _Requirements: 5.12, 5.13, 12.4, 12.11_

  - [x] 15.4 Implement `cli.py` `main()`
    - Startup order from the design: `read_environment` → `load_settings` → `parse_port` → Indicator_Map → logging with `RedactingFilter` on the handler (`httpx`/`httpcore`/`agent_framework`/`botocore` at WARNING; no OpenTelemetry exporter configured) → manual bind to `127.0.0.1` (`EADDRINUSE`/`EACCES` mapped) → uvicorn `serve(sockets=[sock])`, `log_config=None`, `timeout_graceful_shutdown=3` → print Local_URL with the actual port → background STS probe warning; exit code 2 on config errors; `--check` prints OK/FAIL lines and exits 0 only when all pass
    - _Requirements: 12.1, 12.2, 12.6, 12.7, 12.10, 13.2, 13.3, 13.4, 13.5, 13.8_

  - [x] 15.5 Write HTTP tests
    - `tests/test_http.py` (with fakes via `build_container`): NDJSON stream shape for a chat turn, `400 invalid_text` for empty and >2000-char text, `413` for oversized bodies, `400` for a foreign `Host`, CSP and security headers on `/`, CSV headers and filename, `404 dataset_not_found`, `/api/transcribe` error codes, no Credential fragments in any response body
    - Cross-origin: foreign `Origin` on each state-changing `POST` returns `403 forbidden_origin`; `/api/chat` without `X-Friday-Session` or with a non-JSON `Content-Type` is rejected; no response carries `Access-Control-Allow-Origin`
    - _Requirements: 1.7, 3.6, 5.12, 7.5, 7.8, 12.4, 12.5, 14.5_

  - [x] 15.6 Write startup tests
    - `tests/test_startup.py`: missing required vars exit non-zero with one message; malformed Indicator_Map exits non-zero; port in use (pre-bound socket) and invalid port exit non-zero naming the value and cause; server binds only to 127.0.0.1 and prints the Local_URL; exits within 5 s after SIGINT
    - _Requirements: 12.6, 12.7, 13.2, 13.4, 13.5, 13.8_

- [x] 16. Checkpoint - Backend complete offline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Implement the frontend
  - [x] 17.1 Implement `index.html` and full `styles.css`
    - Title "Friday — Your next-gen eco-buddy", App_Header with "Friday" and the exact Tagline, chat `<section role="log" aria-live="polite">`, orb `<aside role="img">`, `<textarea>` Chat_Input and Mic_Button (`aria-pressed`), only `styles.css` and `js/app.js` loaded, no inline scripts/styles
    - Dark_Theme from the existing `:root` tokens, neon Friday message border and labels, visible focus ring, orb states (static idle, `--level`-driven listening/speaking, 1.5 s pulse for working), reduced-motion static colors
    - _Requirements: 1.1, 2.2, 2.3, 2.4, 2.5, 2.6, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

  - [x] 17.2 Implement `js/state.js`
    - Pure `reduce(state, event, ctx)` implementing the design's state diagram; mic clicks in `working` are no-ops
    - _Requirements: 2.1, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 3.1, 3.2, 3.9, 3.10_

  - [x] 17.3 Implement `js/input.js`
    - Pure `validateSubmission(text, waiting)` returning `send`/`ignore`/`too_long` with trimmed text
    - _Requirements: 1.2, 1.3, 1.4, 1.7, 1.9, 3.4_

  - [x] 17.4 Implement `js/api.js`
    - `POST /api/chat` with `X-Friday-Session`, NDJSON line reader over `ReadableStream`, 60 s first-event and idle timers with abort → failure callback; `POST /api/transcribe` upload; session-end beacon
    - _Requirements: 1.8, 3.2, 3.6, 12.5_

  - [x] 17.5 Implement `js/audio.js`
    - One `AudioContext` created lazily, with an `unlock()` that creates it if needed and calls `resume()` inside a user-gesture handler
    - If the context is still suspended when a clip arrives, treat it like a TTS failure: report it so the "Audio unavailable" notice shows and the text is kept
    - FIFO clip queue decoded with `decodeAudioData`, one clip at a time through an `AnalyserNode`, `onstarted`/`ondrained` callbacks, `stop()` for mic interrupts, and holding a `dataset_preview` until its `tool_start` clip starts (released at once if that clip failed)
    - _Requirements: 2.5, 2.10, 4.4, 4.8, 4.9_

  - [x] 17.6 Implement `js/mic.js` and `js/pcm-worklet.js`
    - `getUserMedia` mono, worklet downsampling to 16 kHz Int16 in memory, 60 s auto-stop, analyser level for the orb, discard on typed submit, denial/no-device error callback
    - _Requirements: 2.3, 3.1, 3.2, 3.5, 3.8, 3.10_

  - [x] 17.7 Implement `js/chat.js`
    - Render user/Friday messages with labels, status chips, Preview_Table with the CSV link beside it (fetch → Blob download; 404 → "no longer available" message), "No data rows were returned" for empty previews, stats table, inline chart `<img alt>`, "Audio unavailable" notice, scroll newest into view
    - _Requirements: 1.5, 1.6, 4.8, 7.1, 7.2, 7.3, 7.4, 7.7, 7.8, 7.9, 9.4, 11.6_

  - [x] 17.8 Implement `js/app.js`
    - Wire input, mic, API, audio, chat, and state: session UUID, Enter handling, transcript submission, orb `data-state` and `--level` via `requestAnimationFrame` and `style.setProperty`, Chat_Input always enabled, error messages for mic denial, STT failures, and credential errors, idle when nothing is pending
    - Call the audio `unlock()` on the first Enter submit and the first Mic_Button click, so Friday's later asynchronous audio is allowed to play in Chrome and Safari
    - _Requirements: 1.2, 1.3, 1.8, 1.9, 2.6, 2.7, 2.8, 2.11, 2.12, 3.4, 3.5, 3.6, 3.7, 4.7, 4.8, 4.9_

  - [x] 17.9 Write state machine tests
    - `tests/js/state.test.mjs`: exhaustive state × event transition table
    - _Requirements: 2.1, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 3.1, 3.2, 3.9, 3.10_

  - [x] 17.10 Write input validation tests
    - `tests/js/input.test.mjs`: empty, whitespace-only, exactly 2000, 2001 characters, waiting state
    - _Requirements: 1.2, 1.4, 1.7, 1.9_

  - [x] 17.11 Write audio queue and API timer tests
    - `tests/js/audio.test.mjs` and `tests/js/api.test.mjs`: in-order, non-overlapping playback with a fake `AudioContext`; preview held until the status clip starts; first-event and idle 60 s timers with fake `fetch` and timers
    - Autoplay: `unlock()` resumes the lazily created context; a clip arriving while the context is still suspended reports an audio failure (text kept, "Audio unavailable" notice) and does not block the queue
    - _Requirements: 1.8, 4.4, 4.8, 4.9_

  - [x] 17.12 Extend smoke tests for static assets
    - `tests/test_smoke.py`: no third-party `http(s)://` URL or AWS/FRED hostname in `static/`; `index.html` has the title, header, tagline, and required element IDs; no inline `<script>`, `<style>`, or `style=`
    - _Requirements: 1.1, 12.5, 14.1, 14.2, 14.5_

- [x] 18. Finish the README and the opt-in live test suite
  - [x] 18.1 Complete `README.md`
    - Prerequisites (Python ≥ 3.12, uv, Node for tests, AWS credentials with Bedrock/Transcribe/Polly access and the IAM permissions from the `--check` table, including invoke access to the `us.openai.gpt-5.6-terra` inference profile and its destination-Region foundation models, FRED API key), a one-line note that the Agent runs on the Microsoft Agent Framework harness with only Friday's four tools, copying `.env.example` to `.env`, `./start.sh`, the Local_URL, refreshing expired sandbox credentials and restarting, `make verify-aws`, `make` targets, and the manual checks list
    - _Requirements: 13.6_

  - [x] 18.2 Write the opt-in live test suite
    - `tests/live/` skipped unless `FRIDAY_LIVE_TESTS=1` (the network guard is off here): FRED reports monthly frequency for all 10 default series; one direct `BedrockChatClient` Converse request; one harness turn through `build_container` that makes GPT-5.6 Terra call `fetch_data` and asserts a `dataset_preview` event and a `final(outcome="ok")`; one Polly synthesis; one Transcribe round trip
    - _Requirements: 3.3, 4.6, 5.1, 5.14, 6.14_

- [x] 19. Checkpoint - Everything passes offline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Checkpoint - Refresh AWS credentials and verify access
  - Ask the user to refresh the `AWS_*` values in `.env` themselves (the agent must not read or edit `.env`), then run `make verify-aws` and share the OK/FAIL output. Ensure all tests pass, ask the user if questions arise.

- [ ] 21. Resolve live verification findings
  - [ ] 21.1 Fix any failures reported by `make verify-aws` and the live suite
    - Confirm GPT-5.6 Terra tool calling works through the beta `BedrockChatClient` with the `us.openai.gpt-5.6-terra` inference profile in `us-east-2` (including the 60 s timeout path and credential-error classification through MAF's wrapper exceptions); switch to the `amazon-transcribe` fallback if the streaming SDK misbehaves on Python 3.14; adjust adapter mappings for any real response shapes that differ from the mocks, adding a regression unit test for each fix; rerun `make check` and `FRIDAY_LIVE_TESTS=1 uv run pytest tests/live`
    - _Requirements: 3.3, 4.6, 5.1, 5.2, 12.11_

- [ ] 22. Final checkpoint - Live end-to-end walkthrough
  - Ask the user to run `./start.sh`, open the printed Local_URL, and walk through: "pull inflation" by voice → spoken "Fetching…" status before the Preview_Table, preview + CSV download, spoken offer → "run the stats" → stats table and spoken findings → fetch a second indicator and "plot both" → multi-series chart with legend, with voice in and voice out throughout and the orb changing state. Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Core implementation tasks are never optional.
- Each property test lives in its own file, `tests/test_properties_pNN_<slug>.py`, uses `@settings(max_examples=100)` or higher, and carries the tag `Feature: friday-voice-data-assistant, Property {n}: {title}`.
- Shared generators live in `tests/strategies.py` (task 4.1). Generators used by only one property stay in that test module.
- No task reads `.env` or writes Credential values anywhere. Tests use fakes, mocks, stubs, and `tmp_path` `.env` files with placeholder values. MAF is never allowed to load `.env` or read Credentials from the environment; Settings values are passed explicitly. Live AWS calls happen only in task 15.3's `--check`, the opt-in `tests/live/` suite, and the checkpoints from task 20 onward.
- Every task ends with `make check` passing, per `.kiro/steering/engineering-standards.md`.
- Known POC tradeoffs:
  - The full Session history is sent to the LLM every turn (Req 5.1). Token cost grows with session length, which is acceptable for a POC because tool results are compact summaries.
  - `agent-framework-bedrock` is a beta. Its API names, timeout hooks, and environment handling are assumptions until task 9.0 records them, and live tool-calling behavior is confirmed only in task 21.1.
  - The harness persists history after each model call, so Friday snapshots and restores the `AgentSession` around each turn (Req 5.7, 5.12). Datasets and charts created before a failure in the same turn stay in the Session.
  - Sessions are memory-only and are lost when the server stops.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.1", "2.4", "3.1", "3.4", "4.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.5", "2.6", "3.2", "3.3", "4.2", "4.3", "4.4", "6.1", "6.5", "7.1", "9.0", "9.1", "9.6"] },
    { "id": 4, "tasks": ["4.5", "4.6", "6.2", "6.3", "6.4", "6.6", "6.7", "6.8", "7.2", "9.2", "9.3", "9.5", "11.1", "14.1"] },
    { "id": 5, "tasks": ["7.3", "7.4", "9.4", "10.1", "11.2", "14.2", "14.3", "14.5", "14.6"] },
    { "id": 6, "tasks": ["10.2", "11.3", "11.6", "14.7", "14.8", "14.9", "17.1", "17.2", "17.3", "17.4", "17.5", "17.6"] },
    { "id": 7, "tasks": ["10.3", "10.4", "10.5", "10.6", "10.7", "11.4", "11.5", "12.1", "17.7", "17.9", "17.10", "17.11"] },
    { "id": 8, "tasks": ["12.2", "17.8"] },
    { "id": 9, "tasks": ["12.3", "14.4"] },
    { "id": 10, "tasks": ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "15.1"] },
    { "id": 11, "tasks": ["15.2", "15.3"] },
    { "id": 12, "tasks": ["15.4", "15.5", "17.12"] },
    { "id": 13, "tasks": ["15.6", "18.1", "18.2"] },
    { "id": 14, "tasks": ["21.1"] }
  ]
}
```
