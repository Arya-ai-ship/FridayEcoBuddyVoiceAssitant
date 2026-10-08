"""Repository smoke checks: ``.env.example`` and ``.gitignore`` (Req 12.8, 12.9, 12.12, 12.13).

Only ``.env.example`` is read. ``.env`` is never opened; its ignore status is checked
through ``git check-ignore``.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from friday import constants

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
GITIGNORE = ROOT / ".gitignore"

REQUIRED_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "FRED_API_KEY",
    "BEDROCK_MODEL_ID",
)
OPTIONAL_DEFAULTS = {
    "AWS_SESSION_TOKEN": "unset",
    "POLLY_VOICE_ID": "Joanna",
    "POLLY_ENGINE": "neural",
    "TRANSCRIBE_LANGUAGE_CODE": "en-US",
    "INDICATOR_MAP_PATH": "unset",
    "FRIDAY_PORT": str(constants.DEFAULT_PORT),
}
SECRET_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "FRED_API_KEY")
PLACEHOLDER = re.compile(r"^(|your-[a-z0-9-]+)$")
# Shapes of real Credential values: AWS key IDs, 40-char secrets, 32-char FRED keys.
SECRET_SHAPES = (
    re.compile(r"(AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    re.compile(r"(?<![a-z0-9])[a-z0-9]{32}(?![a-z0-9])"),
)
ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def parse_env_example(text: str) -> dict[str, tuple[str, str]]:
    """Map each variable to ``(value, comment block directly above it)``."""
    entries: dict[str, tuple[str, str]] = {}
    comments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comments.append(stripped.lstrip("#").strip())
        elif match := ASSIGNMENT.match(stripped):
            entries[match.group(1)] = (match.group(2).strip(), " ".join(comments))
            comments = []
        elif not stripped:
            comments = []
    return entries


@pytest.fixture(scope="module")
def env_example() -> dict[str, tuple[str, str]]:
    return parse_env_example(ENV_EXAMPLE.read_text(encoding="utf-8"))


def test_env_example_lists_every_variable(env_example: dict[str, tuple[str, str]]) -> None:
    assert set(env_example) == {*REQUIRED_VARS, *OPTIONAL_DEFAULTS}


@pytest.mark.parametrize("name", REQUIRED_VARS)
def test_required_variables_are_marked(name: str, env_example: dict[str, tuple[str, str]]) -> None:
    comment = env_example[name][1]
    assert "REQUIRED" in comment
    assert "OPTIONAL" not in comment


@pytest.mark.parametrize(("name", "default"), OPTIONAL_DEFAULTS.items())
def test_optional_variables_are_marked_with_default(
    name: str, default: str, env_example: dict[str, tuple[str, str]]
) -> None:
    comment = env_example[name][1]
    assert comment.startswith("OPTIONAL")
    assert f"Default: {default}" in comment


@pytest.mark.parametrize("name", SECRET_VARS)
def test_secret_values_are_placeholders(name: str, env_example: dict[str, tuple[str, str]]) -> None:
    assert PLACEHOLDER.match(env_example[name][0])


def test_env_example_contains_no_credential_shapes() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for shape in SECRET_SHAPES:
        assert shape.search(text) is None, shape.pattern


def test_suggested_model_id_and_region(env_example: dict[str, tuple[str, str]]) -> None:
    model_value, model_comment = env_example["BEDROCK_MODEL_ID"]
    region_value, region_comment = env_example["AWS_REGION"]
    assert model_value == "us.openai.gpt-5.6-terra"
    assert "Suggested value: us.openai.gpt-5.6-terra" in model_comment
    assert "Converse" in model_comment
    assert "mantle" not in model_comment.lower()
    assert region_value == "us-east-2"
    assert "Suggested value: us-east-2" in region_comment


def test_session_token_note(env_example: dict[str, tuple[str, str]]) -> None:
    comment = env_example["AWS_SESSION_TOKEN"][1]
    assert comment.startswith("OPTIONAL")
    assert re.search(r"Required when using temporary \(sandbox\) AWS credentials", comment)


def test_gitignore_lists_env() -> None:
    lines = {line.strip() for line in GITIGNORE.read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines


def _git_ignored(path: str) -> bool:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("git repository not available")
    # Fixed argument list, no shell; only checks ignore rules, never reads the file.
    result = subprocess.run(  # noqa: S603
        [git, "check-ignore", "-q", path], cwd=ROOT, check=False, capture_output=True
    )
    return result.returncode == 0


def test_git_ignores_env_but_not_example() -> None:
    assert _git_ignored(".env")
    assert not _git_ignored(".env.example")


def test_parser_reads_comment_blocks() -> None:
    parsed = parse_env_example("# REQUIRED. Key.\nA=x\n\n# OPTIONAL. Default: 1\nB=1\n")
    assert parsed == {"A": ("x", "REQUIRED. Key."), "B": ("1", "OPTIONAL. Default: 1")}


# --- Static assets (task 17.12, Req 1.1, 12.5, 14.1, 14.2, 14.5) --------------------

STATIC_DIR = ROOT / "src" / "friday" / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# A third-party URL or an AWS/FRED hostname must never appear in a static asset (Req 12.5, 14.5).
FORBIDDEN_IN_STATIC = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"amazonaws\.com", re.IGNORECASE),
    re.compile(r"stlouisfed\.org", re.IGNORECASE),
    re.compile(r"cdn\.", re.IGNORECASE),
    re.compile(r"googleapis\.com", re.IGNORECASE),
)
# Required element IDs app.js wires (Req 1.1).
REQUIRED_IDS = ("chat-log", "voice-orb", "chat-input", "mic-button")


def _static_files() -> list[Path]:
    return [p for p in STATIC_DIR.rglob("*") if p.is_file()]


def test_no_third_party_url_or_cloud_hostname_in_static() -> None:
    for path in _static_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_IN_STATIC:
            assert pattern.search(text) is None, f"{pattern.pattern} found in {path.name}"


def test_index_has_title_header_and_tagline() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "<title>Friday — Your next-gen eco-buddy</title>" in html
    assert "<h1>Friday</h1>" in html
    assert "Your next-gen eco-buddy" in html


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_index_has_required_element_ids(element_id: str) -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert f'id="{element_id}"' in html


def test_index_loads_only_styles_and_app_module() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b[^>]*>", html, re.IGNORECASE)
    assert scripts == ['<script type="module" src="/static/js/app.js">']
    stylesheets = re.findall(r'<link\b[^>]*rel="stylesheet"[^>]*>', html, re.IGNORECASE)
    assert len(stylesheets) == 1 and "styles.css" in stylesheets[0]


def test_index_has_no_inline_script_style_or_style_attribute() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "<style" not in html.lower()
    assert "style=" not in html.lower()
    # The only <script> is the external module (no inline body).
    assert re.search(r"<script\b(?![^>]*\bsrc=)", html, re.IGNORECASE) is None
