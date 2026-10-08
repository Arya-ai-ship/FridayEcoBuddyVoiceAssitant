"""HTTP tests for ``server.create_app`` with fakes via ``build_container`` (task 15.5).

Covers the NDJSON stream shape, text validation, body-size limits, trusted-host and
cross-origin rejection, the required session header and JSON content type, CSP and
security headers, CSV headers/filename and the 404, the transcribe error codes, and that
no response body carries a Credential fragment (Req 1.7, 3.6, 5.12, 7.5, 7.8, 12.4, 12.5, 14.5).
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from urllib.parse import urljoin, urlparse

import pytest
from fastapi.testclient import TestClient

from fakes import Call, FakeFred, FakeSTT, FakeTTS, ScriptedChatClient, Step, calls, text
from friday.config import Settings
from friday.constants import MAX_CHAT_BODY_BYTES, MAX_TEXT_CHARS
from friday.server import create_app
from friday.wiring import build_container

SID = "00000000-0000-4000-8000-000000000000"
BASE = "http://127.0.0.1:8000"
SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - placeholder
REPLY = "<display>Boss, done.</display><spoken>Done, Boss.</spoken>"


def _settings() -> Settings:
    return Settings(
        aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
        aws_secret_access_key=SECRET,
        aws_session_token=None,
        aws_region="us-east-2",
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id="us.openai.gpt-5.6-terra",
    )


def _client(*, script: Sequence[Step] | None = None, stt: FakeSTT | None = None) -> TestClient:
    container = build_container(
        _settings(),
        chat_client=ScriptedChatClient(script if script is not None else [text(REPLY)]),
        tts=FakeTTS(),
        stt=stt or FakeSTT(),
        fred=FakeFred(),
    )
    return TestClient(create_app(container), base_url=BASE)


def _headers(**extra: str) -> dict[str, str]:
    return {"X-Friday-Session": SID, **extra}


def test_index_has_csp_and_security_headers() -> None:
    with _client() as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["Content-Security-Policy"].startswith("default-src 'self'")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "Access-Control-Allow-Origin" not in r.headers


def test_chat_streams_ndjson_ending_in_final() -> None:
    script = [calls(Call("fetch_data", {"indicator": "inflation"})), text(REPLY)]
    with _client(script=script) as client:
        r = client.post("/api/chat", json={"text": "pull inflation"}, headers=_headers())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        assert events[-1]["type"] == "final"
        assert any(e["type"] == "dataset_preview" for e in events)


@pytest.mark.parametrize(
    "body", [{"text": ""}, {"text": "   "}, {"text": "x" * (MAX_TEXT_CHARS + 1)}]
)
def test_invalid_text_is_rejected(body: dict[str, str]) -> None:
    with _client() as client:
        r = client.post("/api/chat", json=body, headers=_headers())
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_text"


def test_oversized_chat_body_is_rejected() -> None:
    with _client() as client:
        big = "a" * (MAX_CHAT_BODY_BYTES + 100)
        r = client.post(
            "/api/chat",
            content=json.dumps({"text": big}),
            headers=_headers(**{"Content-Type": "application/json"}),
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "request_too_large"


def test_chat_requires_the_session_header() -> None:
    with _client() as client:
        r = client.post("/api/chat", json={"text": "hi"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_session"


def test_chat_requires_json_content_type() -> None:
    with _client() as client:
        r = client.post(
            "/api/chat", content="text=hi", headers=_headers(**{"Content-Type": "text/plain"})
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_text"


def test_foreign_host_is_rejected() -> None:
    with _client() as client:
        r = client.get("/", headers={"Host": "evil.example"})
        assert r.status_code == 400


@pytest.mark.parametrize("path", ["/api/chat", "/api/transcribe", "/api/session/end"])
def test_foreign_origin_is_rejected_on_state_changing_posts(path: str) -> None:
    with _client() as client:
        r = client.post(path, headers=_headers(Origin="http://evil.example"), content=b"")
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "forbidden_origin"


def test_no_response_carries_cors_header() -> None:
    with _client() as client:
        r = client.post("/api/chat", json={"text": "hi"}, headers=_headers())
        assert "Access-Control-Allow-Origin" not in r.headers


def test_csv_download_has_attachment_filename() -> None:
    script = [calls(Call("fetch_data", {"indicator": "inflation"})), text(REPLY)]
    with _client(script=script) as client:
        client.post("/api/chat", json={"text": "pull inflation"}, headers=_headers())
        r = client.get(f"/api/datasets/ds-1.csv?session={SID}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        disposition = r.headers["content-disposition"]
        assert disposition.startswith("attachment;") and "ds-1.csv" in disposition


def test_csv_for_unknown_dataset_is_404() -> None:
    with _client() as client:
        r = client.get(f"/api/datasets/ds-99.csv?session={SID}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "dataset_not_found"


def test_transcribe_returns_text_on_success() -> None:
    with _client(stt=FakeSTT(transcript="pull inflation")) as client:
        r = client.post("/api/transcribe", content=b"\x00\x01" * 100, headers=_headers())
        assert r.status_code == 200
        assert r.json()["text"] == "pull inflation"


def test_transcribe_empty_is_a_400_stt_empty() -> None:
    with _client(stt=FakeSTT(failure="empty")) as client:
        r = client.post("/api/transcribe", content=b"\x00\x01" * 100, headers=_headers())
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "stt_empty"


def test_transcribe_credential_error_is_403_aws_credentials() -> None:
    with _client(stt=FakeSTT(failure="credentials")) as client:
        r = client.post("/api/transcribe", content=b"\x00\x01" * 100, headers=_headers())
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "aws_credentials"


def test_session_end_returns_204() -> None:
    with _client() as client:
        r = client.post("/api/session/end", headers=_headers())
        assert r.status_code == 204


def test_no_secret_fragment_in_any_response_body() -> None:
    with _client() as client:
        responses = [
            client.get("/"),
            client.post("/api/chat", json={"text": "hi"}, headers=_headers()),
            client.get(f"/api/datasets/ds-99.csv?session={SID}"),
        ]
        for r in responses:
            assert SECRET not in r.text
            assert SECRET[:8] not in r.text


# --- Every asset the browser loads is actually served ----------------------------

_REF = re.compile(r'(?:src|href)="([^"]+)"')
_IMPORT = re.compile(r'from\s+"(\.{1,2}/[^"]+)"')
_MODULE_URL = re.compile(r'new URL\("(\.{1,2}/[^"]+)",\s*import\.meta\.url\)')
_JS_TYPES = ("text/javascript", "application/javascript")


def test_every_asset_the_page_loads_is_served() -> None:
    """Follow index.html refs, ES-module imports, and module-relative URLs like a browser."""
    with _client() as client:
        page = client.get("/")
        assert page.status_code == 200
        queue = [urljoin(BASE + "/", ref) for ref in _REF.findall(page.text)]
        seen: set[str] = set()
        while queue:
            url = queue.pop()
            if url in seen:
                continue
            seen.add(url)
            r = client.get(urlparse(url).path)
            assert r.status_code == 200, f"{urlparse(url).path} -> {r.status_code}"
            if url.endswith(".js"):
                assert r.headers["content-type"].startswith(_JS_TYPES), url
                refs = _IMPORT.findall(r.text) + _MODULE_URL.findall(r.text)
                queue.extend(urljoin(url, ref) for ref in refs)
            elif url.endswith(".css"):
                assert r.headers["content-type"].startswith("text/css"), url
        paths = {urlparse(u).path for u in seen}
        assert {"/static/styles.css", "/static/js/app.js", "/static/js/pcm-worklet.js"} <= paths
