"""Isolated boto3 sessions built only from explicit ``Settings`` credentials.

boto3 reads ``AWS_PROFILE``, ``AWS_CONFIG_FILE``, and ``AWS_SHARED_CREDENTIALS_FILE``
even when explicit keys are given, and a missing ``AWS_PROFILE`` raises
``ProfileNotFound`` (design: spike finding 8). It also honors endpoint overrides
(``AWS_ENDPOINT_URL``, ``AWS_ENDPOINT_URL_<SERVICE>``). :func:`isolated_boto3_session`
neutralizes those session variables and pins ``ignore_configured_endpoint_urls`` to
``True``, so every AWS client Friday builds (Bedrock, Polly, Transcribe, STS) uses
exactly the configured credentials, region, and the default AWS endpoint.
"""

from __future__ import annotations

import os

import boto3
import botocore.session

from friday.config import Settings

# botocore session variable spec: (config_name, env_var_name, default, conversion).
# No env var and no config key means the value cannot come from the environment, so
# the default always applies (``AWS_IGNORE_CONFIGURED_ENDPOINT_URLS`` is not read).
_ISOLATED_SESSION_VARS: dict[str, tuple[str | None, str | None, str | bool | None, None]] = {
    "profile": (None, None, None, None),
    "config_file": (None, None, os.devnull, None),
    "credentials_file": (None, None, os.devnull, None),
    "ignore_configured_endpoint_urls": (None, None, True, None),
}


def isolated_boto3_session(settings: Settings) -> boto3.Session:
    """Return a boto3 session from the explicit Settings credentials and region.

    No profile, shared config/credentials file, or configured endpoint URL is
    consulted. The session token is passed only when set (``None`` otherwise, Req 12.11).
    """
    core = botocore.session.Session(session_vars=dict(_ISOLATED_SESSION_VARS))
    return boto3.Session(
        botocore_session=core,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        aws_session_token=settings.aws_session_token,
        region_name=settings.aws_region,
    )
