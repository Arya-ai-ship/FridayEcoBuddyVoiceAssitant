#!/bin/sh
uv sync --frozen || { echo "Friday: dependency installation failed" >&2; exit 1; }
exec uv run --no-sync friday "$@"
