.PHONY: run fmt test check verify-aws

# Start Friday locally (same as the documented Start_Command).
run:
	./start.sh

# Auto-format and apply safe lint fixes.
fmt:
	uv run ruff format .
	uv run ruff check --fix .

# Python tests (pytest + Hypothesis) and browser-module tests (node --test).
test:
	uv run pytest && node --test tests/js

# Everything that must pass before a task is done.
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	$(MAKE) test

# Live credential/permission check against AWS and FRED.
verify-aws:
	uv run friday --check
