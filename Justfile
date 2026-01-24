set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

lint:
	uv run --group dev ruff check

format:
	uv run --group dev ruff format

fix:
	uv run --group dev ruff check --fix
	uv run --group dev ruff format

test:
	uv run --group dev pytest

test-all:
	for py in 3.10 3.11 3.12 3.13 3.14; do uv run -p "$py" --group dev pytest; done
