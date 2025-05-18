.PHONY: run install clean check runner
.DEFAULT_GOAL:=runner

run: install
	uv run python runner.py

install: pyproject.toml
	uv sync

clean:
	rm -rf `find . -type d -name __pycache__`
	rm -rf .ruff_cache

check:
	uv run ruff check .

runner: check run clean