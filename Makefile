.PHONY: sync test lab lint clean

sync:
	uv sync

test:
	uv run pytest

lab:
	uv run jupyter lab

lint:
	uv run ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
