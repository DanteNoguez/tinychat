.PHONY: test run lint help all
PYTHON_FILES=tinychat/

all: help

run:
	@uv run python tinychat/examples/quickstart_pingpong.py

test:
	@uv run pytest tinychat/tests/ -v

lint:
	@echo "Checking format with ruff..."
	@uv run ruff format --check $(PYTHON_FILES)
	@echo "Linting with ruff..."
	@uv run ruff check $(PYTHON_FILES)

format:
	@echo "Formatting with ruff..."
	@uv run ruff format $(PYTHON_FILES)
	@echo "Fixing linting issues with ruff..."
	@uv run ruff check --fix $(PYTHON_FILES)

help:
	@echo '----'
	@echo 'lint                - run linters'
	@echo 'format              - format code with ruff'
	@echo 'test                - run tests'
	@echo 'run                 - run ping pong example'