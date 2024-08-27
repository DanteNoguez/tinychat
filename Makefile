.PHONY: test run lint help all
PYTHON_FILES=tinychat/

all: help

run:
	@poetry run uvicorn tinychat.examples.kavak_ai.main:app --reload --port 8081

quickstart:
	@poetry run python tinychat/examples/quickstarts/single_chat.py

test:
	@poetry run pytest tests

lint:
	echo "\nLinting with isort...\n" && \
	poetry run isort --check $(PYTHON_FILES) && \
	echo "\nLinting with black...\n" && \
	poetry run black --check $(PYTHON_FILES)
	# echo "\nLinting with mypy...\n" && \
	# poetry run mypy -p tinychat

format:
	@echo "Formatting with isort..."
	@poetry run isort $(PYTHON_FILES)
	@echo "Formatting with black..."
	@poetry run black $(PYTHON_FILES)

kavak-run:
	@echo "Building Docker image..."
	docker build -t kavak-chat -f tinychat/examples/kavak_ai/Dockerfile .
	@echo "Running Docker container..."
	docker run -p 8081:8081 kavak-chat

kavak-compose:
	@echo "Running docker compose..."
	docker-compose -f tinychat/examples/kavak_ai/docker-compose.yml up --build

kavak-compose-down:
	@echo "Stopping and removing Docker Compose containers..."
	docker-compose -f tinychat/examples/kavak_ai/docker-compose.yml down

help:
	@echo '----'
	@echo 'lint                - run linters'
	@echo 'format              - format code with black and isort'
	@echo 'test                - run tests'
	@echo 'run                 - run example app'
	@echo 'kavak-run          - build and run Docker container'