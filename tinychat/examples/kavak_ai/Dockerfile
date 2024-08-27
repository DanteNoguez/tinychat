FROM python:3.11-bullseye

RUN apt-get -y update
RUN apt-get -y upgrade

WORKDIR /app

COPY ../../../pyproject.toml ../../../poetry.lock ./
COPY ../../../.env ./

RUN pip install --no-cache-dir --upgrade poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi -vvv --all-extras

COPY ../../../tinychat ./tinychat

WORKDIR /app/tinychat/examples/kavak_ai

ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["gunicorn"]
CMD ["-k", "uvicorn.workers.UvicornWorker", "tinychat.examples.kavak_ai.main:app", "-w", "1", "--bind", "0.0.0.0:8081"]