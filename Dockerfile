FROM python:3.11-buster

ENV PYTHONUNBUFFERED=1

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root && rm -rf "$POETRY_CACHE_DIR"

COPY . .

RUN poetry install --without dev

CMD ["poetry", "run", "python3", "src/new_used/main.py"]
