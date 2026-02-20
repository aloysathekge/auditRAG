FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY . /app
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "auditrag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
