FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip uv

COPY . /app
RUN uv sync --no-dev

# Production: no --reload. For dev, override CMD to add --reload.
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "auditrag.main:app", "--host", "0.0.0.0", "--port", "8000"]
