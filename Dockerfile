FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY agentflow/ agentflow/
RUN pip install --no-cache-dir .

FROM python:3.12-slim
RUN addgroup --system --gid 1000 appuser && \
    adduser --system --uid 1000 --gid 1000 appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/agentflow agentflow/
COPY --from=builder /app/pyproject.toml .

RUN mkdir -p /app/logs /app/data /app/uploads /app/knowledge_files /app/outputs && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "agentflow.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
