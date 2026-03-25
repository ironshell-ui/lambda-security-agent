FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

COPY --chown=agent:agent src/ src/
COPY --chown=agent:agent requirements.txt ./

RUN pip install --user --no-cache-dir -r requirements.txt

ENV PATH="/home/agent/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=5s --timeout=3s --retries=15 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9009/.well-known/agent-card.json')" || exit 1

WORKDIR /home/agent/src
ENTRYPOINT ["python", "server.py", "--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
