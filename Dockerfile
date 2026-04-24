FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-optional.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt -r requirements-optional.txt

COPY . .
RUN mkdir -p /app/outputs /app/outputs/plots /tmp/matplotlib

CMD ["python3", "main.py", "once", "--skip-extract"]
