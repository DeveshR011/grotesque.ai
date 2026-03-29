FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Native build/runtime dependencies for common audio and ML wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libasound2-dev \
        libffi-dev \
        libportaudio2 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Exclude Windows-only packages when building on Linux containers.
RUN grep -Eiv '^(pywin32|WMI|PyAudioWPatch)==?' requirements.txt > requirements.docker.txt \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.docker.txt

COPY . .

CMD ["python", "main.py", "--config", "config/config.yaml"]
