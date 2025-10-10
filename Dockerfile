FROM python:3.10-slim

# --- Python runtime settings ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- System dependencies ---
# Add Java (required for Spark) + common build tools + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    openjdk-21-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Java environment variables ---
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# --- Python dependencies ---
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy project source ---
COPY src /app/src
COPY README.md /app/README.md

# --- Default command (keep container running in dev) ---
CMD ["tail", "-f", "/dev/null"]
