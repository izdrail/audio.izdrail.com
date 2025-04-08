FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Install all system dependencies in a single layer
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    gcc \
    g++ \
    gnupg \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    espeak-ng \
    libsndfile1-dev \
    supervisor && \
    curl -fsSL https://deb.nodesource.com/setup_current.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python3 --version && node --version && npm --version

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi[standard] fastapi-cli

# Copy application code and Supervisor config
COPY . .
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port
EXPOSE 1602

# Run Supervisor as the entry point
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]