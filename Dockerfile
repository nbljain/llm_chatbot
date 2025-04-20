FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==1.6.1

# Copy poetry configuration files
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not use virtualenvs (we're already in a container)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the application
COPY . /app/

# Create a data directory for SQLite database
RUN mkdir -p /app/data && chmod 777 /app/data

# Ensure Python outputs are sent to terminal
ENV PYTHONUNBUFFERED=1

# Set environment variables
ENV FRONTEND_HOST=0.0.0.0
ENV FRONTEND_PORT=5000
ENV API_HOST=0.0.0.0 
ENV API_PORT=8000
ENV API_URL=http://localhost:8000

# Expose ports for Streamlit (5000) and FastAPI (8000)
EXPOSE 5000 8000

# Create entrypoint script
RUN echo '#!/bin/bash \n\
cd /app \n\
python main.py' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]