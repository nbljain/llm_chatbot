FROM python:3.11.5-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential gfortran linux-headers-amd64 && \
    apt-get clean

# Set Poetry installation path and add to PATH
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Poetry directly into POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 -

# Copy project files
COPY . /app/

# Disable virtualenv creation
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    FRONTEND_HOST=0.0.0.0 \
    FRONTEND_PORT=5000 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_URL=http://localhost:8000

EXPOSE 5000 8000

CMD ["poetry", "run", "python", "/app/main.py"]
