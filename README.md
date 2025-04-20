# SQL Chatbot

An intelligent SQL query translation and data exploration platform that transforms natural language questions into precise database queries and provides comprehensive data insights.

## Features

- Natural language to SQL translation
- Interactive data visualization
- User authentication system
- Automatic database schema analysis
- Multi-format data export

## Technology Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite/PostgreSQL
- **AI/ML**: LangChain with OpenAI integration
- **Visualization**: Plotly
- **Dependency Management**: Poetry

## Getting Started

### Prerequisites

- Python 3.9+
- Poetry package manager

### Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

### Configuration

The application uses environment variables for configuration. These can be set in your environment or in a `.env` file:

```
# Database settings
DB_TYPE=sqlite  # or postgresql
DB_NAME=sql_chatbot.db
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Frontend settings
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5000

# OpenAI API key
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application

Start the application with:

```bash
poetry run start
```

Or run each component separately:

```bash
# Start the backend
poetry run uvicorn src.backend.api:app --host 0.0.0.0 --port 8000

# Start the frontend
poetry run streamlit run src.frontend.app.py
```

## Usage

1. Access the application at `http://localhost:5000`
2. Create an account or login
3. Ask natural language questions about your database
4. Explore the data visualizations and generated SQL

## Testing

The project includes both unit and integration tests. You can run the tests using the following commands:

```bash
# Install development dependencies
poetry install

# Run all tests
poetry run pytest

# Run only unit tests
poetry run pytest tests/unit

# Run only integration tests
poetry run pytest tests/integration

# Run tests with coverage report
poetry run pytest --cov=src --cov-report=html
```

You can also use the included Makefile for common development tasks:

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Generate coverage report
make coverage

# Run linting checks
make lint

# Format code
make format
```

## Project Structure

```
sql-chatbot/
├── .gitlab-ci.yml       # GitLab CI/CD configuration
├── .streamlit/          # Streamlit configuration
│   └── config.toml      # Streamlit server settings
├── Dockerfile           # Docker container configuration
├── Makefile             # Build and testing automation
├── README.md            # Project documentation
├── main.py              # Application entry point
├── pyproject.toml       # Poetry dependencies and configurations
├── pytest.ini           # Pytest configuration
├── src/                 # Source code
│   ├── backend/         # FastAPI backend
│   │   ├── api.py       # API endpoints
│   │   ├── auth.py      # Authentication
│   │   └── nlp.py       # NLP/LLM integration
│   ├── config.py        # Centralized application configuration
│   ├── database/        # Database modules
│   │   ├── auth.py      # Authentication database functions
│   │   └── db.py        # Database connections and queries
│   ├── frontend/        # Streamlit frontend
│   │   ├── app.py       # Main Streamlit application
│   │   └── auth.py      # Authentication UI
│   └── utils/           # Utility functions
│       └── db_init.py   # Database initialization
└── tests/               # Test suite
    ├── conftest.py      # Test fixtures and configuration
    ├── integration/     # Integration tests
    │   └── test_api.py  # API endpoint tests
    └── unit/            # Unit tests
        ├── test_auth.py # Authentication tests
        ├── test_database.py # Database tests
        └── test_nlp.py  # NLP functionality tests
```

## Continuous Integration / Continuous Deployment

The project includes a GitLab CI/CD configuration in `.gitlab-ci.yml` that sets up an automated pipeline with the following stages:

1. **Lint**: Runs code formatting and style checks using Black, isort, and Flake8.
2. **Test**: Executes unit and integration tests with pytest and collects code coverage metrics.
3. **Build**: Creates a Docker image for the application and pushes it to the GitLab Container Registry.
4. **Deploy**: Deploys the application to staging or production environments (manual trigger).

The pipeline automatically runs when:
- Creating a merge request
- Pushing to the main branch
- Tagging a release

### Docker Deployment

The application can be containerized using the included Dockerfile:

```bash
# Build the Docker image
docker build -t sql-chatbot .

# Run the container
docker run -p 5000:5000 -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key \
  sql-chatbot
```

## License

[MIT License](LICENSE)