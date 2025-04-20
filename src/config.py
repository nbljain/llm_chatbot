"""
Configuration module for the SQL Chatbot application.
Centralizes all configuration settings and uses environment variables with sensible defaults.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Database configuration
DB_TYPE = os.environ.get("DB_TYPE", "sqlite")
DB_NAME = os.environ.get("DB_NAME", "sql_chatbot.db")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")  # Default PostgreSQL port
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# Construct database URL based on type
if DB_TYPE == "sqlite":
    DATABASE_URL = f"sqlite:///{DB_NAME}"
elif DB_TYPE == "postgresql":
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    raise ValueError(f"Unsupported database type: {DB_TYPE}")

# API configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_URL = os.environ.get("API_URL", f"http://127.0.0.1:{API_PORT}")

# Frontend configuration
FRONTEND_HOST = os.environ.get("FRONTEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "5000"))
STREAMLIT_SERVER_PORT = 5000  # Ensure Streamlit uses the correct port

# API endpoints
ENDPOINT_ROOT = ""
ENDPOINT_AUTH = "/auth"
ENDPOINT_TABLES = "/tables"
ENDPOINT_SCHEMA = "/schema"
ENDPOINT_QUERY = "/query"

# Authentication settings
AUTH_TOKEN_EXPIRY_MINUTES = int(os.environ.get("AUTH_TOKEN_EXPIRY_MINUTES", "60"))

# NLP settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
