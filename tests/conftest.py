"""
Configuration file for pytest.
This file contains fixtures and setup/teardown functions
that will be used for SQL Chatbot tests.
"""

import os
import sqlite3
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

# Add the project root directory to the path so we can import our modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.backend.api import app

# Import necessary modules for testing
from src.config import DB_NAME, DB_TYPE


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary database for testing."""
    # Create a temporary file
    db_file = tempfile.NamedTemporaryFile(delete=False)
    db_path = db_file.name
    db_file.close()

    # Set up the test database with some data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the test tables
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS employees (
        employee_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        department TEXT NOT NULL,
        position TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    )
    """
    )

    # Add some test data
    cursor.execute(
        """
    INSERT INTO employees (first_name, last_name, email, department, position, salary, hire_date)
    VALUES ('Test', 'User', 'test@example.com', 'IT', 'Tester', 75000.00, '2023-01-01')
    """
    )

    conn.commit()
    conn.close()

    # Return the path to the temporary database
    yield db_path

    # Clean up the temporary file after tests
    os.unlink(db_path)


@pytest.fixture
def test_client(monkeypatch, test_db):
    """Create a test client for the FastAPI application."""
    # Monkeypatch the database path to use our test database
    monkeypatch.setenv("DB_TYPE", "sqlite")
    monkeypatch.setenv("DB_NAME", test_db)

    # Create a test client using the FastAPI app
    client = TestClient(app)
    return client


@pytest.fixture
def mock_openai_response(monkeypatch):
    """Mock the OpenAI API response for testing."""

    # This fixture can be used to mock OpenAI responses without requiring an API key
    class MockResponse:
        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "SELECT * FROM employees;"}}
                ]
            }

    # Return the mock response object
    return MockResponse()
