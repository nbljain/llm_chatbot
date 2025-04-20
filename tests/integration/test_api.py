"""
Integration tests for API endpoints.
Tests the API functionality with a test client.
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add the project root directory to the path so we can import our modules
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)


def test_root_endpoint(test_client):
    """Test the root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SQL Chatbot API is running"}


def test_tables_endpoint(test_client):
    """Test the tables endpoint."""
    response = test_client.get("/tables")
    assert response.status_code == 200
    tables = response.json()["tables"]
    assert "employees" in tables


def test_schema_endpoint(test_client):
    """Test the schema endpoint for a specific table."""
    response = test_client.post("/schema", json={"table_name": "employees"})
    assert response.status_code == 200
    schema = response.json()["schema"]
    assert "employee_id" in schema
    assert "first_name" in schema
    assert "email" in schema


def test_all_schemas_endpoint(test_client):
    """Test the schema endpoint for all tables."""
    response = test_client.post("/schema", json={})
    assert response.status_code == 200
    schema = response.json()["schema"]
    assert "employees" in schema


def test_query_endpoint_with_simple_question(test_client, monkeypatch):
    """Test the query endpoint with a simple question."""

    # Mock the NLP functions to avoid calling OpenAI
    def mock_generate_sql_query(question):
        return "SELECT * FROM employees LIMIT 5"

    def mock_generate_answer(question, sql_query, query_results):
        return "Here are the first 5 employees."

    monkeypatch.setattr(
        "src.backend.api.generate_sql_query", mock_generate_sql_query
    )
    monkeypatch.setattr(
        "src.backend.api.generate_answer", mock_generate_answer
    )

    # Test the query endpoint
    response = test_client.post(
        "/query", json={"question": "Show me some employees"}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["sql"] == "SELECT * FROM employees LIMIT 5"
    assert result["explanation"] == "Here are the first 5 employees."
    assert "data" in result


def test_query_endpoint_with_error(test_client, monkeypatch):
    """Test the query endpoint with an invalid question."""

    # Mock the NLP function to return an invalid SQL query
    def mock_generate_sql_query(question):
        return "SELECT * FROM nonexistent_table"

    monkeypatch.setattr(
        "src.backend.api.generate_sql_query", mock_generate_sql_query
    )

    # Test the query endpoint
    response = test_client.post(
        "/query", json={"question": "Show me data from a nonexistent table"}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is False
    assert "error" in result
