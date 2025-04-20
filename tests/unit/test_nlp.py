"""
Unit tests for NLP functionality.
Tests SQL generation and response generation with mocked OpenAI responses.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the project root directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.backend.nlp import generate_answer, generate_sql_query, get_table_schema_string


def test_get_table_schema_string(monkeypatch):
    """Test getting the database schema as a string."""

    # Mock the get_all_table_schemas function
    def mock_get_all_table_schemas():
        return {
            "employees": {
                "employee_id": "INTEGER",
                "first_name": "TEXT",
                "last_name": "TEXT",
            },
            "projects": {
                "project_id": "INTEGER",
                "project_name": "TEXT",
                "description": "TEXT",
            },
        }

    monkeypatch.setattr(
        "src.backend.nlp.get_all_table_schemas", mock_get_all_table_schemas
    )

    # Get the schema string
    schema_str = get_table_schema_string()

    # Verify it contains the expected tables and columns
    assert "employees" in schema_str
    assert "projects" in schema_str
    assert "employee_id" in schema_str
    assert "project_name" in schema_str


@patch("src.backend.nlp.get_llm")
def test_generate_sql_query(mock_get_llm, monkeypatch):
    """Test generating SQL from a natural language question."""
    # Mock the LLM chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = (
        "SELECT * FROM employees WHERE department = 'Engineering'"
    )
    mock_get_llm.return_value = mock_chain

    # Mock the schema string function
    monkeypatch.setattr(
        "src.backend.nlp.get_table_schema_string", lambda: "Table schema here"
    )

    # Generate SQL
    sql = generate_sql_query("Show me engineers")

    # Verify the generated SQL
    assert sql == "SELECT * FROM employees WHERE department = 'Engineering'"

    # Verify the chain was called with the right arguments
    mock_chain.invoke.assert_called_once()
    args = mock_chain.invoke.call_args[0][0]
    assert "Show me engineers" in str(args)


@patch("src.backend.nlp.get_llm")
def test_generate_answer(mock_get_llm):
    """Test generating a natural language explanation of SQL results."""
    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "There are 3 engineers in the database."
    mock_get_llm.return_value = mock_llm

    # Sample data
    question = "How many engineers do we have?"
    sql_query = "SELECT * FROM employees WHERE department = 'Engineering'"
    query_results = [
        {"first_name": "John", "last_name": "Smith", "department": "Engineering"},
        {"first_name": "Michael", "last_name": "Williams", "department": "Engineering"},
        {"first_name": "Robert", "last_name": "Miller", "department": "Engineering"},
    ]

    # Generate answer
    answer = generate_answer(question, sql_query, query_results)

    # Verify the answer
    assert answer == "There are 3 engineers in the database."

    # Verify the LLM was called with the right arguments
    mock_llm.invoke.assert_called_once()
    args = mock_llm.invoke.call_args[0][0]
    assert "How many engineers do we have?" in str(args)
    assert "SELECT * FROM employees WHERE department = 'Engineering'" in str(args)
