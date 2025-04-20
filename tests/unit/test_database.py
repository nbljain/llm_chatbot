"""
Unit tests for database functionality.
Tests the basic database operations without requiring a full API server.
"""
import os
import sys
import pytest
import sqlite3

# Add the project root directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_table_names, get_table_schema, execute_sql_query
from src.utils.db_init import initialize_database


def test_get_table_names(test_db, monkeypatch):
    """Test retrieving table names from the database."""
    # Monkeypatch the database configuration to use our test database
    monkeypatch.setattr("src.database.db.DATABASE_URL", f"sqlite:///{test_db}")
    
    # Get table names
    tables = get_table_names()
    
    # Verify the employees table exists
    assert "employees" in tables


def test_get_table_schema(test_db, monkeypatch):
    """Test retrieving schema for a table."""
    # Monkeypatch the database configuration to use our test database
    monkeypatch.setattr("src.database.db.DATABASE_URL", f"sqlite:///{test_db}")
    
    # Get schema for employees table
    schema = get_table_schema("employees")
    
    # Verify the expected columns exist
    assert "employee_id" in schema
    assert "first_name" in schema
    assert "last_name" in schema
    assert "email" in schema
    assert "department" in schema
    assert "position" in schema
    assert "salary" in schema
    assert "hire_date" in schema


def test_execute_sql_query(test_db, monkeypatch):
    """Test executing an SQL query."""
    # Monkeypatch the database configuration to use our test database
    monkeypatch.setattr("src.database.db.DATABASE_URL", f"sqlite:///{test_db}")
    
    # Execute a simple query
    result = execute_sql_query("SELECT * FROM employees WHERE first_name = 'Test'")
    
    # Verify the query was successful
    assert result["success"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["first_name"] == "Test"
    assert result["data"][0]["last_name"] == "User"
    assert result["data"][0]["email"] == "test@example.com"


def test_execute_sql_query_error(test_db, monkeypatch):
    """Test executing an invalid SQL query."""
    # Monkeypatch the database configuration to use our test database
    monkeypatch.setattr("src.database.db.DATABASE_URL", f"sqlite:///{test_db}")
    
    # Execute an invalid query
    result = execute_sql_query("SELECT * FROM non_existent_table")
    
    # Verify the query failed
    assert result["success"] is False
    assert "error" in result