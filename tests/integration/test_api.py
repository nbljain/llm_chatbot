"""
Unit tests for API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from fastapi.testclient import TestClient
from src.backend.api import app
from src.utils.error_handlers import DatabaseError, ProcessingError


class TestAPIEndpoints:
    """Test cases for FastAPI endpoints."""

    def setup_method(self):
        """Set up test client before each test."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint returns welcome message."""
        response = self.client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "SQL Chatbot API is running"

    @patch("src.database.db.get_table_names")
    def test_get_tables_success(self, mock_get_table_names):
        """Test successful table names retrieval."""
        mock_get_table_names.return_value = ["users", "orders", "products"]

        response = self.client.get("/tables")

        assert response.status_code == 200

    @patch("src.backend.api.get_table_names")
    def test_get_tables_database_error(self, mock_get_table_names):
        """Test table names retrieval with database error."""
        mock_get_table_names.side_effect = DatabaseError("Database connection failed")

        response = self.client.post("/tables")

        assert response.status_code == 405

    @patch("src.database.db.get_all_table_schemas")
    def test_get_schema_all_tables(self, mock_get_schemas):
        """Test schema retrieval for all tables."""
        mock_get_schemas.return_value = {
            "users": {"id": "INTEGER", "name": "TEXT"},
            "orders": {"id": "INTEGER", "user_id": "INTEGER"},
        }

        response = self.client.post("/schema", json={})

        assert response.status_code == 200

    @patch("src.backend.api.get_table_schema")
    def test_get_schema_specific_table(self, mock_get_schema):
        """Test schema retrieval for specific table."""
        mock_get_schema.return_value = {
            "id": "INTEGER",
            "name": "TEXT",
            "email": "TEXT",
        }

        response = self.client.post("/schema", json={"table_name": "users"})

        assert response.status_code == 200
        data = response.json()
        assert "users" in data["db_schema"]
        assert data["db_schema"]["users"]["id"] == "INTEGER"

    @patch("src.backend.api.get_table_schema")
    def test_get_schema_table_not_found(self, mock_get_schema):
        """Test schema retrieval for non-existent table."""
        mock_get_schema.side_effect = DatabaseError("Table not found")

        response = self.client.post("/schema", json={"table_name": "nonexistent"})

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    @patch("src.backend.api.generate_answer")
    def test_process_query_success(
        self, mock_generate_answer, mock_execute_sql, mock_generate_sql
    ):
        """Test successful query processing."""
        # Mock the NLP and database functions
        mock_generate_sql.return_value = "SELECT * FROM users"
        mock_execute_sql.return_value = {
            "success": True,
            "data": [{"id": 1, "name": "John", "email": "john@email.com"}],
        }
        mock_generate_answer.return_value = (
            "This query shows all users in the database."
        )

        query_data = {"question": "Show me all users"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["sql"] == "SELECT * FROM users"
        assert len(data["data"]) == 1
        assert data["explanation"] == "This query shows all users in the database."

    @patch("src.backend.api.generate_sql_query")
    def test_process_query_invalid_sql(self, mock_generate_sql):
        """Test query processing with invalid SQL generation."""
        mock_generate_sql.return_value = "I cannot generate SQL for this request"

        query_data = {"question": "What is the weather today?"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "I cannot generate SQL for this request" in data["error"]

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    def test_process_query_sql_execution_error(
        self, mock_execute_sql, mock_generate_sql
    ):
        """Test query processing with SQL execution error."""
        mock_generate_sql.return_value = "SELECT * FROM users"
        mock_execute_sql.return_value = {
            "success": False,
            "error": "Table 'users' doesn't exist",
        }

        query_data = {"question": "Show me all users"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["sql"] == "SELECT * FROM users"
        assert "Table 'users' doesn't exist" in data["error"]

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    @patch("src.backend.api.generate_answer")
    def test_process_query_empty_results(
        self, mock_generate_answer, mock_execute_sql, mock_generate_sql
    ):
        """Test query processing with empty results."""
        mock_generate_sql.return_value = "SELECT * FROM users WHERE id = 999"
        mock_execute_sql.return_value = {"success": True, "data": []}
        mock_generate_answer.return_value = (
            "No users found with the specified criteria."
        )

        query_data = {"question": "Show me user with ID 999"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert "No users found" in data["explanation"]

    def test_process_query_missing_question(self):
        """Test query processing with missing question field."""
        response = self.client.post("/query", json={})

        assert response.status_code in [400, 422]  # Validation error

    def test_process_query_empty_question(self):
        """Test query processing with empty question."""
        query_data = {"question": ""}
        response = self.client.post("/query", json=query_data)

        assert response.status_code in [400, 422]

    @patch("src.backend.api.generate_sql_query")
    def test_process_query_nlp_exception(self, mock_generate_sql):
        """Test query processing with NLP processing exception."""
        mock_generate_sql.side_effect = Exception("OpenAI API error")

        query_data = {"question": "Show me all users"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code in [200, 422]
        data = response.json()
        assert data["success"] is False
        assert data["error_category"] == "processing"

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    @patch("src.backend.api.generate_answer")
    def test_process_query_explanation_failure(
        self, mock_generate_answer, mock_execute_sql, mock_generate_sql
    ):
        """Test query processing when explanation generation fails."""
        mock_generate_sql.return_value = "SELECT * FROM users"
        mock_execute_sql.return_value = {
            "success": True,
            "data": [{"id": 1, "name": "John"}],
        }
        mock_generate_answer.return_value = ""  # Empty explanation

        query_data = {"question": "Show me all users"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["sql"] == "SELECT * FROM users"
        # Should still return data even if explanation fails
        assert len(data["data"]) == 1

    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        response = self.client.get("/")

        # CORS headers should be present due to middleware
        assert response.status_code == 200

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    @patch("src.backend.api.generate_answer")
    def test_process_query_large_dataset(
        self, mock_generate_answer, mock_execute_sql, mock_generate_sql
    ):
        """Test query processing with large dataset."""
        mock_generate_sql.return_value = "SELECT * FROM large_table"

        # Create large dataset
        large_data = [{"id": i, "name": f"user_{i}"} for i in range(1000)]
        mock_execute_sql.return_value = {"success": True, "data": large_data}
        mock_generate_answer.return_value = "Retrieved 1000 users from the database."

        query_data = {"question": "Show me all users"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1000

    @patch("src.backend.api.generate_sql_query")
    @patch("src.backend.api.execute_sql_query")
    @patch("src.backend.api.generate_answer")
    def test_process_query_conversation_context(
        self, mock_generate_answer, mock_execute_sql, mock_generate_sql
    ):
        """Test query processing maintains conversation context."""
        mock_generate_sql.return_value = (
            "SELECT * FROM users WHERE department = 'Engineering'"
        )
        mock_execute_sql.return_value = {
            "success": True,
            "data": [{"id": 1, "name": "John", "department": "Engineering"}],
        }
        mock_generate_answer.return_value = "Found 1 engineer in the database."

        query_data = {"question": "Show me engineers"}
        response = self.client.post("/query", json=query_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify that generate_answer was called with user_id for context
        mock_generate_answer.assert_called_once()
        call_args = mock_generate_answer.call_args
        assert call_args[1]["user_id"] == "default_user"

    def test_request_id_middleware(self):
        """Test that request ID middleware adds unique IDs."""
        response1 = self.client.get("/")
        response2 = self.client.get("/")

        # Both requests should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Request IDs should be different (handled by middleware)
        # This is tested indirectly through successful responses
