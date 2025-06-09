"""
Unit tests for NLP processing module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.backend.nlp import (
    get_llm,
    get_table_schema_string,
    get_conversation_context_string,
    update_conversation_context,
    generate_sql_query,
    generate_answer,
    user_conversation_context,
)


class TestNLPFunctions:
    """Test cases for NLP processing functions."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear conversation context before each test
        user_conversation_context.clear()

    @patch("src.backend.nlp.os.environ.get")
    def test_get_llm_with_api_key(self, mock_env_get):
        """Test LLM initialization with valid API key."""
        mock_env_get.return_value = "test_api_key"

        with patch("src.backend.nlp.ChatOpenAI") as mock_chat_openai:
            mock_llm = Mock()
            mock_chat_openai.return_value = mock_llm

            result = get_llm()

            mock_chat_openai.assert_called_once_with(
                temperature=0, model="gpt-4o", api_key="test_api_key"
            )
            assert result == mock_llm

    @patch("src.backend.nlp.os.environ.get")
    def test_get_llm_without_api_key(self, mock_env_get):
        """Test LLM initialization without API key raises error."""
        mock_env_get.return_value = None

        with pytest.raises(
            ValueError, match="OPENAI_API_KEY environment variable not set"
        ):
            get_llm()

    @patch("src.backend.nlp.get_all_table_schemas")
    @patch("src.database.relationships.get_relationships_for_llm")
    def test_get_table_schema_string_with_tables(
        self, mock_get_relationships, mock_get_schemas
    ):
        """Test schema string generation with tables."""
        mock_get_schemas.return_value = {
            "users": {"id": "INTEGER", "name": "TEXT"},
            "orders": {"id": "INTEGER", "user_id": "INTEGER", "amount": "REAL"},
        }
        mock_get_relationships.return_value = (
            "Table relationships:\nusers.id = orders.user_id"
        )

        result = get_table_schema_string()

        assert "Database Schema:" in result
        assert "Table: users" in result
        assert "Table: orders" in result
        assert "id (INTEGER)" in result
        assert "name (TEXT)" in result
        assert "Table relationships:" in result

    @patch("src.backend.nlp.get_all_table_schemas")
    def test_get_table_schema_string_no_tables(self, mock_get_schemas):
        """Test schema string generation with no tables."""
        mock_get_schemas.return_value = {}

        result = get_table_schema_string()

        assert result == "No tables found in the database."

    def test_get_conversation_context_string_no_user(self):
        """Test conversation context retrieval with no user ID."""
        result = get_conversation_context_string(None)
        assert result == ""

    def test_get_conversation_context_string_new_user(self):
        """Test conversation context retrieval for new user."""
        result = get_conversation_context_string("new_user")
        assert result == ""

    def test_get_conversation_context_string_with_history(self):
        """Test conversation context retrieval with existing history."""
        user_id = "test_user"
        user_conversation_context[user_id] = [
            {
                "question": "Show me users",
                "sql": "SELECT * FROM users",
                "results": [{"id": 1, "name": "John"}],
                "explanation": "This query shows all users",
            }
        ]

        result = get_conversation_context_string(user_id)

        assert "Previous conversation history:" in result
        assert "Question 1: Show me users" in result
        assert "SQL: SELECT * FROM users" in result
        assert "Results:" in result

    def test_update_conversation_context_new_user(self):
        """Test updating conversation context for new user."""
        user_id = "new_user"
        question = "Show me users"
        sql_query = "SELECT * FROM users"
        query_results = [{"id": 1, "name": "John"}]
        explanation = "This shows all users"

        update_conversation_context(
            user_id, question, sql_query, query_results, explanation
        )

        assert user_id in user_conversation_context
        assert len(user_conversation_context[user_id]) == 1

        interaction = user_conversation_context[user_id][0]
        assert interaction["question"] == question
        assert interaction["sql"] == sql_query
        assert interaction["results"] == query_results
        assert interaction["explanation"] == explanation

    def test_update_conversation_context_limit(self):
        """Test conversation context limit enforcement."""
        user_id = "test_user"

        # Add 12 interactions (more than the 10 limit)
        for i in range(12):
            update_conversation_context(
                user_id,
                f"Question {i}",
                f"SQL {i}",
                [{"result": i}],
                f"Explanation {i}",
            )

        # Should only keep the last 10
        assert len(user_conversation_context[user_id]) == 10
        assert user_conversation_context[user_id][0]["question"] == "Question 2"
        assert user_conversation_context[user_id][-1]["question"] == "Question 11"

    @patch("src.backend.nlp.get_llm")
    @patch("src.backend.nlp.get_table_schema_string")
    @patch("src.backend.nlp.get_conversation_context_string")
    def test_generate_sql_query_success(
        self, mock_get_context, mock_get_schema, mock_get_llm
    ):
        """Test successful SQL query generation."""
        # Mock dependencies
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "SELECT * FROM users WHERE name = 'John'"
        mock_llm.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        mock_get_schema.return_value = "Schema info"
        mock_get_context.return_value = "Context info"

        result = generate_sql_query("Show me user John", "test_user")

        assert result == "SELECT * FROM users WHERE name = 'John'"
        mock_get_llm.assert_called_once()
        mock_get_schema.assert_called_once()
        mock_get_context.assert_called_once_with("test_user")

    @patch("src.backend.nlp.get_llm")
    @patch("src.backend.nlp.get_table_schema_string")
    @patch("src.backend.nlp.get_conversation_context_string")
    def test_generate_sql_query_with_markdown(
        self, mock_get_context, mock_get_schema, mock_get_llm
    ):
        """Test SQL query generation with markdown formatting."""
        # Mock dependencies
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "```sql\nSELECT * FROM users\n```"
        mock_llm.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        mock_get_schema.return_value = "Schema info"
        mock_get_context.return_value = "Context info"

        result = generate_sql_query("Show me users", "test_user")

        assert result == "SELECT * FROM users"

    @patch("src.backend.nlp.get_llm")
    @patch("src.backend.nlp.update_conversation_context")
    def test_generate_answer_success(self, mock_update_context, mock_get_llm):
        """Test successful answer generation."""
        # Mock dependencies
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "This query shows all users in the database."
        mock_llm.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        question = "Show me users"
        sql_query = "SELECT * FROM users"
        query_results = [{"id": 1, "name": "John"}]
        user_id = "test_user"

        result = generate_answer(question, sql_query, query_results, user_id)

        assert result == "This query shows all users in the database."
        mock_update_context.assert_called_once_with(
            user_id, question, sql_query, query_results, result
        )

    @patch("src.backend.nlp.get_llm")
    def test_generate_answer_large_results(self, mock_get_llm):
        """Test answer generation with large result set."""
        # Mock dependencies
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Results are truncated due to size."
        mock_llm.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        # Create large result set
        large_results = [{"id": i, "data": "x" * 100} for i in range(100)]

        result = generate_answer(
            "Show data", "SELECT * FROM large_table", large_results
        )

        assert result == "Results are truncated due to size."
        # Verify that the LLM was called with truncated data
        call_args = mock_llm.call_args[0][0]
        prompt_content = call_args[1].content
        assert "(truncated)" in prompt_content
