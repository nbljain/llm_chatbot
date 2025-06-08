import pytest
from unittest.mock import patch, MagicMock

from src.backend.nlp import (
    generate_sql_query,
    generate_answer,
    update_conversation_context,
    get_conversation_context_string,
    user_conversation_context,
)

# Use a dummy user ID
USER_ID = "test_user"


@pytest.fixture
def mock_llm():
    with patch("src.backend.nlp.get_llm") as mock_get_llm:
        llm_instance = MagicMock()
        mock_get_llm.return_value = llm_instance
        yield llm_instance


@pytest.fixture
def mock_schema():
    with patch("src.backend.nlp.get_all_table_schemas") as mock_schema:
        mock_schema.return_value = {
            "employees": {"id": "INTEGER", "name": "TEXT"}
        }
        with patch("src.backend.nlp.get_relationships_for_llm") as mock_rels:
            mock_rels.return_value = ""
            yield


def test_generate_sql_query_success(mock_llm, mock_schema):
    mock_llm.return_value = MagicMock(content="SELECT * FROM employees;")
    sql = generate_sql_query("Show all employees", user_id=USER_ID)
    assert "SELECT * FROM employees" in sql


def test_generate_sql_query_with_code_block(mock_llm, mock_schema):
    mock_llm.return_value = MagicMock(content="```sql\nSELECT * FROM employees;\n```")
    sql = generate_sql_query("Show all employees", user_id=USER_ID)
    assert sql == "SELECT * FROM employees;"


def test_generate_sql_query_failure(mock_schema):
    with patch("src.backend.nlp.get_llm", side_effect=Exception("API Error")):
        sql = generate_sql_query("Show all employees", user_id=USER_ID)
        assert "Error generating SQL" in sql


def test_generate_answer_success(mock_llm):
    question = "Show all employees"
    sql_query = "SELECT * FROM employees;"
    results = [{"id": 1, "name": "Alice"}]

    mock_llm.return_value = MagicMock(content="This query returns all employees.")
    explanation = generate_answer(question, sql_query, results, user_id=USER_ID)
    assert "This query returns all employees" in explanation


def test_generate_answer_failure():
    with patch("src.backend.nlp.get_llm", side_effect=Exception("API Failure")):
        explanation = generate_answer(
            "Any question", "SELECT * FROM test;", [], user_id=USER_ID
        )
        assert "Error generating explanation" in explanation


def test_update_and_get_conversation_context():
    question = "How many employees?"
    sql = "SELECT COUNT(*) FROM employees;"
    results = [{"COUNT(*)": 42}]
    explanation = "There are 42 employees."

    update_conversation_context(USER_ID, question, sql, results, explanation)
    context = get_conversation_context_string(USER_ID)

    assert "Question 1" in context
    assert "SELECT COUNT(*)" in context
    assert "There are 42 employees" not in context  # explanation isn't in context
    assert USER_ID in user_conversation_context
