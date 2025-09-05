import pytest
from unittest.mock import patch, MagicMock
from src.database import db

@pytest.fixture
def mock_engine():
    with patch("src.database.db.get_engine") as mock_engine:
        yield mock_engine

@pytest.fixture
def mock_inspector():
    with patch("src.database.db.inspect") as mock_inspect:
        yield mock_inspect

def test_get_table_names_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users", "orders"]
    tables = db.get_table_names()
    assert tables == ["users", "orders"]

def test_get_table_schema_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users"]
    mock_inspector.return_value.get_columns.return_value = [
        {"name": "id", "type": MagicMock(__str__=lambda s: "INTEGER")},
        {"name": "name", "type": MagicMock(__str__=lambda s: "VARCHAR")},
    ]
    schema = db.get_table_schema("users")
    assert schema == {"id": "INTEGER", "name": "VARCHAR"}

def test_get_table_schema_table_not_found(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["orders"]
    schema = db.get_table_schema("users")
    assert schema == {}

def test_get_all_table_schemas_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users"]
    mock_inspector.return_value.get_columns.return_value = [
        {"name": "id", "type": MagicMock(__str__=lambda s: "INTEGER")}
    ]
    schemas = db.get_all_table_schemas()
    assert schemas == {"users": {"id": "INTEGER"}}

def test_execute_sql_query_success(mock_engine):
    mock_conn = MagicMock()
    mock_row = MagicMock()
    mock_row._mapping = {"id": 1, "name": "Alice"}

    mock_execute = MagicMock(return_value=[mock_row])
    mock_conn.connect.return_value.__enter__.return_value.execute = mock_execute
    mock_engine.return_value = mock_conn

    result = db.execute_sql_query("SELECT * FROM users")
    assert result["success"] is True
    assert result["data"] == [{"id": 1, "name": "Alice"}]

def test_execute_sql_query_failure(mock_engine):
    mock_conn = MagicMock()
    mock_conn.connect.return_value.__enter__.return_value.execute.side_effect = Exception("SQL Error")
    mock_engine.return_value = mock_conn

    result = db.execute_sql_query("SELECT * FROM users")
    assert result["success"] is False
    assert "SQL Error" in result["error"]
