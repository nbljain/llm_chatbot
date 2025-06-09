import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import OperationalError, IntegrityError
from src.utils.error_handlers import DatabaseError
from src.database.db import (
    get_table_schema,
    get_all_table_schemas,
    execute_sql_query,
    get_table_names
)


@pytest.fixture
def mock_engine_connect():
    with patch("src.database.db.engine.connect") as mock_conn:
        yield mock_conn


@pytest.fixture
def mock_inspector():
    with patch("src.database.db.inspect") as mock_inspect:
        yield mock_inspect


def test_get_table_names_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users", "orders"]
    tables = get_table_names()
    assert tables == ["users", "orders"]


def test_get_table_schema_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users"]
    mock_inspector.return_value.get_columns.return_value = [
        {"name": "id", "type": MagicMock(__str__=lambda s: "INTEGER")},
        {"name": "name", "type": MagicMock(__str__=lambda s: "VARCHAR")},
    ]
    schema = get_table_schema("users")
    assert schema == {"id": "INTEGER", "name": "VARCHAR"}


def test_get_table_schema_table_not_found(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["orders"]
    schema = get_table_schema("users")
    assert schema == {}


def test_get_all_table_schemas_success(mock_inspector):
    mock_inspector.return_value.get_table_names.return_value = ["users"]
    mock_inspector.return_value.get_columns.return_value = [
        {"name": "id", "type": MagicMock(__str__=lambda s: "INTEGER")}
    ]
    schemas = get_all_table_schemas()
    assert schemas == {"users": {"id": "INTEGER"}}


# def test_execute_sql_query_success(mock_engine_connect):
#     mock_conn = MagicMock()

#     # Create a mock row with a _mapping attribute that is a dict
#     mock_row = MagicMock()
#     mock_row._mapping = {"id": 1, "name": "Alice"}

#     mock_execute = MagicMock(return_value=[mock_row])
#     mock_conn.return_value.__enter__.return_value.execute = mock_execute
#     mock_engine_connect.return_value = mock_conn

#     result = execute_sql_query("SELECT * FROM users")
#     assert result["success"] is True
#     assert result["data"] == [{"id": 1, "name": "Alice"}]
