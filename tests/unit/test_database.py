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
        {"name": "id", "type": MagicMock(__str__=lambda s: "INT
