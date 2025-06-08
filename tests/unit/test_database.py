"""
Unit tests for database operations module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sqlite3
from src.database.db import (
    get_table_names,
    get_table_schema,
    get_all_table_schemas,
    execute_sql_query,
    db_error_handler
)
from src.utils.error_handlers import DatabaseError


class TestDatabaseFunctions:
    """Test cases for database operation functions."""

    @patch('src.database.db.create_engine')
    def test_get_table_names_success(self, mock_create_engine):
        """Test successful retrieval of table names."""
        # Mock database connection and results
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [('users',), ('orders',), ('products',)]
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        result = get_table_names()
        
        assert result == ['users', 'orders', 'products']
        mock_connection.execute.assert_called_once()

    @patch('src.database.db.create_engine')
    def test_get_table_names_exception(self, mock_create_engine):
        """Test table names retrieval with database exception."""
        mock_create_engine.side_effect = Exception("Database connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            get_table_names()
        
        assert "Error retrieving table names" in str(exc_info.value)

    @patch('src.database.db.create_engine')
    def test_get_table_schema_success(self, mock_create_engine):
        """Test successful retrieval of table schema."""
        # Mock database connection and results
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ('id', 'INTEGER', 0, None, 1),
            ('name', 'TEXT', 0, None, 0),
            ('email', 'TEXT', 0, None, 0)
        ]
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        result = get_table_schema('users')
        
        expected = {
            'id': 'INTEGER',
            'name': 'TEXT',
            'email': 'TEXT'
        }
        assert result == expected
        mock_connection.execute.assert_called_once()

    @patch('src.database.db.create_engine')
    def test_get_table_schema_exception(self, mock_create_engine):
        """Test table schema retrieval with database exception."""
        mock_create_engine.side_effect = Exception("Database error")
        
        with pytest.raises(DatabaseError) as exc_info:
            get_table_schema('users')
        
        assert "Error retrieving schema for table" in str(exc_info.value)

    @patch('src.database.db.get_table_names')
    @patch('src.database.db.get_table_schema')
    def test_get_all_table_schemas_success(self, mock_get_schema, mock_get_names):
        """Test successful retrieval of all table schemas."""
        mock_get_names.return_value = ['users', 'orders']
        mock_get_schema.side_effect = [
            {'id': 'INTEGER', 'name': 'TEXT'},
            {'id': 'INTEGER', 'user_id': 'INTEGER', 'amount': 'REAL'}
        ]
        
        result = get_all_table_schemas()
        
        expected = {
            'users': {'id': 'INTEGER', 'name': 'TEXT'},
            'orders': {'id': 'INTEGER', 'user_id': 'INTEGER', 'amount': 'REAL'}
        }
        assert result == expected
        assert mock_get_schema.call_count == 2

    @patch('src.database.db.get_table_names')
    def test_get_all_table_schemas_exception(self, mock_get_names):
        """Test all table schemas retrieval with exception."""
        mock_get_names.side_effect = DatabaseError("Table names error")
        
        with pytest.raises(DatabaseError):
            get_all_table_schemas()

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_success(self, mock_create_engine):
        """Test successful SQL query execution."""
        # Mock database connection and results
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            (1, 'John', 'john@email.com'),
            (2, 'Jane', 'jane@email.com')
        ]
        mock_result.keys.return_value = ['id', 'name', 'email']
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        query = "SELECT id, name, email FROM users"
        result = execute_sql_query(query)
        
        expected = {
            'success': True,
            'data': [
                {'id': 1, 'name': 'John', 'email': 'john@email.com'},
                {'id': 2, 'name': 'Jane', 'email': 'jane@email.com'}
            ]
        }
        assert result == expected

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_no_results(self, mock_create_engine):
        """Test SQL query execution with no results."""
        # Mock database connection with empty results
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = ['id', 'name']
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        query = "SELECT * FROM users WHERE id = 999"
        result = execute_sql_query(query)
        
        expected = {
            'success': True,
            'data': []
        }
        assert result == expected

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_syntax_error(self, mock_create_engine):
        """Test SQL query execution with syntax error."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_connection.execute.side_effect = sqlite3.OperationalError("syntax error")
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        query = "SELCT * FROM users"  # Invalid SQL
        result = execute_sql_query(query)
        
        assert result['success'] is False
        assert 'syntax error' in result['error']

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_connection_error(self, mock_create_engine):
        """Test SQL query execution with connection error."""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        query = "SELECT * FROM users"
        result = execute_sql_query(query)
        
        assert result['success'] is False
        assert 'Connection failed' in result['error']

    def test_db_error_handler_decorator_success(self):
        """Test database error handler decorator with successful function."""
        @db_error_handler
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"

    def test_db_error_handler_decorator_exception(self):
        """Test database error handler decorator with exception."""
        @db_error_handler
        def failing_function():
            raise Exception("Test error")
        
        with pytest.raises(DatabaseError) as exc_info:
            failing_function()
        
        assert "Database operation failed" in str(exc_info.value)

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_large_result_set(self, mock_create_engine):
        """Test SQL query execution with large result set."""
        # Mock database connection with large results
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        
        # Create large dataset
        large_data = [(i, f'user_{i}', f'user_{i}@email.com') for i in range(1000)]
        mock_result.fetchall.return_value = large_data
        mock_result.keys.return_value = ['id', 'name', 'email']
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        query = "SELECT * FROM users"
        result = execute_sql_query(query)
        
        assert result['success'] is True
        assert len(result['data']) == 1000
        assert result['data'][0] == {'id': 1, 'name': 'user_1', 'email': 'user_1@email.com'}

    @patch('src.database.db.create_engine')
    def test_execute_sql_query_insert_operation(self, mock_create_engine):
        """Test SQL query execution for INSERT operation."""
        # Mock database connection for INSERT
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []
        mock_result.rowcount = 1
        
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_create_engine.return_value = mock_engine
        
        query = "INSERT INTO users (name, email) VALUES ('Test', 'test@email.com')"
        result = execute_sql_query(query)
        
        assert result['success'] is True
        assert result['data'] == []

    @patch('src.database.db.DB_TYPE', 'databricks')
    @patch('src.database.db.get_databricks_connection')
    def test_databricks_connection_success(self, mock_get_databricks_conn):
        """Test successful Databricks connection."""
        # Mock Databricks connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('table1',), ('table2',)]
        mock_connection.cursor.return_value = mock_cursor
        mock_get_databricks_conn.return_value = mock_connection
        
        result = get_table_names()
        
        assert result == ['table1', 'table2']
        mock_get_databricks_conn.assert_called_once()

    @patch('src.database.db.DB_TYPE', 'databricks')
    @patch('src.database.db.get_databricks_connection')
    def test_databricks_connection_failure(self, mock_get_databricks_conn):
        """Test Databricks connection failure."""
        mock_get_databricks_conn.side_effect = Exception("Databricks connection failed")
        
        with pytest.raises(DatabaseError) as exc_info:
            get_table_names()
        
        assert "Error retrieving table names" in str(exc_info.value)
