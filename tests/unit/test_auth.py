"""
Unit tests for authentication functionality.
Tests user registration, login, and password hashing.
"""
import os
import sys
import pytest
import sqlite3
import tempfile

# Add the project root directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.auth import hash_password, create_user_table, register_user, authenticate_user, update_user_password


@pytest.fixture
def auth_db():
    """Create a temporary database for auth testing."""
    # Create a temporary file
    db_file = tempfile.NamedTemporaryFile(delete=False)
    db_path = db_file.name
    db_file.close()
    
    # Connect to the database and create the users table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    
    # Return the path to the temporary database
    yield db_path
    
    # Clean up the temporary file after tests
    os.unlink(db_path)


def test_hash_password():
    """Test password hashing."""
    # Hash a password
    result = hash_password("test_password")
    
    # Verify the hash and salt are returned
    assert "password_hash" in result
    assert "password_salt" in result
    assert result["password_hash"] != "test_password"  # Make sure the password is hashed
    
    # Verify hashing with the same salt produces the same hash
    result2 = hash_password("test_password", result["password_salt"])
    assert result2["password_hash"] == result["password_hash"]
    
    # Verify hashing a different password with the same salt produces a different hash
    result3 = hash_password("different_password", result["password_salt"])
    assert result3["password_hash"] != result["password_hash"]


def test_register_user(auth_db, monkeypatch):
    """Test user registration."""
    # Monkeypatch the sqlite connection to use our test database
    def mock_connect(*args, **kwargs):
        return sqlite3.connect(auth_db)
    
    monkeypatch.setattr("sqlite3.connect", mock_connect)
    
    # Register a test user
    result = register_user("testuser", "test@example.com", "password123")
    
    # Verify registration was successful
    assert result["success"] is True
    assert "user" in result
    assert result["user"]["username"] == "testuser"
    assert result["user"]["email"] == "test@example.com"
    
    # Try to register the same user again (should fail)
    result2 = register_user("testuser", "another@example.com", "password456")
    assert result2["success"] is False
    assert "error" in result2
    
    # Try to register with the same email (should fail)
    result3 = register_user("anotheruser", "test@example.com", "password789")
    assert result3["success"] is False
    assert "error" in result3


def test_authenticate_user(auth_db, monkeypatch):
    """Test user authentication."""
    # Monkeypatch the sqlite connection to use our test database
    def mock_connect(*args, **kwargs):
        return sqlite3.connect(auth_db)
    
    monkeypatch.setattr("sqlite3.connect", mock_connect)
    
    # Register a test user
    register_user("authuser", "auth@example.com", "authpass123")
    
    # Test successful authentication
    result = authenticate_user("authuser", "authpass123")
    assert result["success"] is True
    assert "user" in result
    assert result["user"]["username"] == "authuser"
    assert result["user"]["email"] == "auth@example.com"
    
    # Test failed authentication with wrong password
    result2 = authenticate_user("authuser", "wrongpass")
    assert result2["success"] is False
    assert "error" in result2
    
    # Test failed authentication with nonexistent user
    result3 = authenticate_user("nonexistent", "authpass123")
    assert result3["success"] is False
    assert "error" in result3


def test_update_password(auth_db, monkeypatch):
    """Test password update."""
    # Monkeypatch the sqlite connection to use our test database
    def mock_connect(*args, **kwargs):
        return sqlite3.connect(auth_db)
    
    monkeypatch.setattr("sqlite3.connect", mock_connect)
    
    # Register a test user
    register_result = register_user("pwuser", "pw@example.com", "oldpass123")
    user_id = register_result["user"]["id"]
    
    # Test successful password update
    result = update_user_password(user_id, "oldpass123", "newpass456")
    assert result["success"] is True
    
    # Verify the new password works
    auth_result = authenticate_user("pwuser", "newpass456")
    assert auth_result["success"] is True
    
    # Verify the old password no longer works
    auth_result2 = authenticate_user("pwuser", "oldpass123")
    assert auth_result2["success"] is False
    
    # Test failed password update with wrong current password
    result2 = update_user_password(user_id, "wrongpass", "newpass789")
    assert result2["success"] is False
    assert "error" in result2