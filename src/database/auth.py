import hashlib
import logging
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

# Import from the project
from src.config import DATABASE_URL, DB_NAME

# Set up direct SQLite connection for auth operations
if DATABASE_URL.startswith("sqlite"):
    # Extract the database file path from the SQLite connection string
    db_path = DB_NAME
else:
    # Fallback for non-SQLite databases
    raise ValueError("This implementation only supports SQLite databases")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Hash a password with a salt for secure storage

    Args:
        password: The user's plain text password
        salt: Optional salt, if None a new salt will be generated

    Returns:
        Dict containing the hashed password and salt
    """
    if salt is None:
        salt = secrets.token_hex(16)

    # Create a hash with the password and salt
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,  # Number of iterations
    ).hex()

    return {"password_hash": password_hash, "salt": salt}


def create_user_table():
    """Create the users table if it doesn't exist"""
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            # Create the users table
            cursor.execute(
                """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
            """
            )
            conn.commit()
            logger.info("Created users table")
        else:
            logger.info("Users table already exists")

        # Close the connection
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error creating users table: {e}")
        raise


def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Register a new user

    Args:
        username: The username for the new user
        email: The email address for the new user
        password: The password for the new user

    Returns:
        Dict containing success status and user data or error message
    """
    try:
        # Ensure the users table exists
        create_user_table()

        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Check if the username already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return {"success": False, "error": "Username already exists"}

            # Check if the email already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return {"success": False, "error": "Email already exists"}

            # Hash the password
            password_data = hash_password(password)

            # Insert the new user
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    password_data["password_hash"],
                    password_data["salt"],
                ),
            )

            # Commit the transaction
            conn.commit()

            # Get the user ID
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user_id = cursor.fetchone()[0]

            return {
                "success": True,
                "user": {"id": user_id, "username": username, "email": email},
            }
        except Exception as e:
            # Rollback in case of error
            conn.rollback()
            raise
        finally:
            # Close the connection
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        return {"success": False, "error": str(e)}


def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticate a user with username and password

    Args:
        username: The username to authenticate
        password: The password to authenticate

    Returns:
        Dict containing success status and user data or error message
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        try:
            # Get the user by username
            cursor.execute(
                """
                SELECT id, username, email, password_hash, salt
                FROM users
                WHERE username = ?
                """,
                (username,),
            )
            user = cursor.fetchone()

            if not user:
                return {"success": False, "error": "Invalid username or password"}

            # Check the password
            user_salt = user["salt"]
            stored_hash = user["password_hash"]

            # Hash the provided password with the stored salt
            password_data = hash_password(password, user_salt)

            if password_data["password_hash"] != stored_hash:
                return {"success": False, "error": "Invalid username or password"}

            # Update last login time
            cursor.execute(
                """
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user["id"],),
            )
            conn.commit()

            return {
                "success": True,
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                },
            }
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return {"success": False, "error": str(e)}


def update_user_password(
    user_id: int, current_password: str, new_password: str
) -> Dict[str, Any]:
    """Update a user's password

    Args:
        user_id: The ID of the user
        current_password: The current password
        new_password: The new password

    Returns:
        Dict containing success status or error message
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        try:
            # Get the current user data
            cursor.execute(
                """
                SELECT password_hash, salt
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            )
            user = cursor.fetchone()

            if not user:
                return {"success": False, "error": "User not found"}

            # Verify current password
            user_salt = user["salt"]
            stored_hash = user["password_hash"]

            # Hash the provided password with the stored salt
            password_data = hash_password(current_password, user_salt)

            if password_data["password_hash"] != stored_hash:
                return {"success": False, "error": "Current password is incorrect"}

            # Hash the new password
            new_password_data = hash_password(new_password)

            # Update the password
            cursor.execute(
                """
                UPDATE users
                SET password_hash = ?, salt = ?
                WHERE id = ?
                """,
                (
                    new_password_data["password_hash"],
                    new_password_data["salt"],
                    user_id,
                ),
            )
            conn.commit()

            return {"success": True}
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error updating user password: {e}")
        return {"success": False, "error": str(e)}
