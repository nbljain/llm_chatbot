import os
import time
import traceback
from functools import wraps
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

try:
    from databricks import sql as databricks_sql

    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

from src.config import (
    DATABASE_URL,
    DB_TYPE,
    DATABRICKS_SERVER_HOSTNAME,
    DATABRICKS_HTTP_PATH,
    DATABRICKS_TOKEN,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
)

from src.utils.error_handlers import DatabaseError
from src.utils.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Database connection with retry logic
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

engine = None
databricks_connection = None


def get_databricks_connection():
    """Get a Databricks SQL connection"""
    if not DATABRICKS_AVAILABLE:
        raise DatabaseError(
            message="Databricks SQL connector not available. Please install databricks-sql-connector.",
            error_code="DATABRICKS_NOT_AVAILABLE",
        )

    if not all([DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN]):
        raise DatabaseError(
            message="Databricks connection requires DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and DATABRICKS_TOKEN environment variables.",
            error_code="DATABRICKS_CONFIG_MISSING",
        )

    return databricks_sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


# Initialize database connection based on type
if DB_TYPE == "databricks":
    logger.info("Initializing Databricks connection...")
    for attempt in range(MAX_RETRIES):
        try:
            databricks_connection = get_databricks_connection()
            # Test the connection
            with databricks_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            logger.info(
                f"Successfully connected to Databricks at {DATABRICKS_SERVER_HOSTNAME}"
            )
            break
        except Exception as e:
            logger.warning(
                f"Databricks connection attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}",
                extra={"error": str(e), "attempt": attempt + 1},
            )
            if attempt == MAX_RETRIES - 1:
                logger.error(
                    f"Failed to connect to Databricks after {MAX_RETRIES} attempts: {str(e)}",
                    extra={"error": str(e), "traceback": traceback.format_exc()},
                )
                raise DatabaseError(
                    message=f"Failed to connect to Databricks: {str(e)}",
                    error_code="DATABRICKS_CONNECTION_ERROR",
                    details=traceback.format_exc(),
                )
            time.sleep(RETRY_DELAY)
else:
    # Initialize SQLite/PostgreSQL database connection
    for attempt in range(MAX_RETRIES):
        try:
            engine = create_engine(DATABASE_URL)
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_name = DATABASE_URL.split("/")[-1]
            logger.info(f"Successfully connected to database at {db_name}")
            break
        except SQLAlchemyError as e:
            logger.warning(
                f"Database connection attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}",
                extra={"error": str(e), "attempt": attempt + 1},
            )
            if attempt == MAX_RETRIES - 1:
                logger.error(
                    f"Failed to connect to database after {MAX_RETRIES} attempts: {str(e)}",
                    extra={"error": str(e), "traceback": traceback.format_exc()},
                )
                raise DatabaseError(
                    message=f"Failed to connect to database: {str(e)}",
                    error_code="DB_CONNECTION_ERROR",
                    details=traceback.format_exc(),
                )
            time.sleep(RETRY_DELAY)


# Decorator for database operations with error handling
def db_error_handler(func):
    """Decorator to handle database errors and provide consistent error logging"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = func.__name__
        try:
            return func(*args, **kwargs)
        except OperationalError as e:
            logger.error(
                f"Database operational error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e)},
            )
            if "no such table" in str(e).lower():
                raise DatabaseError(
                    message=f"Table not found: {str(e)}",
                    error_code="DB_TABLE_NOT_FOUND",
                    details=traceback.format_exc(),
                )
            elif "syntax error" in str(e).lower():
                raise DatabaseError(
                    message=f"SQL syntax error: {str(e)}",
                    error_code="DB_SYNTAX_ERROR",
                    details=traceback.format_exc(),
                )
            else:
                raise DatabaseError(
                    message=f"Database operation error: {str(e)}",
                    error_code="DB_OPERATIONAL_ERROR",
                    details=traceback.format_exc(),
                )
        except IntegrityError as e:
            logger.error(
                f"Database integrity error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e)},
            )
            raise DatabaseError(
                message=f"Data integrity error: {str(e)}",
                error_code="DB_INTEGRITY_ERROR",
                details=traceback.format_exc(),
            )
        except SQLAlchemyError as e:
            logger.error(
                f"SQLAlchemy error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e)},
            )
            raise DatabaseError(
                message=f"Database error: {str(e)}",
                error_code="DB_SQLALCHEMY_ERROR",
                details=traceback.format_exc(),
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in {operation_name}: {str(e)}",
                extra={
                    "operation": operation_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            raise DatabaseError(
                message=f"Unexpected database error: {str(e)}",
                error_code="DB_UNEXPECTED_ERROR",
                details=traceback.format_exc(),
            )

    return wrapper


@db_error_handler
def get_table_names() -> List[str]:
    """Get all table names from the database

    Returns:
        List[str]: List of table names from the database

    Raises:
        DatabaseError: If there's an error retrieving the tables
    """
    logger.debug("Getting table names from database")

    if DB_TYPE == "databricks":
        with databricks_connection.cursor() as cursor:
            # Query to get tables from the current catalog and schema
            cursor.execute(f"SHOW TABLES IN {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}")
            result = cursor.fetchall()
            # Databricks returns results as tuples, extract table names
            tables = [
                row[1] for row in result
            ]  # Table name is usually in the second column
    else:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

    logger.debug(f"Found {len(tables)} tables in database")
    return tables


@db_error_handler
def get_table_schema(table_name: str) -> Dict[str, str]:
    """Get schema for a specific table

    Args:
        table_name: Name of the table to get schema for

    Returns:
        Dict[str, str]: Dictionary mapping column names to their data types

    Raises:
        DatabaseError: If there's an error retrieving the schema
    """
    logger.debug(f"Getting schema for table: {table_name}")

    if DB_TYPE == "databricks":
        with databricks_connection.cursor() as cursor:
            # Check if table exists
            available_tables = get_table_names()
            if table_name not in available_tables:
                logger.warning(f"Table {table_name} not found in database")
                return {}

            # Get table schema using DESCRIBE command
            cursor.execute(
                f"DESCRIBE {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.{table_name}"
            )
            result = cursor.fetchall()

            # Parse the result to extract column names and types
            schema = {}
            for row in result:
                if (
                    len(row) >= 2 and row[0] and row[1]
                ):  # Ensure we have column name and type
                    column_name = row[0].strip()
                    column_type = row[1].strip()
                    # Skip partition information and other metadata
                    if not column_name.startswith("#") and column_name != "":
                        schema[column_name] = column_type
    else:
        inspector = inspect(engine)

        # Check if table exists
        if table_name not in inspector.get_table_names():
            logger.warning(f"Table {table_name} not found in database")
            return {}

        columns = inspector.get_columns(table_name)
        schema = {column["name"]: column["type"].__str__() for column in columns}

    logger.debug(f"Found {len(schema)} columns for table {table_name}")
    return schema


@db_error_handler
def get_all_table_schemas() -> Dict[str, Dict[str, str]]:
    """Get schemas for all tables in the database

    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping table names to their schemas

    Raises:
        DatabaseError: If there's an error retrieving the schemas
    """
    logger.debug("Getting schemas for all tables")
    table_schemas = {}
    tables = get_table_names()

    for table in tables:
        try:
            table_schemas[table] = get_table_schema(table)
        except Exception as e:
            logger.warning(f"Error getting schema for table {table}: {str(e)}")
            # Continue with other tables even if one fails
            table_schemas[table] = {}

    logger.debug(f"Retrieved schemas for {len(table_schemas)} tables")
    return table_schemas


@db_error_handler
def execute_sql_query(query: str) -> Dict[str, Any]:
    """Execute SQL query and return results

    Args:
        query: SQL query string to execute

    Returns:
        Dict containing:
            success (bool): Whether the query was successful
            data (List[Dict]): Query results if successful
            error (str): Error message if unsuccessful

    Raises:
        DatabaseError: If there's an unexpected error executing the query
    """
    # Log query with limited length to avoid logging huge queries
    max_log_length = 500
    safe_query = (
        query[:max_log_length] + "..." if len(query) > max_log_length else query
    )
    logger.info(f"Executing SQL query: {safe_query}")

    try:
        start_time = time.time()

        if DB_TYPE == "databricks":
            with databricks_connection.cursor() as cursor:
                # Execute the query
                cursor.execute(query)

                # Get column names from cursor description
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    # Fetch all results
                    result_rows = cursor.fetchall()

                    # Convert to list of dictionaries
                    rows = []
                    for row in result_rows:
                        row_dict = dict(zip(columns, row))
                        rows.append(row_dict)
                else:
                    # Query didn't return results (e.g., INSERT, UPDATE, DELETE)
                    rows = []
        else:
            with engine.connect() as connection:
                # Execute the query
                result = connection.execute(text(query))

                # Convert row objects to dictionaries
                rows = [dict(row._mapping) for row in result]

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log execution metrics
        logger.info(
            f"Query executed successfully in {execution_time:.2f}s with {len(rows)} results",
            extra={"execution_time": execution_time, "result_count": len(rows)},
        )

        return {"success": True, "data": rows}

    except Exception as e:
        error_message = str(e)
        logger.error(
            f"SQL execution error: {error_message}",
            extra={"query": safe_query, "error": error_message},
        )

        # Create a more user-friendly error message
        if (
            "syntax error" in error_message.lower()
            or "sqlexception" in error_message.lower()
        ):
            user_message = "SQL syntax error in the generated query. Please try rephrasing your question."
        elif "table" in error_message.lower() and (
            "not found" in error_message.lower()
            or "does not exist" in error_message.lower()
        ):
            user_message = "The query references a table that doesn't exist. Please check your question."
        elif "column" in error_message.lower() and (
            "not found" in error_message.lower()
            or "does not exist" in error_message.lower()
        ):
            user_message = "The query references a column that doesn't exist. Please check your question."
        else:
            user_message = f"Error executing SQL query: {error_message}"

        return {"success": False, "error": user_message}
