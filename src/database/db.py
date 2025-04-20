import os
import time
import traceback
from functools import wraps
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from src.config import DATABASE_URL
from src.utils.db_init import initialize_database
from src.utils.error_handlers import DatabaseError
from src.utils.logging_config import get_logger

# Get logger
logger = get_logger(__name__)

# Initialize the database before connecting
initialize_database()

# Database connection with retry logic
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

engine = None

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
            extra={"error": str(e), "attempt": attempt + 1}
        )
        if attempt == MAX_RETRIES - 1:
            logger.error(
                f"Failed to connect to database after {MAX_RETRIES} attempts: {str(e)}",
                extra={"error": str(e), "traceback": traceback.format_exc()}
            )
            raise DatabaseError(
                message=f"Failed to connect to database: {str(e)}",
                error_code="DB_CONNECTION_ERROR",
                details=traceback.format_exc()
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
                extra={"operation": operation_name, "error": str(e)}
            )
            if "no such table" in str(e).lower():
                raise DatabaseError(
                    message=f"Table not found: {str(e)}",
                    error_code="DB_TABLE_NOT_FOUND",
                    details=traceback.format_exc()
                )
            elif "syntax error" in str(e).lower():
                raise DatabaseError(
                    message=f"SQL syntax error: {str(e)}",
                    error_code="DB_SYNTAX_ERROR",
                    details=traceback.format_exc()
                )
            else:
                raise DatabaseError(
                    message=f"Database operation error: {str(e)}",
                    error_code="DB_OPERATIONAL_ERROR",
                    details=traceback.format_exc()
                )
        except IntegrityError as e:
            logger.error(
                f"Database integrity error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e)}
            )
            raise DatabaseError(
                message=f"Data integrity error: {str(e)}",
                error_code="DB_INTEGRITY_ERROR",
                details=traceback.format_exc()
            )
        except SQLAlchemyError as e:
            logger.error(
                f"SQLAlchemy error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e)}
            )
            raise DatabaseError(
                message=f"Database error: {str(e)}",
                error_code="DB_SQLALCHEMY_ERROR",
                details=traceback.format_exc()
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in {operation_name}: {str(e)}",
                extra={"operation": operation_name, "error": str(e), "traceback": traceback.format_exc()}
            )
            raise DatabaseError(
                message=f"Unexpected database error: {str(e)}",
                error_code="DB_UNEXPECTED_ERROR",
                details=traceback.format_exc()
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
    safe_query = query[:max_log_length] + "..." if len(query) > max_log_length else query
    logger.info(f"Executing SQL query: {safe_query}")
    
    try:
        with engine.connect() as connection:
            # Execute the query
            start_time = time.time()
            result = connection.execute(text(query))
            
            # Convert row objects to dictionaries
            rows = [dict(row._mapping) for row in result]
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log execution metrics
            logger.info(
                f"Query executed successfully in {execution_time:.2f}s with {len(rows)} results",
                extra={"execution_time": execution_time, "result_count": len(rows)}
            )
            
            return {"success": True, "data": rows}
    except OperationalError as e:
        error_message = str(e)
        logger.error(
            f"SQL operational error: {error_message}",
            extra={"query": safe_query, "error": error_message}
        )
        
        # Create a more user-friendly error message
        if "syntax error" in error_message.lower():
            user_message = "SQL syntax error in the generated query. Please try rephrasing your question."
        elif "no such table" in error_message.lower():
            user_message = "The query references a table that doesn't exist. Please check your question."
        elif "no such column" in error_message.lower():
            user_message = "The query references a column that doesn't exist. Please check your question."
        else:
            user_message = f"Error executing SQL query: {error_message}"
            
        return {"success": False, "error": user_message}
    except SQLAlchemyError as e:
        error_message = str(e)
        logger.error(
            f"SQL alchemy error: {error_message}",
            extra={"query": safe_query, "error": error_message}
        )
        return {"success": False, "error": f"Database error: {error_message}"}
    except Exception as e:
        error_message = str(e)
        logger.exception(
            f"Unexpected error executing query: {error_message}",
            extra={"query": safe_query}
        )
        return {"success": False, "error": f"Unexpected error: {error_message}"}
