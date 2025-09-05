import os
import pyodbc
from sqlalchemy import create_engine, inspect, text
from azure.identity import DefaultAzureCredential
from functools import lru_cache

AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")


ODBC_DRIVER = "{ODBC Driver 17 for SQL Server}"
ODBC_CONNECTION_STRING = (
    f"Driver={ODBC_DRIVER};"
    f"Server={AZURE_SQL_SERVER};"
    f"Database={AZURE_SQL_DATABASE};"
)

@lru_cache
def get_pyodbc_connection():
    """Create a PyODBC connection with Azure AD access token."""
    credential = DefaultAzureCredential()
    token = credential.get_token("https://database.windows.net/.default")

    conn = pyodbc.connect(
        ODBC_CONNECTION_STRING,
        attrs_before={"AccessToken": token.token}
    )
    return conn

@lru_cache
def get_engine():
    """Create a SQLAlchemy engine using pyodbc + Azure AD token."""
    conn = get_pyodbc_connection()
    # Use the active connection inside SQLAlchemy
    engine = create_engine("mssql+pyodbc://", creator=lambda: conn)
    return engine

def get_table_names():
    inspector = inspect(get_engine())
    return inspector.get_table_names()

def get_table_schema(table_name: str):
    inspector = inspect(get_engine())
    if table_name not in inspector.get_table_names():
        return {}
    columns = inspector.get_columns(table_name)
    return {col["name"]: str(col["type"]) for col in columns}

def get_all_table_schemas():
    inspector = inspect(get_engine())
    schemas = {}
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        schemas[table] = {col["name"]: str(col["type"]) for col in columns}
    return schemas

def execute_sql_query(query: str):
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text(query))
            rows = [dict(row._mapping) for row in result]
            return {"success": True, "data": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}
