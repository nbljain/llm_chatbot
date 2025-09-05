import os
import time
import traceback
from functools import wraps
from typing import Any, Dict, List

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from src.utils.error_handlers import DatabaseError
from src.utils.logging_config import get_logger
from azure.identity import DefaultAzureCredential
import urllib

# 🔹 Azure SQL details (no password needed)
server = os.getenv("AZURE_SQL_SERVER")
database = os.getenv("AZURE_SQL_DATABASE")

# 🔹 Acquire token for Azure SQL
credential = DefaultAzureCredential()
token = credential.get_token("https://database.windows.net/.default")

access_token = token.token.encode("utf-16-le")
quoted = urllib.parse.quote_plus(
    f"Driver={{ODBC Driver 17 for SQL Server}};"
    f"Server={server};Database={database};"
    f"Authentication=ActiveDirectoryAccessToken"
)

# Create SQLAlchemy engine
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={quoted}",
    connect_args={"attrs_before": {1256: access_token}},  # 1256 = SQL_COPT_SS_ACCESS_TOKEN
    fast_executemany=True
)

# Logger
logger = get_logger(__name__)

# Retry logic
MAX_RETRIES = 3
RETRY_DELAY = 2

for attempt in range(MAX_RETRIES):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"✅ Connected to Azure SQL Database: {database}")
        break
    except SQLAlchemyError as e:
        logger.warning(
            f"Attempt {attempt+1}/{MAX_RETRIES} failed: {str(e)}",
            extra={"error": str(e), "attempt": attempt + 1},
        )
        if attempt == MAX_RETRIES - 1:
            logger.error(
                f"❌ Could not connect after {MAX_RETRIES} attempts",
                extra={"traceback": traceback.format_exc()},
            )
            raise DatabaseError(
                message=f"Failed to connect to Azure SQL Database: {str(e)}",
                error_code="AZURE_SQL_CONNECTION_ERROR",
                details=traceback.format_exc(),
            )
        time.sleep(RETRY_DELAY)


# === Your decorators & DB functions remain same === #
