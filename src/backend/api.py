import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.backend.nlp import (
    generate_answer,
    generate_sql_query,
    update_conversation_context,
)
from src.database.db import execute_sql_query, get_table_names, get_table_schema
from src.utils.error_handlers import (
    DatabaseError,
    ProcessingError,
    ValidationError,
    setup_error_handlers,
)
from src.utils.logging_config import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="SQL Chatbot API")

# Set up error handlers
setup_error_handlers(app, debug_mode=os.environ.get("DEBUG", "false").lower() == "true")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with appropriate domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Generate request ID
    request_id = str(uuid4())

    # Add request ID to request state
    request.state.request_id = request_id

    # Add basic request logging
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} from {client_host}",
        extra={"request_id": request_id},
    )

    # Process request
    try:
        # Call next middleware or route handler
        response = await call_next(request)

        # Add request ID header to response
        response.headers["X-Request-ID"] = request_id

        # Log response status
        logger.info(
            f"Response {request_id}: {response.status_code}",
            extra={"request_id": request_id, "status_code": response.status_code},
        )

        return response
    except Exception as e:
        # Log unhandled exception
        logger.exception(
            f"Unhandled exception in request {request_id}: {str(e)}",
            extra={"request_id": request_id},
        )
        raise


# Define request and response models
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    error: Optional[str] = None


class SchemaRequest(BaseModel):
    table_name: Optional[str] = None


class TableListResponse(BaseModel):
    tables: List[str]


class SchemaResponse(BaseModel):
    db_schema: Dict[str, Any]


# API endpoints
@app.get("/")
async def root():
    return {"message": "SQL Chatbot API is running"}


@app.get("/tables", response_model=TableListResponse)
async def get_tables(request: Request):
    """Get all table names from the database"""
    request_id = getattr(request.state, "request_id", "unknown")

    # Log the request
    logger.info(f"Getting table list", extra={"request_id": request_id})

    try:
        tables = get_table_names()
        logger.debug(
            f"Retrieved {len(tables)} tables from database",
            extra={"request_id": request_id, "tables_count": len(tables)},
        )
        return {"tables": tables}
    except Exception as e:
        logger.error(
            f"Error retrieving table names: {str(e)}", extra={"request_id": request_id}
        )
        raise DatabaseError(
            message="Failed to retrieve database tables",
            error_code="DB_TABLE_LIST_ERROR",
            details=str(e),
        )


@app.post("/schema", response_model=SchemaResponse)
async def get_schema(request: SchemaRequest, req: Request):
    """Get schema for a specific table or all tables"""
    request_id = getattr(req.state, "request_id", "unknown")

    # Log the request
    if request.table_name:
        logger.info(
            f"Getting schema for table {request.table_name}",
            extra={"request_id": request_id, "table_name": request.table_name},
        )
    else:
        logger.info(f"Getting all table schemas", extra={"request_id": request_id})

    try:
        if request.table_name:
            schema = get_table_schema(request.table_name)
            if not schema:
                logger.warning(
                    f"Table {request.table_name} not found",
                    extra={"request_id": request_id, "table_name": request.table_name},
                )
                raise ValidationError(
                    message=f"Table {request.table_name} not found",
                    error_code="TABLE_NOT_FOUND",
                    status_code=404,
                )

            logger.debug(
                f"Retrieved schema for table {request.table_name}",
                extra={"request_id": request_id, "table_name": request.table_name},
            )
            return {"db_schema": {request.table_name: schema}}
        else:
            # Get all table schemas
            tables = get_table_names()
            full_schema = {}

            for table in tables:
                try:
                    full_schema[table] = get_table_schema(table)
                except Exception as e:
                    logger.warning(
                        f"Error getting schema for table {table}: {str(e)}",
                        extra={"request_id": request_id, "table_name": table},
                    )
                    # Continue with other tables even if one fails

            logger.debug(
                f"Retrieved schemas for {len(full_schema)} tables",
                extra={"request_id": request_id, "tables_count": len(full_schema)},
            )
            return {"db_schema": full_schema}
    except ValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving database schema: {str(e)}",
            extra={"request_id": request_id},
        )
        raise DatabaseError(
            message="Failed to retrieve database schema",
            error_code="DB_SCHEMA_ERROR",
            details=str(e),
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, req: Request):
    """Process natural language query and return SQL results with explanation"""
    request_id = getattr(req.state, "request_id", "unknown")

    # Validate request
    if not request.question or request.question.strip() == "":
        logger.warning(f"Empty question submitted", extra={"request_id": request_id})
        raise ValidationError(
            message="Question cannot be empty", error_code="EMPTY_QUESTION"
        )

    # Use a generic user ID for conversation memory
    user_id = "default_user"

    # Log the query request
    logger.info(
        f"Processing query: {request.question[:50]}...",
        extra={"request_id": request_id, "question_length": len(request.question)},
    )

    try:
        # Generate SQL from natural language with user context for conversation memory
        logger.debug(
            f"Generating SQL for question: {request.question}",
            extra={"request_id": request_id},
        )
        generated_sql = generate_sql_query(request.question, user_id)

        # Check if the result is an error message
        if not generated_sql.upper().startswith(
            ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN")
        ):
            logger.warning(
                f"Failed to generate SQL: {generated_sql}",
                extra={"request_id": request_id, "error": generated_sql},
            )
            return {"success": False, "error": generated_sql}

        # Log the generated SQL
        logger.info(
            f"Generated SQL query: {generated_sql}",
            extra={"request_id": request_id, "sql_query": generated_sql},
        )

        # Execute the generated SQL
        logger.debug(f"Executing SQL query", extra={"request_id": request_id})
        query_result = execute_sql_query(generated_sql)

        if not query_result["success"]:
            logger.warning(
                f"SQL execution error: {query_result.get('error', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "sql_query": generated_sql,
                    "error": query_result.get("error"),
                },
            )
            return {
                "success": False,
                "sql": generated_sql,
                "error": query_result["error"],
            }

        # Log query results summary
        result_count = len(query_result["data"]) if query_result["data"] else 0
        logger.info(
            f"SQL query returned {result_count} results",
            extra={"request_id": request_id, "result_count": result_count},
        )

        # Generate a natural language explanation of the results with user context
        logger.debug(
            f"Generating explanation for query results",
            extra={"request_id": request_id},
        )
        explanation = generate_answer(
            question=request.question,
            sql_query=generated_sql,
            query_results=query_result["data"],
            user_id=user_id,
        )

        # Return successful response with explanation
        response = {"success": True, "sql": generated_sql, "data": query_result["data"]}

        # Add explanation if available
        if explanation:
            response["explanation"] = explanation

            logger.debug(
                f"Generated explanation for query results",
                extra={"request_id": request_id},
            )

            # Update conversation context with this interaction
            update_conversation_context(
                user_id=user_id,
                question=request.question,
                sql_query=generated_sql,
                query_results=query_result["data"],
                explanation=explanation,
            )
        else:
            logger.warning(
                f"Failed to generate explanation", extra={"request_id": request_id}
            )

        return response
    except Exception as e:
        logger.exception(
            f"Error processing query: {str(e)}", extra={"request_id": request_id}
        )
        raise ProcessingError(
            message="Failed to process natural language query",
            error_code="QUERY_PROCESSING_ERROR",
            details=str(e),
        )


# Run the API with Uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
