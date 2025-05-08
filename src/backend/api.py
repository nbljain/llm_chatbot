import os
import sys
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.backend.auth import get_user_from_request
from src.backend.auth import router as auth_router
from src.backend.nlp import (
    generate_answer,
    generate_sql_query,
    update_conversation_context,
)
from src.database.db import (
    execute_sql_query,
    get_table_names,
    get_table_schema,
)
from src.utils.error_handlers import (
    AuthenticationError,
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
setup_error_handlers(
    app, debug_mode=os.environ.get("DEBUG", "false").lower() == "true"
)

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
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
            },
        )

        return response
    except Exception as e:
        # Log unhandled exception
        logger.exception(
            f"Unhandled exception in request {request_id}: {str(e)}",
            extra={"request_id": request_id},
        )
        raise


# Include the authentication router
app.include_router(auth_router)


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

    # Check if user is authenticated
    user_data = get_user_from_request(request)
    if not user_data:
        logger.warning(
            f"Unauthenticated request to /tables from {request.client.host}",
            extra={"request_id": request_id},
        )
        raise AuthenticationError(
            message="Authentication required to access database tables",
            error_code="AUTH_REQUIRED",
        )

    # Log the request
    logger.info(
        f"Getting table list for user {user_data.get('username')}",
        extra={"request_id": request_id, "user_id": user_data.get("id")},
    )

    try:
        tables = get_table_names()
        logger.debug(
            f"Retrieved {len(tables)} tables from database",
            extra={"request_id": request_id, "tables_count": len(tables)},
        )
        return {"tables": tables}
    except Exception as e:
        logger.error(
            f"Error retrieving table names: {str(e)}",
            extra={"request_id": request_id},
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

    # Check if user is authenticated
    user_data = get_user_from_request(req)
    if not user_data:
        logger.warning(
            f"Unauthenticated request to /schema from {req.client.host}",
            extra={"request_id": request_id},
        )
        raise AuthenticationError(
            message="Authentication required to access database schema",
            error_code="AUTH_REQUIRED",
        )

    # Log the request
    if request.table_name:
        logger.info(
            f"Getting schema for table {request.table_name} for user {user_data.get('username')}",
            extra={
                "request_id": request_id,
                "user_id": user_data.get("id"),
                "table_name": request.table_name,
            },
        )
    else:
        logger.info(
            f"Getting all table schemas for user {user_data.get('username')}",
            extra={"request_id": request_id, "user_id": user_data.get("id")},
        )

    try:
        if request.table_name:
            schema = get_table_schema(request.table_name)
            if not schema:
                logger.warning(
                    f"Table {request.table_name} not found",
                    extra={
                        "request_id": request_id,
                        "table_name": request.table_name,
                    },
                )
                raise ValidationError(
                    message=f"Table {request.table_name} not found",
                    error_code="TABLE_NOT_FOUND",
                    status_code=404,
                )

            logger.debug(
                f"Retrieved schema for table {request.table_name}",
                extra={
                    "request_id": request_id,
                    "table_name": request.table_name,
                },
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
                extra={
                    "request_id": request_id,
                    "tables_count": len(full_schema),
                },
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

    # Check if user is authenticated
    user_data = get_user_from_request(req)
    if not user_data:
        logger.warning(
            f"Unauthenticated request to /query from {req.client.host}",
            extra={"request_id": request_id},
        )
        raise AuthenticationError(
            message="Authentication required to execute queries",
            error_code="AUTH_REQUIRED",
        )

    # Validate request
    if not request.question or request.question.strip() == "":
        logger.warning(
            f"Empty question submitted by user {user_data.get('username')}",
            extra={"request_id": request_id, "user_id": user_data.get("id")},
        )
        raise ValidationError(
            message="Question cannot be empty", error_code="EMPTY_QUESTION"
        )

    # Get user ID for conversation memory
    user_id = str(user_data.get("id"))

    # Log the query request
    logger.info(
        f"Processing query from user {user_data.get('username')}: {request.question[:50]}...",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "question_length": len(request.question),
        },
    )

    try:
        # Generate SQL from natural language with user context for conversation memory
        logger.debug(
            f"Generating SQL for question: {request.question}",
            extra={"request_id": request_id, "user_id": user_id},
        )
        sql_result = generate_sql_query(request.question, user_id)

        if not sql_result["success"]:
            logger.warning(
                f"Failed to generate SQL: {sql_result.get('error', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "error": sql_result.get("error"),
                },
            )
            return {"success": False, "error": sql_result["error"]}

        # Log the generated SQL
        logger.info(
            f"Generated SQL query: {sql_result['sql']}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "sql_query": sql_result["sql"],
            },
        )

        # Execute the generated SQL
        logger.debug(
            f"Executing SQL query",
            extra={"request_id": request_id, "user_id": user_id},
        )
        query_result = execute_sql_query(sql_result["sql"])

        if not query_result["success"]:
            logger.warning(
                f"SQL execution error: {query_result.get('error', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "sql_query": sql_result["sql"],
                    "error": query_result.get("error"),
                },
            )
            return {
                "success": False,
                "sql": sql_result["sql"],
                "error": query_result["error"],
            }

        # Log query results summary
        result_count = len(query_result["data"]) if query_result["data"] else 0
        logger.info(
            f"SQL query returned {result_count} results",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "result_count": result_count,
            },
        )

        # Generate a natural language explanation of the results with user context
        logger.debug(
            f"Generating explanation for query results",
            extra={"request_id": request_id, "user_id": user_id},
        )
        explanation_result = generate_answer(
            question=request.question,
            sql_query=sql_result["sql"],
            query_results=query_result["data"],
            user_id=user_id,
        )

        # Return successful response with explanation
        response = {
            "success": True,
            "sql": sql_result["sql"],
            "data": query_result["data"],
        }

        # Add explanation if available
        if explanation_result["success"]:
            explanation = explanation_result["explanation"]
            response["explanation"] = explanation

            logger.debug(
                f"Generated explanation for query results",
                extra={"request_id": request_id, "user_id": user_id},
            )

            # Update conversation context with this interaction
            update_conversation_context(
                user_id=user_id,
                question=request.question,
                sql_query=sql_result["sql"],
                query_results=query_result["data"],
                explanation=explanation,
            )
        else:
            logger.warning(
                f"Failed to generate explanation: {explanation_result.get('error', 'Unknown error')}",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "error": explanation_result.get("error"),
                },
            )

        return response
    except Exception as e:
        logger.exception(
            f"Error processing query: {str(e)}",
            extra={"request_id": request_id, "user_id": user_id},
        )
        raise ProcessingError(
            message="Failed to process natural language query",
            error_code="QUERY_PROCESSING_ERROR",
            details=str(e),
        )

@app.post("/query_stream")
async def process_query_stream(request: QueryRequest, req: Request):
    """Process natural language query with streaming response"""
    request_id = getattr(req.state, "request_id", "unknown")
    
    # Validate request
    if not request.question or request.question.strip() == "":
        logger.warning(
            f"Empty question submitted",
            extra={"request_id": request_id}
        )
        raise ValidationError(
            message="Question cannot be empty",
            error_code="EMPTY_QUESTION"
        )

    # Use a generic user ID for conversation memory
    user_id = "default_user"
    
    # Log the query request
    logger.info(
        f"Processing streaming query: {request.question[:50]}...",
        extra={
            "request_id": request_id,
            "question_length": len(request.question)
        }
    )
    
    async def stream_response():
        try:
            # Initial stage - Starting
            yield json.dumps({
                "stage": "starting",
                "message": "Starting to process your query..."
            }) + "\n"
            
            await asyncio.sleep(0.2)  # Small delay for more natural streaming
            
            # Generate SQL from natural language
            yield json.dumps({
                "stage": "generating_sql",
                "message": "Translating your question to SQL..."
            }) + "\n"
            
            # Actual SQL generation
            sql_result = generate_sql_query(request.question, user_id)
            
            if not sql_result["success"]:
                yield json.dumps({
                    "stage": "error",
                    "message": "Failed to generate SQL",
                    "error": sql_result["error"],
                    "success": False
                }) + "\n"
                return
                
            # SQL Generated
            yield json.dumps({
                "stage": "sql_generated",
                "message": "SQL query generated successfully",
                "sql": sql_result["sql"]
            }) + "\n"
            
            await asyncio.sleep(0.2)  # Small delay
            
            # Executing SQL
            yield json.dumps({
                "stage": "executing_sql",
                "message": "Executing SQL query..."
            }) + "\n"
            
            # Execute the generated SQL
            query_result = execute_sql_query(sql_result["sql"])
            
            if not query_result["success"]:
                yield json.dumps({
                    "stage": "error",
                    "message": "Failed to execute SQL query",
                    "error": query_result["error"],
                    "sql": sql_result["sql"],
                    "success": False
                }) + "\n"
                return
                
            # Query execution successful
            yield json.dumps({
                "stage": "query_executed",
                "message": "Query executed successfully",
                "data": query_result["data"],
                "sql": sql_result["sql"]
            }) + "\n"
            
            await asyncio.sleep(0.2)  # Small delay
            
            # Generating explanation
            yield json.dumps({
                "stage": "generating_explanation",
                "message": "Generating explanation of results..."
            }) + "\n"
            
            # Generate a natural language explanation
            explanation_result = generate_answer(
                question=request.question,
                sql_query=sql_result["sql"],
                query_results=query_result["data"],
                user_id=user_id,
            )
            
            # If explanation is available, stream it in chunks for a more engaging experience
            if explanation_result["success"]:
                explanation = explanation_result["explanation"]
                
                # Split explanation into sentences to create natural pauses
                import re
                sentences = re.split(r'(?<=[.!?])\s+', explanation)
                
                # Stream each sentence with a short delay for a more natural typing effect
                accumulated_explanation = ""
                
                for i, sentence in enumerate(sentences):
                    accumulated_explanation += sentence + " "
                    
                    # Send the explanation chunk
                    yield json.dumps({
                        "stage": "explanation_chunk",
                        "message": "Generating explanation...",
                        "chunk": sentence,
                        "chunk_number": i + 1,
                        "total_chunks": len(sentences),
                        "accumulated_explanation": accumulated_explanation.strip(),
                    }) + "\n"
                    
                    # Add a small delay between chunks to simulate typing
                    await asyncio.sleep(0.3)
                
                # Update conversation context with this interaction
                update_conversation_context(
                    user_id=user_id,
                    question=request.question,
                    sql_query=sql_result["sql"],
                    query_results=query_result["data"],
                    explanation=explanation,
                )
            
            # Final response with everything
            final_response = {
                "stage": "complete",
                "message": "Processing complete",
                "success": True,
                "sql": sql_result["sql"],
                "data": query_result["data"]
            }
            
            # Add explanation to the final response
            if explanation_result["success"]:
                final_response["explanation"] = explanation_result["explanation"]
            
            yield json.dumps(final_response) + "\n"
            
        except Exception as e:
            # Handle any exceptions
            logger.exception(
                f"Error in streaming query: {str(e)}",
                extra={"request_id": request_id}
            )
            yield json.dumps({
                "stage": "error",
                "message": "An error occurred during processing",
                "error": str(e),
                "success": False
            }) + "\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson"
    )


# Run the API with Uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
