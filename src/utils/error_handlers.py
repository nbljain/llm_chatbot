"""
Error handling utilities for SQL Chatbot application.
Provides custom exceptions, error handling middleware, and error response formats.
"""

import logging
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for better client-side handling"""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    
    success: bool = False
    error: str
    error_details: Optional[str] = None
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    error_code: Optional[str] = None
    status_code: int = 400
    validation_errors: Optional[List[Dict[str, Any]]] = None


class SQLChatbotError(Exception):
    """Base exception class for SQL Chatbot application"""
    
    def __init__(
        self,
        message: str,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        error_code: Optional[str] = None,
        status_code: int = 400,
        details: Optional[str] = None,
    ):
        self.message = message
        self.error_category = error_category
        self.error_code = error_code
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class DatabaseError(SQLChatbotError):
    """Exception for database-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.DATABASE,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class AuthenticationError(SQLChatbotError):
    """Exception for authentication-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 401,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.AUTHENTICATION,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class AuthorizationError(SQLChatbotError):
    """Exception for authorization-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 403,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.AUTHORIZATION,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ProcessingError(SQLChatbotError):
    """Exception for errors during query processing"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 422,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.PROCESSING,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ExternalServiceError(SQLChatbotError):
    """Exception for errors in external services (like OpenAI)"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 502,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.EXTERNAL_SERVICE,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ValidationError(SQLChatbotError):
    """Exception for validation errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 400,
        details: Optional[str] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ):
        self.validation_errors = validation_errors
        super().__init__(
            message,
            error_category=ErrorCategory.VALIDATION,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


class ConfigurationError(SQLChatbotError):
    """Exception for configuration-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[str] = None,
    ):
        super().__init__(
            message,
            error_category=ErrorCategory.CONFIGURATION,
            error_code=error_code,
            status_code=status_code,
            details=details,
        )


def format_error_response(
    exception: Union[SQLChatbotError, Exception],
    request_id: Optional[str] = None,
    include_traceback: bool = False,
) -> Dict[str, Any]:
    """Format an exception into a standardized error response
    
    Args:
        exception: The exception to format
        request_id: Optional request ID for tracing
        include_traceback: Whether to include traceback in detail (debug only)
    
    Returns:
        Dict containing standardized error response
    """
    # Get exception details
    error_category = getattr(exception, "error_category", ErrorCategory.UNKNOWN)
    error_code = getattr(exception, "error_code", None)
    status_code = getattr(exception, "status_code", 500)
    details = getattr(exception, "details", None)
    validation_errors = getattr(exception, "validation_errors", None)
    
    # Get traceback if requested
    if include_traceback:
        tb = traceback.format_exc()
        if details:
            details = f"{details}\n\nTraceback:\n{tb}"
        else:
            details = f"Traceback:\n{tb}"
    
    # Construct error response
    response = {
        "success": False,
        "error": str(exception),
        "error_details": details,
        "error_category": error_category,
        "error_code": error_code,
        "status_code": status_code,
    }
    
    # Add validation errors if present
    if validation_errors:
        response["validation_errors"] = validation_errors
    
    # Add request ID if provided
    if request_id:
        response["request_id"] = request_id
    
    return response


def setup_error_handlers(app: FastAPI, debug_mode: bool = False):
    """Set up FastAPI exception handlers
    
    Args:
        app: FastAPI application instance
        debug_mode: Whether to include detailed error information
    """
    
    @app.exception_handler(SQLChatbotError)
    async def sqlchatbot_exception_handler(request: Request, exc: SQLChatbotError):
        """Handle custom SQL Chatbot exceptions"""
        # Get request details for logging
        client_host = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        method = request.method
        
        # Create request ID from client info
        request_id = f"{client_host}:{endpoint}:{method}"
        
        # Log the error with context
        log_message = (
            f"Error handling request {request_id} - "
            f"{exc.error_category}: {str(exc)}"
        )
        logger.error(log_message, extra={"request_id": request_id})
        
        # Create error response
        error_response = format_error_response(
            exc, request_id=request_id, include_traceback=debug_mode
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        # Extract validation errors
        validation_errors = exc.errors()
        
        # Create custom validation error
        custom_error = ValidationError(
            message="Invalid request parameters",
            error_code="VALIDATION_ERROR",
            validation_errors=validation_errors,
        )
        
        # Log validation errors
        logger.warning(
            f"Validation error: {str(exc)}",
            extra={"validation_errors": validation_errors},
        )
        
        # Create error response
        error_response = format_error_response(
            custom_error, include_traceback=debug_mode
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response,
        )
    
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions"""
        # Log HTTP exceptions
        logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={"status_code": exc.status_code},
        )
        
        # Let FastAPI handle it
        return await http_exception_handler(request, exc)
    
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions as a last resort"""
        # Log unhandled exception with full traceback
        logger.exception(f"Unhandled exception: {str(exc)}")
        
        # Create custom error
        custom_error = SQLChatbotError(
            message="An unexpected error occurred",
            error_category=ErrorCategory.UNKNOWN,
            status_code=500,
            details=str(exc) if debug_mode else None,
        )
        
        # Create error response
        error_response = format_error_response(
            custom_error, include_traceback=debug_mode
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response,
        )