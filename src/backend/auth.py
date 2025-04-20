import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr

from src.config import AUTH_TOKEN_EXPIRY_MINUTES, ENDPOINT_AUTH

# Import from project
from src.database.auth import (
    authenticate_user,
    register_user,
    update_user_password,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create an API router
router = APIRouter(prefix=ENDPOINT_AUTH, tags=["auth"])


# Define request and response models
class UserRegistrationRequest(BaseModel):
    username: str
    email: str  # Using str instead of EmailStr for simplicity
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str


class AuthResponse(BaseModel):
    success: bool
    user: Optional[UserResponse] = None
    error: Optional[str] = None
    token: Optional[str] = None


# Simple in-memory token store - in a production app, use a proper token mechanism
# Format: {token: {user_id: int, expires: datetime}}
active_tokens = {}


def generate_token(user_id: int) -> str:
    """Generate a secure token for authentication"""
    token = secrets.token_hex(32)
    # Set token expiration
    expires = datetime.now() + timedelta(minutes=AUTH_TOKEN_EXPIRY_MINUTES)
    # Store token
    active_tokens[token] = {"user_id": user_id, "expires": expires}
    return token


def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """Validate token and return current user"""
    if token not in active_tokens:
        return None

    token_data = active_tokens[token]
    # Check if token is expired
    if token_data["expires"] < datetime.now():
        del active_tokens[token]
        return None

    return {"user_id": token_data["user_id"]}


def get_user_from_request(request: Request) -> Optional[Dict[str, Any]]:
    """Extract and validate token from cookie or Authorization header"""
    # Try to get token from cookie
    token = request.cookies.get("auth_token")

    # If not in cookie, try to get from Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")

    # If still no token, authentication fails
    if not token:
        return None

    return get_current_user(token)


@router.post("/register", response_model=AuthResponse)
async def register(request: UserRegistrationRequest, response: Response):
    """Register a new user"""
    # Username validation
    if len(request.username) < 3:
        return {
            "success": False,
            "error": "Username must be at least 3 characters long",
        }

    # Password validation
    if len(request.password) < 6:
        return {
            "success": False,
            "error": "Password must be at least 6 characters long",
        }

    # Register the user
    result = register_user(request.username, request.email, request.password)

    if result["success"]:
        # Generate token
        token = generate_token(result["user"]["id"])
        # Set token in cookie
        response.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            max_age=AUTH_TOKEN_EXPIRY_MINUTES * 60,  # Convert to seconds
            path="/",
        )
        return {"success": True, "user": result["user"], "token": token}
    else:
        # Return the error from the registration function
        return {"success": False, "error": result["error"]}


@router.post("/login", response_model=AuthResponse)
async def login(request: UserLoginRequest, response: Response):
    """Authenticate a user and return a token"""
    result = authenticate_user(request.username, request.password)

    if result["success"]:
        # Generate token
        token = generate_token(result["user"]["id"])
        # Set token in cookie
        response.set_cookie(
            key="auth_token",
            value=token,
            httponly=True,
            max_age=AUTH_TOKEN_EXPIRY_MINUTES * 60,  # Convert to seconds
            path="/",
        )
        return {"success": True, "user": result["user"], "token": token}
    else:
        # Return the error from the authentication function
        return {"success": False, "error": result["error"]}


@router.post("/logout")
async def logout(response: Response, request: Request):
    """Log out a user by clearing their token"""
    token = request.cookies.get("auth_token")
    if token and token in active_tokens:
        del active_tokens[token]

    # Clear the cookie
    response.delete_cookie(key="auth_token")

    return {"success": True}


@router.post("/change-password", response_model=AuthResponse)
async def change_password(request: PasswordChangeRequest, req: Request):
    """Change a user's password"""
    # Get current user from token
    user_data = get_user_from_request(req)
    if not user_data:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Password validation
    if len(request.new_password) < 6:
        return {
            "success": False,
            "error": "New password must be at least 6 characters long",
        }

    # Update the password
    result = update_user_password(
        user_data["user_id"], request.current_password, request.new_password
    )

    return result


@router.get("/me", response_model=AuthResponse)
async def get_current_user_info(request: Request):
    """Get information about the currently authenticated user"""
    user_data = get_user_from_request(request)
    if not user_data:
        return {"success": False, "error": "Not authenticated"}

    # In a real application, you would fetch the user data from the database
    # using the user_id from the token
    # For now, we'll just return the user_id
    return {
        "success": True,
        "user": {"id": user_data["user_id"], "username": "", "email": ""},
    }
