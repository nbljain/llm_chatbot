import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

# Import configuration
from src.config import API_URL, ENDPOINT_AUTH


def login_form() -> None:
    """Render the login form"""
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password")
                return

            # Attempt to login
            response = login_user(username, password)

            if response["success"]:
                st.success("Login successful!")
                # Save user data to session state
                st.session_state.is_authenticated = True
                st.session_state.user = response["user"]
                st.session_state.token = response["token"]
                # Rerun the app to update the UI
                time.sleep(1)  # Small delay to show success message
                st.rerun()
            else:
                st.error(f"Login failed: {response.get('error', 'Unknown error')}")


def register_form() -> None:
    """Render the registration form"""
    st.subheader("Register")

    with st.form("register_form"):
        username = st.text_input("Username (at least 3 characters)")
        email = st.text_input("Email")
        password = st.text_input("Password (at least 6 characters)", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            # Validate inputs
            if not username or not email or not password:
                st.error("Please fill out all fields")
                return

            if len(username) < 3:
                st.error("Username must be at least 3 characters")
                return

            if len(password) < 6:
                st.error("Password must be at least 6 characters")
                return

            if password != confirm_password:
                st.error("Passwords do not match")
                return

            # Attempt to register
            response = register_user(username, email, password)

            if response["success"]:
                st.success("Registration successful! You are now logged in.")
                # Save user data to session state
                st.session_state.is_authenticated = True
                st.session_state.user = response["user"]
                st.session_state.token = response["token"]
                # Rerun the app to update the UI
                time.sleep(1)  # Small delay to show success message
                st.rerun()
            else:
                st.error(
                    f"Registration failed: {response.get('error', 'Unknown error')}"
                )


def login_user(username: str, password: str) -> Dict[str, Any]:
    """Send login request to the backend"""
    try:
        response = requests.post(
            f"{API_URL}{ENDPOINT_AUTH}/login",
            json={"username": username, "password": password},
        )

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            # Handle HTTP errors
            return {"success": False, "error": f"HTTP Error: {response.status_code}"}
    except Exception as e:
        # Handle network or other errors
        return {"success": False, "error": str(e)}


def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Send registration request to the backend"""
    try:
        response = requests.post(
            f"{API_URL}{ENDPOINT_AUTH}/register",
            json={"username": username, "email": email, "password": password},
        )

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            # Handle HTTP errors
            return {"success": False, "error": f"HTTP Error: {response.status_code}"}
    except Exception as e:
        # Handle network or other errors
        return {"success": False, "error": str(e)}


def logout_user() -> Dict[str, Any]:
    """Send logout request to the backend"""
    try:
        # Use the current session token
        response = requests.post(
            f"{API_URL}{ENDPOINT_AUTH}/logout",
            cookies={"auth_token": st.session_state.get("token", "")},
        )

        # Clear session state
        if "is_authenticated" in st.session_state:
            del st.session_state.is_authenticated
        if "user" in st.session_state:
            del st.session_state.user
        if "token" in st.session_state:
            del st.session_state.token

        return {"success": True}
    except Exception as e:
        # Handle network or other errors
        return {"success": False, "error": str(e)}


def auth_required(func):
    """Decorator to require authentication for a function"""

    def wrapper(*args, **kwargs):
        if not st.session_state.get("is_authenticated", False):
            st.error("You must be logged in to access this feature")
            login_register_page()
            return None
        return func(*args, **kwargs)

    return wrapper


def login_register_page():
    """Display the login/register page"""
    st.title("SQL Chatbot - Authentication")

    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_form()

    with tab2:
        register_form()

    st.info("Please login or register to use the SQL Chatbot")


def update_query_backend(original_query_backend):
    """Monkey patch the query_backend function to include authentication"""

    def authenticated_query_backend(
        endpoint: str, data: Dict = {}, method: str = "GET"
    ) -> Dict:
        """Make an authenticated request to the backend API"""
        try:
            # Add authentication token to cookies if available
            cookies = {}
            if st.session_state.get("token"):
                cookies["auth_token"] = st.session_state.token

            # Add debugging information
            st.sidebar.info(f"Connecting to: {API_URL}/{endpoint}")

            if method == "GET":
                response = requests.get(
                    f"{API_URL}/{endpoint}", cookies=cookies, timeout=10
                )
            else:  # POST
                response = requests.post(
                    f"{API_URL}/{endpoint}", json=data, cookies=cookies, timeout=10
                )

            # Log the response status
            st.sidebar.success(f"Response status: {response.status_code}")

            # Handle authentication errors
            if response.status_code == 401:
                # Clear authentication state
                if "is_authenticated" in st.session_state:
                    del st.session_state.is_authenticated
                return {
                    "success": False,
                    "error": "Authentication required. Please log in.",
                }

            response.raise_for_status()  # Raise exception for other HTTP errors
            return response.json()
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            st.sidebar.error(f"Backend error details: {type(e).__name__}: {str(e)}")
            return {"success": False, "error": str(e)}

    # Return the authenticated function
    return authenticated_query_backend
