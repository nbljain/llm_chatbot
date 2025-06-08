"""
Streamlit frontend for SQL Chatbot - Clean Chat Interface
"""

import os
import sys
import time
import json
from typing import Dict, List, Any
import logging

import streamlit as st
import requests
import pandas as pd
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure page
st.set_page_config(
    page_title="Chatbot for Employee Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get backend URL from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize logging
logger = logging.getLogger(__name__)


def query_backend(endpoint: str, data: Dict = {}, method: str = "GET") -> Dict:
    """Make a request to the backend API with robust error handling"""
    url = f"{BACKEND_URL}{endpoint}"

    try:
        if method.upper() == "GET":
            response = requests.get(url, params=data, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            return {"success": False, "error": f"Unsupported HTTP method: {method}"}

        response.raise_for_status()
        return response.json()

    except Timeout:
        logger.error(f"Timeout error when requesting {url}")
        return {
            "success": False,
            "error": "Request timed out. The backend might be busy processing your query.",
        }
    except ConnectionError:
        logger.error(f"Connection error when requesting {url}")
        return {
            "success": False,
            "error": "Cannot connect to the backend. Please ensure the backend server is running.",
        }
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {e.response.status_code} when requesting {url}")
        if e.response.status_code == 500:
            return {
                "success": False,
                "error": "Internal server error. There might be an issue with your query or the database.",
            }
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except Exception as e:
        logger.error(f"Unexpected error when requesting {url}: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_session_id" not in st.session_state:
    st.session_state.user_session_id = str(hash(time.time()))

# Create the main app layout
st.title("🤖 Employee Chatbot")

# Sidebar with information
with st.sidebar:
    st.header("📊 Database Info")

    # Get database schema
    schema_response = query_backend("/schema", {}, "POST")

    # Check if response contains error or db_schema
    if "error" in schema_response:
        st.error(
            f"Database connection failed: {schema_response.get('error', 'Unknown error')}"
        )
    elif "db_schema" in schema_response:
        db_schema = schema_response.get("db_schema", {})

        if db_schema:
            # Filter out system tables like sqlite_sequence
            user_tables = {
                k: v for k, v in db_schema.items() if not k.startswith("sqlite_")
            }

            st.success(f"Connected to database with {len(user_tables)} tables")

            # Show tables in an expandable section
            with st.expander("View Tables", expanded=False):
                for table_name, columns in user_tables.items():
                    with st.container():
                        st.markdown(f"**📋 {table_name.title()}**")

                        # Create a formatted table for columns
                        col_data = []
                        for col_name, col_type in columns.items():
                            col_data.append({"Column": col_name, "Type": col_type})

                        if col_data:
                            df_cols = pd.DataFrame(col_data)
                            st.dataframe(
                                df_cols, hide_index=True, use_container_width=True
                            )

                        st.markdown("---")
        else:
            st.warning("No tables found in database")
    else:
        st.error("Unexpected response format from database")


# Main chat interface
st.markdown("### Chat with your database")

# Display chat history with custom layout
for message in st.session_state.chat_history:
    if message["role"] == "user":
        # User message on the right
        col1, col2, col3 = st.columns([2, 1, 7])
        with col3:
            with st.container():
                st.markdown(
                    f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>You:</strong> {message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        # Assistant message on the left
        col1, col2, col3 = st.columns([7, 1, 2])
        with col1:
            with st.container():
                st.markdown(
                    """
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 Employee Chatbot:</strong>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Assistant message content
                if "explanation" in message:
                    st.write(message["explanation"])

                if "error" in message:
                    st.error(message["error"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display user message with right alignment
    col1, col2, col3 = st.columns([2, 1, 7])
    with col3:
        with st.container():
            st.markdown(
                f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>You:</strong> {prompt}
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Process the query and display response with left alignment
    col1, col2, col3 = st.columns([7, 1, 2])
    with col1:
        with st.container():
            st.markdown(
                """
            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                <strong>🤖 Chatbot:</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.spinner("Processing your question..."):
                # Send the question to the backend
                query_data = {
                    "question": prompt,
                    "user_id": st.session_state.user_session_id,
                }

                response = query_backend("/query", query_data, "POST")

                # Handle API response properly
                if response.get("success") == True:
                    explanation = response.get("explanation", "")
                    sql_query = response.get("sql", "")
                    data = response.get("data", [])

                    # Stream the explanation
                    if explanation:
                        response_container = st.empty()
                        streamed_text = ""
                        for i in range(
                            0, len(explanation), 5
                        ):  # Stream 5 characters at a time
                            streamed_text += explanation[i : i + 5]
                            response_container.write(streamed_text)
                            time.sleep(0.01)  # Small delay for streaming effect

                    # Add to chat history (without displaying data tables)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "explanation": explanation,
                            "data": data,
                            "sql": sql_query,
                        }
                    )
                else:
                    # Handle errors
                    error_message = response.get("error", "Unknown error occurred")
                    st.error(f"Error: {error_message}")

                    # Add error to chat history
                    st.session_state.chat_history.append(
                        {"role": "assistant", "error": error_message}
                    )

# Clear chat button
if st.button("Clear Chat History", type="secondary"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.markdown("---")
st.caption("Chatbot powered by LangChain, FastAPI, and Streamlit")
