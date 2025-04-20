import streamlit as st

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(page_title="SQL Chatbot", page_icon="💬", layout="wide")

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import pandas as pd
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configuration
from src.config import API_URL

# Import auth module
from src.frontend.auth import (
    auth_required,
    login_register_page,
    logout_user,
    update_query_backend,
)

# Tooltips removed as requested


# Function to fetch data from the backend API with improved error handling
def query_backend(endpoint: str, data: Dict = {}, method: str = "GET") -> Dict:
    """Make a request to the backend API with robust error handling
    
    Args:
        endpoint: API endpoint path (without leading /)
        data: Dict of parameters for GET or JSON body for POST
        method: HTTP method ("GET" or "POST")
        
    Returns:
        Dict with API response or error information
    """
    # Constants for retry logic
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # Add debugging information
    st.sidebar.info(f"Connecting to: {API_URL}/{endpoint}")
    
    # Get authentication token from session state
    auth_token = st.session_state.get("token")

    # Prepare headers with auth token if available
    headers = {}
    cookies = {}
    if auth_token:
        # Include token in both cookies and Authorization header for maximum compatibility
        cookies = {"auth_token": auth_token}
        headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Log request information but limit data logging for sensitive/large payloads
    if endpoint == "query" and data and "question" in data:
        question = data["question"]
        truncated_question = (question[:50] + "...") if len(question) > 50 else question
        log_data = {**data, "question": truncated_question}
    else:
        log_data = data
    
    st.sidebar.info(f"Request: {method} {endpoint} - Data: {log_data}")
    
    # Implement retry logic for transient errors
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(
                    f"{API_URL}/{endpoint}", 
                    headers=headers, 
                    cookies=cookies, 
                    params=data,
                    timeout=15  # Increased timeout
                )
            else:  # POST
                response = requests.post(
                    f"{API_URL}/{endpoint}",
                    json=data,
                    headers=headers,
                    cookies=cookies,
                    timeout=15,
                )

            # Log the response status
            if response.status_code >= 200 and response.status_code < 300:
                st.sidebar.success(f"Response status: {response.status_code}")
            else:
                st.sidebar.warning(f"Response status: {response.status_code}")

            # Handle different HTTP status codes with more user-friendly messages
            if response.status_code == 401:
                st.sidebar.warning("Authentication issue detected:")
                st.sidebar.info(f"Token in session: {'Yes' if auth_token else 'No'}")
                st.sidebar.info(f"Headers sent: {headers}")
                st.sidebar.info(f"Cookies sent: {cookies}")
                
                # Clear authentication state
                st.warning("Your session has expired. Please log in again.")
                for key in ["is_authenticated", "token", "user"]:
                    if key in st.session_state:
                        del st.session_state[key]
                return {"success": False, "error": "Authentication required", "auth_error": True}
                
            elif response.status_code == 404:
                st.error(f"API endpoint '{endpoint}' not found. The service might be down.")
                return {"success": False, "error": f"Endpoint not found: {endpoint}"}
                
            elif response.status_code >= 500:
                # For server errors, retry a few times
                if attempt < MAX_RETRIES - 1:
                    st.warning(f"Server error. Retrying... (Attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    st.error("The server is experiencing issues. Please try again later.")
                    return {"success": False, "error": f"Server error: {response.status_code}"}
            
            # For other errors, try to parse the error message from response
            elif response.status_code >= 400:
                try:
                    error_json = response.json()
                    error_message = error_json.get("error", f"Error {response.status_code}")
                    st.error(f"Request error: {error_message}")
                    return {"success": False, "error": error_message}
                except Exception:
                    st.error(f"Request failed with status code {response.status_code}")
                    return {"success": False, "error": f"Error {response.status_code}: {response.text}"}
            
            # Success case - parse JSON response
            try:
                return response.json()
            except ValueError as e:
                st.error(f"Invalid JSON response received")
                st.sidebar.error(f"JSON parsing error: {str(e)}")
                st.sidebar.error(f"Response content: {response.text[:500]}...")
                return {"success": False, "error": "Invalid response format"}
                
        except requests.exceptions.Timeout:
            # Handle timeout errors
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Request timed out. Retrying... (Attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error("The request timed out. The server might be busy or experiencing issues.")
                return {"success": False, "error": "Request timed out after multiple attempts"}
                
        except requests.exceptions.ConnectionError:
            # Handle connection errors
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Connection error. Retrying... (Attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error("Unable to connect to the API. The backend server might be down.")
                return {"success": False, "error": "Connection error: Unable to reach API server"}
                
        except Exception as e:
            # Handle all other exceptions
            error_type = type(e).__name__
            st.error(f"Error connecting to backend: {error_type}")
            st.sidebar.error(f"Backend error details: {error_type}: {str(e)}")
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": f"{error_type}: {str(e)}"}


# Function to detect data types suitable for visualization
def detect_chart_type(df: pd.DataFrame) -> str:
    """Detect suitable chart type based on dataframe content"""
    # No data or too little data
    if df.empty or len(df) < 2:
        return "none"

    # Count number of columns by data type
    num_columns = df.select_dtypes(include=["number"]).columns
    cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    date_columns = df.select_dtypes(include=["datetime"]).columns

    # If we have exactly one category column and one numeric column, bar chart is good
    if len(cat_columns) == 1 and len(num_columns) == 1:
        # If category has few unique values, bar chart is suitable
        if df[cat_columns[0]].nunique() <= 15:
            return "bar"

    # If we have multiple numeric columns, line chart might be good
    if len(num_columns) >= 2:
        return "line"

    # If we have one category and multiple numerics, grouped bar might be good
    if len(cat_columns) == 1 and len(num_columns) >= 2:
        if df[cat_columns[0]].nunique() <= 10:
            return "grouped_bar"

    # If we have two numeric columns, scatter plot might be suitable
    if len(num_columns) == 2:
        return "scatter"

    # If we have one numeric column, histogram might be suitable
    if len(num_columns) == 1:
        return "histogram"

    # Default to no recommended chart
    return "none"


# Function to display query results
def display_results(results: Dict[str, Any]) -> None:
    """Display query results in Streamlit"""
    if not results.get("success", False):
        st.error(f"Error: {results.get('error', 'Unknown error')}")
        return

    # Display the explanation if available
    explanation = results.get("explanation")
    if explanation:
        st.info(explanation)

    # Display the SQL query
    with st.expander("Generated SQL Query", expanded=True):
        st.code(results.get("sql", ""), language="sql")

    # Display the data results
    data = results.get("data", [])
    if data:
        st.subheader("Query Results")

        # Convert to DataFrame for better display
        df = pd.DataFrame(data)

        # Show data table with styling
        with st.expander("Data Table", expanded=True):
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Found {len(data)} {'row' if len(data) == 1 else 'rows'}")

        # Visualization section
        st.subheader("Visualizations")
        
        # Try to detect suitable chart type
        chart_type = detect_chart_type(df)

        # Convert any date-like strings to datetime
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    # Try to parse as datetime
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        # Convert numeric columns if they're stored as strings
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass

        # Get numeric and categorical columns after conversion
        num_columns = df.select_dtypes(include=["number"]).columns.tolist()
        cat_columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        # Create tabs for different visualization options
        viz_tabs = st.tabs(["Chart Selection", "Custom Chart", "Data Statistics"])

        with viz_tabs[0]:
            # Recommended chart based on data
            if chart_type != "none":
                # Show recommended visualization
                st.subheader("Recommended Visualization")

                if chart_type == "bar":
                    cat_col = cat_columns[0]
                    num_col = num_columns[0]
                    st.bar_chart(df.set_index(cat_col)[num_col])
                    st.caption(f"Bar chart showing {num_col} by {cat_col}")

                elif chart_type == "line":
                    # Use the first column as index if it looks like a good candidate
                    index_col = df.columns[0]
                    line_cols = num_columns[:3]  # Limit to first 3 numeric columns
                    st.line_chart(df[line_cols])
                    st.caption(f"Line chart showing trends in {', '.join(line_cols)}")

                elif chart_type == "scatter":
                    st.subheader("Scatter Plot")
                    x_col = num_columns[0]
                    y_col = num_columns[1]
                    fig = {
                        "data": [{"type": "scatter", "x": df[x_col], "y": df[y_col]}],
                        "layout": {
                            "title": f"{y_col} vs {x_col}",
                            "xaxis": {"title": x_col},
                            "yaxis": {"title": y_col},
                        },
                    }
                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "grouped_bar":
                    st.write("Multiple numeric values grouped by category")
                    cat_col = cat_columns[0]
                    chart_data = df.set_index(cat_col)[
                        num_columns[:3]
                    ]  # Limit to first 3 numeric columns
                    st.bar_chart(chart_data)

                elif chart_type == "histogram":
                    st.subheader("Histogram")
                    num_col = num_columns[0]
                    fig = {
                        "data": [{"type": "histogram", "x": df[num_col]}],
                        "layout": {"title": f"Distribution of {num_col}"},
                    }
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No specific chart type automatically detected for this data")

        with viz_tabs[1]:
            st.subheader("Create Custom Chart")

            # Only show options if we have enough data
            if len(num_columns) > 0:
                chart_types = [
                    "Bar Chart",
                    "Line Chart",
                    "Scatter Plot",
                    "Pie Chart",
                    "Histogram",
                ]
                selected_chart = st.selectbox("Select Chart Type", chart_types)

                if selected_chart == "Bar Chart":
                    if cat_columns:
                        x_axis = st.selectbox("Select X-axis (Categories)", cat_columns)
                        y_axis = st.selectbox("Select Y-axis (Values)", num_columns)
                        st.bar_chart(df.set_index(x_axis)[y_axis])
                    else:
                        st.warning("Bar charts need categorical data for the x-axis")

                elif selected_chart == "Line Chart":
                    selected_columns = st.multiselect(
                        "Select columns to plot", num_columns, default=num_columns[:2]
                    )
                    if selected_columns:
                        st.line_chart(df[selected_columns])
                    else:
                        st.info("Please select at least one column")

                elif selected_chart == "Scatter Plot":
                    if len(num_columns) >= 2:
                        x_axis = st.selectbox("Select X-axis", num_columns)
                        y_axis = st.selectbox(
                            "Select Y-axis",
                            [col for col in num_columns if col != x_axis],
                            index=0,
                        )
                        fig = {
                            "data": [
                                {"type": "scatter", "x": df[x_axis], "y": df[y_axis]}
                            ],
                            "layout": {
                                "title": f"{y_axis} vs {x_axis}",
                                "xaxis": {"title": x_axis},
                                "yaxis": {"title": y_axis},
                            },
                        }
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Scatter plots require at least two numeric columns")

                elif selected_chart == "Pie Chart":
                    if cat_columns and num_columns:
                        cat_col = st.selectbox("Select Categories", cat_columns)
                        val_col = st.selectbox("Select Values", num_columns)

                        # Group by category and sum values
                        pie_data = df.groupby(cat_col)[val_col].sum().reset_index()

                        fig = {
                            "data": [
                                {
                                    "type": "pie",
                                    "labels": pie_data[cat_col],
                                    "values": pie_data[val_col],
                                    "hole": 0.4,
                                }
                            ],
                            "layout": {"title": f"{val_col} by {cat_col}"},
                        }
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Pie charts need both categorical and numeric data")

                elif selected_chart == "Histogram":
                    num_col = st.selectbox("Select Numeric Column", num_columns)
                    bins = st.slider(
                        "Number of bins", min_value=5, max_value=50, value=20
                    )
                    fig = {
                        "data": [
                            {"type": "histogram", "x": df[num_col], "nbinsx": bins}
                        ],
                        "layout": {"title": f"Distribution of {num_col}"},
                    }
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric data for visualization")

        with viz_tabs[2]:
            st.subheader("Data Statistics")

            # Show basic statistics for numeric columns
            if num_columns:
                st.write("Numeric Columns Statistics")
                st.dataframe(df[num_columns].describe(), use_container_width=True)

            # Show value counts for categorical columns (top 10)
            if cat_columns:
                st.write("Categorical Columns Value Counts")
                selected_cat = st.selectbox("Select column", cat_columns)
                st.dataframe(
                    df[selected_cat].value_counts().head(10).reset_index(),
                    use_container_width=True,
                )
    else:
        st.info("No data returned by the query.")


# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main app layout
st.title("💬 SQL Chatbot")
st.subheader("Ask questions about your database in natural language")

# Check if user is authenticated
if not st.session_state.get("is_authenticated", False):
    login_register_page()
    st.stop()  # Stop execution if not authenticated

# Sidebar with database info
with st.sidebar:
    st.header("Database Information")

    # Show logged in user and logout button
    if st.session_state.get("is_authenticated", False):
        st.write(f"Logged in as: {st.session_state.user.get('username', 'User')}")
        if st.button("Logout"):
            logout_user()
            st.rerun()

    # Fetch and display table list
    if st.button("Refresh Database Schema"):
        with st.spinner("Fetching database schema..."):
            tables_response = query_backend("tables")

            if tables_response.get("tables"):
                st.session_state.tables = tables_response.get("tables", [])
                st.success(
                    f"Found {len(st.session_state.tables)} tables in the database"
                )
            else:
                st.error("Could not fetch database tables")

    # Display tables if available
    if hasattr(st.session_state, "tables") and st.session_state.tables:
        st.subheader("Database Tables")
        for table in st.session_state.tables:
            with st.expander(table):
                # Fetch schema for this table when the expander is clicked
                schema_resp = query_backend(
                    "schema", {"table_name": table}, method="POST"
                )
                if schema_resp.get("db_schema", {}).get(table):
                    schema_data = schema_resp["db_schema"][table]
                    for col, type_info in schema_data.items():
                        st.text(f"• {col} ({type_info})")
                else:
                    st.text("Could not fetch schema")

# Display chat history
st.subheader("Chat History")
for i, chat in enumerate(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"**You**: {chat['content']}")
    else:
        if "sql" in chat:
            with st.chat_message("assistant"):
                st.markdown("**SQL Chatbot**:")

                # Display explanation if available
                if "explanation" in chat:
                    st.info(chat["explanation"])

                with st.expander("Generated SQL", expanded=False):
                    st.code(chat["sql"], language="sql")

                if "data" in chat and chat["data"]:
                    df = pd.DataFrame(chat["data"])

                    # Show data table with expander
                    with st.expander("Data Results", expanded=True):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.caption(
                            f"Found {len(chat['data'])} {'row' if len(chat['data']) == 1 else 'rows'}"
                        )

                    # Add a "Show Visualizations" button for each chat response
                    viz_key = f"viz_btn_{i}"
                    show_viz = False

                    if st.button(f"Show Visualizations", key=viz_key):
                        show_viz = True

                    if show_viz and not df.empty:
                        # Try to detect suitable chart type
                        chart_type = detect_chart_type(df)

                        # Convert any date-like strings to datetime
                        for col in df.columns:
                            if df[col].dtype == "object":
                                try:
                                    df[col] = pd.to_datetime(df[col])
                                except:
                                    pass

                        # Convert numeric columns if stored as strings
                        for col in df.columns:
                            if df[col].dtype == "object":
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except:
                                    pass

                        # Get numeric and categorical columns after conversion
                        num_columns = df.select_dtypes(
                            include=["number"]
                        ).columns.tolist()
                        cat_columns = df.select_dtypes(
                            include=["object", "string", "category"]
                        ).columns.tolist()

                        # Create visualization based on detected type
                        if chart_type != "none" and num_columns:
                            if chart_type == "bar" and cat_columns and num_columns:
                                cat_col = cat_columns[0]
                                num_col = num_columns[0]
                                st.bar_chart(df.set_index(cat_col)[num_col])

                            elif chart_type == "line" and num_columns:
                                line_cols = num_columns[
                                    :3
                                ]  # Limit to first 3 numeric columns
                                st.line_chart(df[line_cols])

                            elif chart_type == "scatter" and len(num_columns) >= 2:
                                x_col = num_columns[0]
                                y_col = num_columns[1]
                                fig = {
                                    "data": [
                                        {
                                            "type": "scatter",
                                            "x": df[x_col],
                                            "y": df[y_col],
                                        }
                                    ],
                                    "layout": {
                                        "title": f"{y_col} vs {x_col}",
                                        "xaxis": {"title": x_col},
                                        "yaxis": {"title": y_col},
                                    },
                                }
                                st.plotly_chart(fig, use_container_width=True)

                            elif (
                                chart_type == "grouped_bar"
                                and cat_columns
                                and len(num_columns) >= 2
                            ):
                                cat_col = cat_columns[0]
                                chart_data = df.set_index(cat_col)[num_columns[:3]]
                                st.bar_chart(chart_data)

                            elif chart_type == "histogram" and num_columns:
                                num_col = num_columns[0]
                                fig = {
                                    "data": [{"type": "histogram", "x": df[num_col]}],
                                    "layout": {"title": f"Distribution of {num_col}"},
                                }
                                st.plotly_chart(fig, use_container_width=True)

                            # Show statistics
                            with st.expander("Data Statistics"):
                                if num_columns:
                                    st.write("Numeric Statistics")
                                    st.dataframe(df[num_columns].describe())
                        else:
                            st.info("No suitable visualization detected for this data")

                elif "error" in chat:
                    st.error(chat["error"])
                else:
                    st.info("No data returned by the query.")
        else:
            st.markdown(f"**SQL Chatbot**: {chat['content']}")

# Example queries section
st.subheader("Example Queries")
st.write("Click on any example to try it:")

example_queries = [
    "Show me the average salary by department",
    "Which projects have the highest budget?",
    "List all employees in the Marketing department",
    "What is the total budget for projects in each department?",
    "Who are the top 5 highest paid employees?",
    "How many employees are assigned to each project?",
    "Show me projects that end after December 2025",
    "What is the average hours allocated per employee on each project?",
]

# Create buttons for each example query in a grid
cols = st.columns(2)
for i, query in enumerate(example_queries):
    col_idx = i % 2
    if cols[col_idx].button(f"📝 {query}", key=f"example_{i}"):
        st.session_state.current_query = query
        # Automatically rerun to update the text area
        st.rerun()
        
# Store current query in session state if not exists
if "current_query" not in st.session_state:
    st.session_state.current_query = "Show me the average salary by department"

# Input area
st.subheader("Ask a Question")
user_input = st.text_area(
    "Enter your question in natural language:",
    st.session_state.current_query,
    height=100,
)

if st.button("Submit Question"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Show loading spinner
        with st.spinner("Generating SQL and fetching results..."):
            try:
                # Query the backend
                result = query_backend("query", {"question": user_input}, method="POST")

                # Log raw response for debugging
                st.sidebar.write("Raw API Response:", result)

                # Add response to chat history
                if result.get("success", False):
                    chat_response = {
                        "role": "assistant",
                        "sql": result.get("sql", ""),
                        "data": result.get("data", []),
                    }

                    # Add explanation if available
                    if "explanation" in result:
                        chat_response["explanation"] = result["explanation"]
                else:
                    chat_response = {
                        "role": "assistant",
                        "content": "I couldn't process your query.",
                        "error": result.get("error", "Unknown error"),
                    }

                st.session_state.chat_history.append(chat_response)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.sidebar.error(f"Exception details: {traceback.format_exc()}")

                # Add error to chat history
                chat_response = {
                    "role": "assistant",
                    "content": "I encountered an error while processing your query.",
                    "error": str(e),
                }
                st.session_state.chat_history.append(chat_response)

        # Rerun to update the UI with new chat history
        st.rerun()
    else:
        st.warning("Please enter a question.")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Footer
st.markdown("---")
st.caption("SQL Chatbot powered by LangChain, FastAPI, and Streamlit")

# Apply authentication patch to query_backend
query_backend = update_query_backend(query_backend)
