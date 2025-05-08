import streamlit as st
import pandas as pd
import requests
import os
import sys
import time
import json
import traceback
from typing import Any, Dict, List, Optional
from functools import lru_cache

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(page_title="SQL Chatbot", page_icon="💬", layout="wide")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configuration
from src.config import API_URL

#------------------------------------------------------------------------------
# HELPER FUNCTIONS
#------------------------------------------------------------------------------

@lru_cache(maxsize=32)
def cached_query_backend(endpoint: str, data_str: str, method: str) -> Dict:
    """Cacheable version of query_backend function"""
    import json
    data = json.loads(data_str) if data_str else {}
    try:
        # Handle different HTTP methods
        if method.upper() == "GET":
            response = requests.get(
                f"{API_URL}/{endpoint}", 
                params=data,
                timeout=30
            )
        else:  # POST
            response = requests.post(
                f"{API_URL}/{endpoint}",
                json=data,
                timeout=30
            )
            
        # Handle different HTTP status codes
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False, 
                "error": f"HTTP Error: {response.status_code}",
                "status_code": response.status_code
            }
                
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def query_backend(endpoint: str, data: Dict = {}, method: str = "GET") -> Dict:
    """Make a request to the backend API with robust error handling"""
    try:
        # Log request details at debug level for transparency
        log_data = {k: v for k, v in data.items() if k != "api_key"}
        
        st.sidebar.info(f"Request: {method} {endpoint} - Data: {log_data}")
        
        # Special case: Don't cache user queries - they should always be processed
        if endpoint == "query" and method.upper() == "POST":
            # Handle direct (no caching)
            if method.upper() == "GET":
                response = requests.get(
                    f"{API_URL}/{endpoint}", 
                    params=data,
                    timeout=30
                )
            else:  # POST
                response = requests.post(
                    f"{API_URL}/{endpoint}",
                    json=data,
                    timeout=30
                )
                
            if response.status_code == 200:
                result = response.json()
            else:
                result = {
                    "success": False, 
                    "error": f"HTTP Error: {response.status_code}",
                    "status_code": response.status_code
                }
        else:
            # For other endpoints, use cached version
            import json
            data_str = json.dumps(data, sort_keys=True) if data else ""
            result = cached_query_backend(endpoint, data_str, method)
            
        # Log response code
        if result.get("status_code", 200) == 200:
            st.sidebar.success(f"Response status: 200")
        else:
            st.sidebar.warning(f"Response status: {result.get('status_code', 'Unknown')}")
            
        return result
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server might be overloaded or unavailable.")
        return {"success": False, "error": "Request timed out"}
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please check your network connection.")
        return {"success": False, "error": "Connection error"}
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
def stream_query(question: str, placeholder) -> Dict:
    """Stream a query response from the backend API using a simpler approach"""
    try:
        # Log request details
        st.sidebar.info(f"Processing question: {question[:30]}...")
        timestamp = time.strftime("%H:%M")
        
        # Show the first status message with WhatsApp/Teams style
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-bot">
                <strong>SQL Chatbot</strong><br>
                Starting to process your query...
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3) # Small pause for effect
        
        # Show SQL generation message
        timestamp = time.strftime("%H:%M")
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-bot">
                <strong>SQL Chatbot</strong><br>
                Translating your question to SQL...
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Now make the actual query to get the full response
        response = query_backend("query", {"question": question}, method="POST")
        
        # Log raw response
        st.sidebar.write("API Response:", response)
        
        if not response.get("success", False):
            # Show error message
            error_msg = response.get("error", "Unknown error occurred")
            timestamp = time.strftime("%H:%M")
            placeholder.markdown(f"""
            <div class="message-container">
                <div class="avatar avatar-bot">SQL</div>
                <div class="message-error">
                    <strong>Error</strong><br>
                    {error_msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return response
        
        # Show SQL execution message
        timestamp = time.strftime("%H:%M")
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-bot">
                <strong>SQL Chatbot</strong><br>
                SQL query generated! Now executing...
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3) # Small pause for effect
        
        # Show results processing message
        timestamp = time.strftime("%H:%M")
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-bot">
                <strong>SQL Chatbot</strong><br>
                Query executed successfully! Generating explanation...
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3) # Small pause for effect
        
        # Final message
        timestamp = time.strftime("%H:%M")
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-bot">
                <strong>SQL Chatbot</strong><br>
                Processing complete!
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Return the successful response
        return response
        
    except Exception as e:
        # Handle any unexpected exceptions
        timestamp = time.strftime("%H:%M")
        placeholder.markdown(f"""
        <div class="message-container">
            <div class="avatar avatar-bot">SQL</div>
            <div class="message-error">
                <strong>Error</strong><br>
                An unexpected error occurred: {str(e)}
                <div class="timestamp">{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.error(f"Error details: {str(e)}")
        return {"success": False, "error": f"Error during query processing: {str(e)}"}

@lru_cache(maxsize=16)
def cached_detect_chart_type(df_hash: str) -> dict:
    """Cached version of chart type detection using DataFrame hash"""
    # Regenerate DataFrame from hash (this is just a simple way to make pandas DataFrame hashable)
    import hashlib
    import pickle
    
    # Get all cached DataFrames from session state
    if "cached_dataframes" not in st.session_state:
        # This should never happen since we always cache before calling this function
        return {
            "primary": "none",
            "secondary": "none",
            "explanation": "No data available for visualization"
        }
    
    df = st.session_state.cached_dataframes.get(df_hash)
    if df is None:
        return {
            "primary": "none",
            "secondary": "none",
            "explanation": "Data no longer available"
        }
    
    # Now do the actual chart detection
    # Get column data types
    num_columns = df.select_dtypes(include=["number"]).columns.tolist()
    cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    
    # Get basic shape info
    row_count = len(df)
    col_count = len(df.columns)
    
    # Default response
    result = {
        "primary": "none",
        "secondary": "none",
        "explanation": "No suitable visualization detected for this data"
    }
    
    # CASE 1: Detect data suitable for pie charts (1 categorical column, 1 numeric column)
    if len(cat_columns) == 1 and len(num_columns) == 1:
        cat_col = cat_columns[0]
        unique_cats = df[cat_col].nunique()
        
        # If small number of categories (2-7), pie chart works well
        if 2 <= unique_cats <= 7:
            result = {
                "primary": "pie",
                "secondary": "bar",
                "explanation": f"Distribution of {num_columns[0]} across {unique_cats} categories of {cat_col}"
            }
        # If more categories, bar chart is better than pie
        elif unique_cats <= 15:
            result = {
                "primary": "bar",
                "secondary": "none",
                "explanation": f"Bar chart showing {num_columns[0]} by {cat_col}"
            }
    
    # CASE 2: Time series data (datetime + numeric columns)
    elif len(date_columns) > 0 and len(num_columns) > 0:
        result = {
            "primary": "line",
            "secondary": "bar" if len(num_columns) == 1 else "none",
            "explanation": f"Time series data with {len(num_columns)} numeric metrics"
        }
    
    # CASE 3: Multiple numeric columns but no datetime - scatter plots or grouped bars
    elif len(num_columns) >= 2 and len(num_columns) <= 4 and len(cat_columns) == 0:
        result = {
            "primary": "scatter",
            "secondary": "line",
            "explanation": f"Relationship between {num_columns[0]} and {num_columns[1]}"
        }
    
    # CASE 4: One category and multiple numeric columns - grouped bar charts
    elif len(cat_columns) == 1 and len(num_columns) >= 2:
        result = {
            "primary": "grouped_bar",
            "secondary": "bar",
            "explanation": f"Multiple metrics compared across {cat_columns[0]} categories"
        }
    
    # CASE 5: Single numeric column - histogram
    elif len(num_columns) == 1 and len(cat_columns) == 0:
        result = {
            "primary": "histogram",
            "secondary": "none",
            "explanation": f"Distribution of values in {num_columns[0]}"
        }
    
    # CASE 6: Just categorical columns - counts bar chart
    elif len(cat_columns) >= 1 and len(num_columns) == 0:
        result = {
            "primary": "bar",
            "secondary": "none",
            "explanation": f"Count of records by {cat_columns[0]}"
        }
    
    # CASE 7: Multiple numeric, one categorical - could be grouped bar or scatter with color
    elif len(num_columns) >= 2 and len(cat_columns) == 1:
        if df[cat_columns[0]].nunique() <= 10:  # Not too many categories
            result = {
                "primary": "scatter",
                "secondary": "grouped_bar",
                "explanation": f"Relationship between {num_columns[0]} and {num_columns[1]} grouped by {cat_columns[0]}"
            }
    
    return result

def detect_chart_type(df: pd.DataFrame) -> dict:
    """Detect suitable chart type based on dataframe content"""
    # Create a hash for the DataFrame
    import hashlib
    import pickle
    df_bytes = pickle.dumps(df)
    df_hash = hashlib.md5(df_bytes).hexdigest()
    
    # Store the DataFrame in session state for later retrieval
    if "cached_dataframes" not in st.session_state:
        st.session_state.cached_dataframes = {}
    st.session_state.cached_dataframes[df_hash] = df
    
    # Use the cached function
    return cached_detect_chart_type(df_hash)

def return_to_chat():
    """Helper function to return to chat mode"""
    st.session_state.app_mode = "chat"
    st.rerun()

def display_visualization(df: pd.DataFrame, chart_type: str, chart_options: Dict = {}):
    """Display visualization based on chart type and options"""
    # Get column types for chart configuration
    num_columns = df.select_dtypes(include=["number"]).columns.tolist()
    cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    
    # UBS color palette
    UBS_COLORS = {
        'red': '#ec0016',         # UBS Red
        'cappuccino': '#8c5b3a',  # UBS Cappuccino
        'gray': '#71716f',        # UBS Gray
        'black': '#191919',       # UBS Black
        'light_gray': '#e6e6e6',  # UBS Light Gray
        'white': '#ffffff'        # White
    }
    
    # Create a list of UBS colors for charts
    UBS_COLOR_SEQUENCE = [UBS_COLORS['red'], UBS_COLORS['cappuccino'], UBS_COLORS['gray'], 
                          UBS_COLORS['black'], UBS_COLORS['light_gray']]
    
    try:
        import plotly.express as px
        
        if chart_type == "bar" and cat_columns and num_columns:
            # Get columns to use (use provided options or defaults)
            x_col = chart_options.get("x_col", cat_columns[0])
            y_col = chart_options.get("y_col", num_columns[0])
            
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col,
                title=chart_options.get("title", f"{y_col} by {x_col}"),
                color_discrete_sequence=[UBS_COLORS['red']]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "line":
            line_cols = chart_options.get("y_cols", num_columns[:3])
            
            # If we have a date column, use it as x-axis
            if date_columns:
                x_col = chart_options.get("x_col", date_columns[0])
                fig = px.line(
                    df, 
                    x=x_col, 
                    y=line_cols,
                    title=chart_options.get("title", "Trends over time"),
                    markers=True,
                    color_discrete_sequence=UBS_COLOR_SEQUENCE
                )
            else:
                # Otherwise use index
                fig = px.line(
                    df, 
                    y=line_cols,
                    title=chart_options.get("title", f"Trends in {', '.join(line_cols)}"),
                    markers=True,
                    color_discrete_sequence=UBS_COLOR_SEQUENCE
                )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "scatter" and len(num_columns) >= 2:
            x_col = chart_options.get("x_col", num_columns[0])
            y_col = chart_options.get("y_col", num_columns[1])
            color_col = chart_options.get("color_col", num_columns[2] if len(num_columns) > 2 else None)
            
            if color_col:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=chart_options.get("title", f"{y_col} vs {x_col}"),
                    labels={x_col: x_col, y_col: y_col},
                    trendline="ols" if len(df) < 1000 else None
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=chart_options.get("title", f"{y_col} vs {x_col}"),
                    labels={x_col: x_col, y_col: y_col},
                    trendline="ols" if len(df) < 1000 else None
                )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "pie" and cat_columns and num_columns:
            cat_col = chart_options.get("cat_col", cat_columns[0])
            val_col = chart_options.get("val_col", num_columns[0])
            
            # Group by category and sum values if needed
            if len(df) > len(df[cat_col].unique()):
                pie_data = df.groupby(cat_col)[val_col].sum().reset_index()
            else:
                pie_data = df
            
            fig = px.pie(
                pie_data, 
                values=val_col, 
                names=cat_col,
                title=chart_options.get("title", f"{val_col} by {cat_col}"),
                hole=0.4,  # Donut style
                color_discrete_sequence=UBS_COLOR_SEQUENCE
            )
            # Add percentage labels
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hoverinfo='label+percent+value'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "histogram" and num_columns:
            num_col = chart_options.get("num_col", num_columns[0])
            fig = px.histogram(
                df, 
                x=num_col, 
                title=chart_options.get("title", f"Distribution of {num_col}"),
                nbins=chart_options.get("nbins", 20), 
                marginal="box",
                color_discrete_sequence=[UBS_COLORS['gray']]
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "grouped_bar" and cat_columns and len(num_columns) >= 2:
            cat_col = chart_options.get("cat_col", cat_columns[0])
            chart_cols = chart_options.get("chart_cols", num_columns[:4])  # Limit to first 4 numeric columns
            
            # Melt the dataframe for proper grouping
            melted_df = df.melt(
                id_vars=[cat_col], value_vars=chart_cols, 
                var_name='Metric', value_name='Value'
            )
            fig = px.bar(
                melted_df, 
                x=cat_col, 
                y='Value', 
                color='Metric',
                title=chart_options.get("title", f"Multiple metrics by {cat_col}"),
                barmode='group',
                color_discrete_sequence=UBS_COLOR_SEQUENCE
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {str(e)}")

def display_data_statistics(df: pd.DataFrame):
    """Display data statistics"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # UBS color palette for this function
        UBS_COLORS = {
            'red': '#ec0016',         # UBS Red
            'cappuccino': '#8c5b3a',  # UBS Cappuccino
            'gray': '#71716f',        # UBS Gray
            'black': '#191919',       # UBS Black
            'light_gray': '#e6e6e6',  # UBS Light Gray
            'white': '#ffffff'        # White
        }
        
        st.write("### Data Overview")
        
        # Get column types
        num_columns = df.select_dtypes(include=["number"]).columns.tolist()
        cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        
        # Data quality: missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("#### Missing Values")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': round(missing_data.values / len(df) * 100, 2)
            }).query('`Missing Values` > 0')
            
            if not missing_df.empty:
                st.dataframe(missing_df)
                
                # Plot missing values
                fig = px.bar(
                    missing_df, 
                    x='Column', 
                    y='Percentage',
                    title='Missing Values by Column (%)',
                    color='Percentage',
                    color_continuous_scale=[UBS_COLORS['light_gray'], UBS_COLORS['red']],
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Numeric data analysis
        if num_columns:
            st.write("#### Numeric Columns Statistics")
            
            # Show basic statistics
            numeric_stats = df[num_columns].describe().T
            numeric_stats['skew'] = df[num_columns].skew()
            numeric_stats = numeric_stats.round(2)
            st.dataframe(numeric_stats)
            
            # Distribution plots for numeric columns
            if len(num_columns) > 0:
                st.write("#### Distributions")
                
                # Create a subplot for each numeric column, maximum 6
                display_cols = num_columns[:6]
                fig = make_subplots(
                    rows=len(display_cols), 
                    cols=1, 
                    subplot_titles=[f"Distribution of {col}" for col in display_cols],
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(display_cols):
                    # Add histogram trace
                    fig.add_trace(
                        go.Histogram(
                            x=df[col], 
                            name=col,
                            marker_color=UBS_COLORS['red'],
                            nbinsx=30
                        ),
                        row=i+1, 
                        col=1
                    )
                
                # Update layout
                fig.update_layout(
                    height=300 * len(display_cols),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix for numeric columns
            if len(num_columns) > 1:
                st.write("#### Correlation Matrix")
                corr = df[num_columns].corr().round(2)
                
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=[UBS_COLORS['red'], 'white', UBS_COLORS['gray']],
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show highest correlations
                corr_pairs = corr.unstack().sort_values(ascending=False)
                # Remove self-correlations
                corr_pairs = corr_pairs[corr_pairs < 1.0]
                
                if not corr_pairs.empty:
                    st.write("Top 5 Strongest Correlations:")
                    top_corr = pd.DataFrame({
                        'Variables': [f"{i[0]} vs {i[1]}" for i in corr_pairs.index[:5]],
                        'Correlation': corr_pairs.values[:5]
                    })
                    st.dataframe(top_corr)
        
        # Categorical data analysis
        if cat_columns:
            st.write("#### Categorical Columns")
            
            for col in cat_columns[:4]:  # Limit to first 4 categories
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Count']
                
                # Only show top 15 categories if there are many
                if len(value_counts) > 15:
                    top_counts = value_counts.iloc[:15]
                    fig = px.bar(
                        top_counts, 
                        x=col, 
                        y='Count',
                        title=f"Top 15 categories in {col}",
                        color='Count',
                        color_continuous_scale=[UBS_COLORS['light_gray'], UBS_COLORS['cappuccino']]
                    )
                else:
                    fig = px.bar(
                        value_counts, 
                        x=col, 
                        y='Count',
                        title=f"Categories in {col}",
                        color='Count',
                        color_continuous_scale=[UBS_COLORS['light_gray'], UBS_COLORS['cappuccino']]
                    )
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating statistics: {str(e)}")

def custom_chart_form(df: pd.DataFrame, key_prefix: str = "custom_chart"):
    """Create a form for custom chart creation"""
    # Get column types
    num_columns = df.select_dtypes(include=["number"]).columns.tolist()
    cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
    
    # UBS color palette for this function
    UBS_COLORS = {
        'red': '#ec0016',         # UBS Red
        'cappuccino': '#8c5b3a',  # UBS Cappuccino
        'gray': '#71716f',        # UBS Gray
        'black': '#191919',       # UBS Black
        'light_gray': '#e6e6e6',  # UBS Light Gray
        'white': '#ffffff'        # White
    }
    
    with st.form(key=f"{key_prefix}_form"):
        chart_types = [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Pie Chart",
            "Histogram",
            "Grouped Bar Chart"
        ]
        selected_chart = st.selectbox(
            "Select Chart Type", 
            chart_types, 
            key=f"{key_prefix}_type"
        )
        
        # Dynamic options based on chart type
        if selected_chart == "Bar Chart" and cat_columns and num_columns:
            x_axis = st.selectbox("Select X-axis (Categories)", cat_columns, key=f"{key_prefix}_bar_x")
            y_axis = st.selectbox("Select Y-axis (Values)", num_columns, key=f"{key_prefix}_bar_y")
            options = {"x_col": x_axis, "y_col": y_axis}
            chart_code = "bar"
            
        elif selected_chart == "Line Chart" and num_columns:
            if date_columns:
                x_axis = st.selectbox("Select X-axis (Time)", date_columns, key=f"{key_prefix}_line_x")
                options = {"x_col": x_axis}
            else:
                options = {}
                
            y_columns = st.multiselect(
                "Select Y-axis values (Metrics)", 
                num_columns, 
                default=num_columns[:min(3, len(num_columns))],
                key=f"{key_prefix}_line_y"
            )
            options["y_cols"] = y_columns
            chart_code = "line"
            
        elif selected_chart == "Scatter Plot" and len(num_columns) >= 2:
            x_axis = st.selectbox("Select X-axis", num_columns, key=f"{key_prefix}_scatter_x")
            y_axis = st.selectbox(
                "Select Y-axis", 
                [col for col in num_columns if col != x_axis], 
                key=f"{key_prefix}_scatter_y"
            )
            
            color_options = ["None"] + cat_columns + num_columns
            color_col = st.selectbox("Color points by", color_options, key=f"{key_prefix}_scatter_color")
            
            options = {
                "x_col": x_axis, 
                "y_col": y_axis,
                "color_col": None if color_col == "None" else color_col
            }
            chart_code = "scatter"
            
        elif selected_chart == "Pie Chart" and cat_columns and num_columns:
            cat_col = st.selectbox("Select Categories", cat_columns, key=f"{key_prefix}_pie_cat")
            val_col = st.selectbox("Select Values", num_columns, key=f"{key_prefix}_pie_val")
            options = {"cat_col": cat_col, "val_col": val_col}
            chart_code = "pie"
            
        elif selected_chart == "Histogram" and num_columns:
            num_col = st.selectbox("Select Column", num_columns, key=f"{key_prefix}_hist_col")
            bins = st.slider("Number of bins", 5, 50, 20, key=f"{key_prefix}_hist_bins")
            options = {"num_col": num_col, "nbins": bins}
            chart_code = "histogram"
            
        elif selected_chart == "Grouped Bar Chart" and cat_columns and len(num_columns) >= 2:
            cat_col = st.selectbox("Select Categories (X-axis)", cat_columns, key=f"{key_prefix}_grouped_cat")
            metrics = st.multiselect(
                "Select Metrics to Compare", 
                num_columns, 
                default=num_columns[:min(4, len(num_columns))],
                key=f"{key_prefix}_grouped_metrics"
            )
            options = {"cat_col": cat_col, "chart_cols": metrics}
            chart_code = "grouped_bar"
            
        else:
            st.warning("Cannot create the selected chart type with the available data columns")
            options = {}
            chart_code = "none"
            
        chart_title = st.text_input("Chart Title (optional)", key=f"{key_prefix}_title")
        if chart_title:
            options["title"] = chart_title
            
        submit_button = st.form_submit_button("Create Chart")
        
        if submit_button and chart_code != "none":
            display_visualization(df, chart_code, options)

#------------------------------------------------------------------------------
# MAIN APP LAYOUT
#------------------------------------------------------------------------------

# Initialize app mode and visualization state
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "chat"
    
if "viz_active_data" not in st.session_state:
    st.session_state.viz_active_data = None
    st.session_state.viz_active_query = ""
    st.session_state.viz_active_title = ""
    
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle different app modes
if st.session_state.app_mode == "chat":
    # Main page layout
    st.title("💬 SQL Chatbot")
    st.markdown("Ask questions about your database in plain English")
    
    # Sidebar with information
    with st.sidebar:
        st.subheader("About")
        st.markdown("""
        This app allows you to query SQL databases using natural language.
        It translates your questions into SQL and provides visualizations of the results.
        """)
        
        # Show database schema information
        st.subheader("Database Schema")
        try:
            # Get all table schemas in a single API call
            all_schemas_response = query_backend(
                "schema",
                {"table_name": None},  # None means get all tables
                method="POST"
            )
            
            if all_schemas_response.get("db_schema"):
                db_schema = all_schemas_response["db_schema"]
                # Get tables from schema keys
                table_names = list(db_schema.keys())
                
                for table_name in table_names:
                    with st.expander(f"Table: {table_name}"):
                        table_schema = db_schema.get(table_name, {})
                        
                        # Create a table to display the schema
                        schema_data = []
                        for col_name, col_type in table_schema.items():
                            schema_data.append([col_name, col_type])
                            
                        if schema_data:
                            schema_df = pd.DataFrame(
                                schema_data, 
                                columns=["Column Name", "Data Type"]
                            )
                            st.dataframe(schema_df, hide_index=True)
            else:
                st.warning("Could not load database schema")
        except Exception as e:
            st.error(f"Error loading schema: {str(e)}")
    
    # Display chat history in a WhatsApp/Teams style
    st.subheader("Conversation")
    
    # Apply custom CSS for chat interface
    st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 5px;
        background-color: #f5f5f5;
        margin-bottom: 20px;
    }
    .message-container {
        display: flex;
        margin-bottom: 10px;
    }
    .message-user {
        background-color: #e6f7ff;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    .message-bot {
        background-color: #ffffff;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    .message-error {
        background-color: #fee8e9;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    .timestamp {
        font-size: 0.7em;
        color: #888;
        margin-top: 5px;
        text-align: right;
    }
    .message-user::before {
        content: "";
        position: absolute;
        bottom: 0;
        left: -10px;
        width: 20px;
        height: 20px;
        background-color: #e6f7ff;
        border-bottom-right-radius: 15px;
        z-index: -1;
    }
    .message-bot::before, .message-error::before {
        content: "";
        position: absolute;
        bottom: 0;
        right: -10px;
        width: 20px;
        height: 20px;
        background-color: inherit;
        border-bottom-left-radius: 15px;
        z-index: -1;
    }
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #ccc;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        color: white;
        font-weight: bold;
    }
    .avatar-user {
        background-color: #1e88e5;
    }
    .avatar-bot {
        background-color: #ec0016;
    }
    </style>
    """, unsafe_allow_html=True)

    chat_container = st.container(height=500, border=False)
    
    with chat_container:
        # Create a div with the chat-container class
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
        current_date = time.strftime("%B %d, %Y")
        st.markdown(f'<div style="text-align: center; margin: 10px 0; color: #777; font-size: 0.8em;">{current_date}</div>', unsafe_allow_html=True)
        
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                # User message on the left with timestamp
                timestamp = time.strftime("%H:%M")
                st.markdown(f"""
                <div class="message-container">
                    <div class="avatar avatar-user">U</div>
                    <div class="message-user">
                        <strong>You</strong><br>
                        {chat['content']}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bot message on the right with timestamp
                timestamp = time.strftime("%H:%M")
                
                # Check if it's an SQL result or just a text message
                if "sql" in chat and "data" in chat:
                    st.markdown(f"""
                    <div class="message-container">
                        <div class="avatar avatar-bot">SQL</div>
                        <div class="message-bot">
                            <strong>SQL Chatbot</strong><br>
                            I've translated your question to SQL.
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display the SQL query in a code block
                    with st.expander("View SQL Query"):
                        st.code(chat["sql"], language="sql")
                    
                    # Display data results
                    if chat["data"]:
                        # Convert to DataFrame for display
                        df = pd.DataFrame(chat["data"])
                        
                        # Data preview with download option
                        st.write("**Query Results:**")
                        with st.container(border=True):
                            st.dataframe(df, use_container_width=True)
                            row_count = len(df)
                            col_count = len(df.columns)
                            st.caption(f"Results: {row_count} rows × {col_count} columns")
                    
                    # Add data to session state for visualization
                    visualization_btn = st.button(
                        "📊 Open in Visualization Mode", 
                        key=f"viz_btn_{i}",
                        type="primary",
                    )
                    
                    if visualization_btn:
                        st.session_state.viz_active_data = df
                        st.session_state.viz_active_query = chat.get("sql", "")
                        # Find the corresponding user question
                        # This is more robust than assuming a fixed pattern
                        chat_index = i
                        while chat_index >= 0:
                            if "role" in st.session_state.chat_history[chat_index] and st.session_state.chat_history[chat_index]["role"] == "user":
                                question = st.session_state.chat_history[chat_index]["content"]
                                break
                            chat_index -= 1
                        else:
                            # Fallback if no user question is found
                            question = "Data Analysis"
                            
                        # Limit the question length for the title
                        if len(question) > 50:
                            short_question = question[:47] + "..."
                        else:
                            short_question = question
                        st.session_state.viz_active_title = f"Analysis: {short_question}"
                        # Set the app mode to visualization
                        st.session_state.app_mode = "visualization"
                        # Force a rerun to switch to visualization mode
                        st.rerun()
                    
                    # Display explanation if available
                    if "explanation" in chat:
                        # Add a slight delay between messages to simulate typing
                        timestamp = time.strftime("%H:%M:%S")
                        st.markdown(f"""
                        <div class="message-container">
                            <div class="avatar avatar-bot">SQL</div>
                            <div class="message-bot">
                                <strong>Explanation</strong><br>
                                {chat['explanation']}
                                <div class="timestamp">{timestamp}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                elif "error" in chat:
                    # Error message from the bot
                    st.markdown(f"""
                    <div class="message-container">
                        <div class="avatar avatar-bot">SQL</div>
                        <div class="message-error">
                            <strong>Error</strong><br>
                            {chat['error']}
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Regular bot message
                    st.markdown(f"""
                    <div class="message-container">
                        <div class="avatar avatar-bot">SQL</div>
                        <div class="message-bot">
                            <strong>SQL Chatbot</strong><br>
                            {chat['content']}
                            <div class="timestamp">{timestamp}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Close the chat container div
        st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    # Add custom CSS for the chat input area
    st.markdown("""
    <style>
    .chat-input-container {
        display: flex;
        border-radius: 20px;
        background-color: white;
        padding: 10px;
        margin-top: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    .chat-input-area {
        flex-grow: 1;
        border: none;
        outline: none;
        padding: 10px;
        border-radius: 20px;
        resize: none;
        font-family: inherit;
    }
    .send-button {
        background-color: #ec0016;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-left: 10px;
    }
    .send-button:hover {
        background-color: #d10014;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat input with custom styling
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    cols = st.columns([0.85, 0.15])
    
    with cols[0]:
        user_input = st.text_input(
            label="Message SQL Chatbot...",
            value=st.session_state.current_query,
            label_visibility="collapsed",
            key="chat_input",
        )
    
    send_btn = cols[1].button("Send", type="primary", key="send_btn", use_container_width=True)
    
    if send_btn or user_input and user_input != st.session_state.current_query:
        if user_input:
            # Update the current query in session state
            st.session_state.current_query = user_input
            
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Create placeholder for streaming responses
            chat_container = st.empty()
            
            # Show initial processing message with modern styling
            with chat_container.container():
                placeholder = st.empty()
                placeholder.markdown(f"""
                <div class="message-container">
                    <div class="avatar avatar-bot">SQL</div>
                    <div class="message-bot">
                        <strong>SQL Chatbot</strong><br>
                        Starting to process your query...
                        <div class="timestamp">{time.strftime("%H:%M")}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            try:
                # Use the streaming function to get real-time updates from the backend
                result = stream_query(user_input, placeholder)
                
                # Log raw response for debugging
                st.sidebar.write("Stream response result:", result)
                
                if result.get("success", False):
                    # Stream completed successfully, create the chat response
                    chat_response = {
                        "role": "assistant",
                        "sql": result.get("sql", ""),
                        "data": result.get("data", []),
                    }
                    
                    # Add explanation if available
                    if "explanation" in result:
                        chat_response["explanation"] = result["explanation"]
                else:
                    # Error occurred during streaming
                    chat_response = {
                        "role": "assistant",
                        "content": "I couldn't process your query.",
                        "error": result.get("error", "Unknown error"),
                    }
                
                # Add response to chat history
                st.session_state.chat_history.append(chat_response)
                
            except Exception as e:
                # Update to use our modern message style for error handling
                timestamp = time.strftime("%H:%M")
                placeholder.markdown(f"""
                <div class="message-container">
                    <div class="avatar avatar-bot">SQL</div>
                    <div class="message-error">
                        <strong>Error</strong><br>
                        I encountered an error while processing your query: {str(e)}
                        <div class="timestamp">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
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

# Visualization mode - a completely separate page for data visualization
elif st.session_state.app_mode == "visualization":
    # Get the data from session state
    df = st.session_state.viz_active_data
    
    if df is None or df.empty:
        st.error("No data available for visualization")
        if st.button("Return to Chat", type="primary", use_container_width=True):
            return_to_chat()
    else:
        # Page layout with modern aesthetic
        st.title("📊 Data Visualization", anchor=False)
        
        # Add a prominent return button at the top
        if st.button("← Return to Chat", key="return_btn", type="primary", use_container_width=True):
            return_to_chat()
            
        # Add some space
        st.write("")
        
        # Add the title in a container for better visual separation
        with st.container(border=True):
            st.header(st.session_state.viz_active_title, divider="blue")
        
        # Show the SQL query that generated this data
        with st.expander("SQL Query", expanded=False):
            st.code(st.session_state.viz_active_query, language="sql")
        
        # Data preview in a styled container
        st.markdown("### Dataset Preview")
        with st.container(border=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(df)} rows")
        
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
        
        # Get column types for visualization
        num_columns = df.select_dtypes(include=["number"]).columns.tolist()
        cat_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        date_columns = df.select_dtypes(include=["datetime"]).columns.tolist()
        
        # Main visualization section with tabs
        st.write("")  # Add space
        st.markdown("## Visualizations")
        
        # Create tabs to separate different visualizations with better styling
        tab1, tab2, tab3 = st.tabs([
            "🌟 Recommended Charts", 
            "🔧 Custom Chart", 
            "📊 Data Statistics"
        ])
        
        # Tab 1: Recommended Charts
        with tab1:
            st.markdown("### 🌟 Auto-detected Charts")
            
            # Analyze data and recommend chart types
            chart_recommendations = detect_chart_type(df)
            primary_chart = chart_recommendations.get("primary", "none")
            secondary_chart = chart_recommendations.get("secondary", "none")
            chart_explanation = chart_recommendations.get("explanation", "No chart detected")
            
            # Display chart explanation
            st.info(f"Chart recommendation: {chart_explanation}")
            
            # Generate and display the recommended charts 
            if primary_chart != "none":
                if secondary_chart != "none":
                    # Two-column layout for multiple charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"#### Primary Chart: {primary_chart.title()}")
                        display_visualization(df, primary_chart)
                    
                    with col2:
                        st.write(f"#### Secondary Chart: {secondary_chart.title()}")
                        display_visualization(df, secondary_chart)
                else:
                    # Single chart - no columns needed
                    st.write(f"#### Primary Chart: {primary_chart.title()}")
                    display_visualization(df, primary_chart)
            else:
                st.info("No suitable visualization detected for this data")
        
        # Tab 2: Custom Chart
        with tab2:
            st.markdown("### 🔧 Create Your Own Chart")
            custom_chart_form(df)
                
        # Tab 3: Data Statistics
        with tab3:
            st.markdown("### 📊 Data Statistics and Insights")
            display_data_statistics(df)
            
        # Add export options
        st.markdown("## Export Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="query_result.csv",
            mime="text/csv",
        )
