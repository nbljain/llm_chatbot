import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from src.database.db import get_all_table_schemas

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

# Global conversation contexts for each user
user_conversation_context = {}


def get_llm():
    """Initialize and return the language model"""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return ChatOpenAI(temperature=0, model=MODEL_NAME, api_key=api_key)


def get_table_schema_string():
    """Get database schema as a formatted string for the LLM prompt"""
    from src.database.relationships import get_relationships_for_llm
    
    schema_dict = get_all_table_schemas()
    
    if not schema_dict:
        return "No tables found in the database."
    
    schema_str = "Database Schema:\n"
    
    for table_name, columns in schema_dict.items():
        schema_str += f"Table: {table_name}\n"
        schema_str += "Columns:\n"
        
        for col_name, col_type in columns.items():
            schema_str += f"  - {col_name} ({col_type})\n"
        
        schema_str += "\n"
    
    # Add table relationships information
    relationships_str = get_relationships_for_llm()
    if relationships_str and relationships_str != "No table relationships defined.":
        schema_str += relationships_str
    
    return schema_str


def get_conversation_context_string(user_id: Optional[str]) -> str:
    """Get conversation context as a formatted string"""
    if not user_id or user_id not in user_conversation_context:
        return ""
    
    context = user_conversation_context[user_id]
    if not context:
        return ""
    
    context_str = "\nPrevious conversation history:\n"
    for i, interaction in enumerate(context[-3:]):  # Only include the last 3 interactions
        context_str += f"Question {i+1}: {interaction['question']}\n"
        context_str += f"SQL: {interaction['sql']}\n"
        if interaction.get("results"):
            # Truncate long results
            results_str = json.dumps(interaction['results'], indent=2)
            if len(results_str) > 200:
                results_str = results_str[:200] + "..."
            context_str += f"Results: {results_str}\n"
    
    return context_str


def update_conversation_context(user_id, question, sql_query, query_results, explanation):
    """Update the conversation context for a user"""
    if user_id not in user_conversation_context:
        user_conversation_context[user_id] = []
    
    # Add the new interaction
    interaction = {
        "question": question,
        "sql": sql_query,
        "results": query_results,
        "explanation": explanation
    }
    
    user_conversation_context[user_id].append(interaction)
    
    # Keep only the last 10 interactions to prevent memory from growing too large
    if len(user_conversation_context[user_id]) > 10:
        user_conversation_context[user_id] = user_conversation_context[user_id][-10:]


def generate_sql_query(user_question: str, user_id: Optional[str] = None) -> str:
    """Generate SQL from natural language question
    
    Args:
        user_question: The natural language question to translate to SQL
        user_id: Optional user identifier for maintaining conversation history
    
    Returns:
        Generated SQL query string or error message
    """
    logger.info(f"Processing question: {user_question}")
    
    llm = get_llm()
    
    # Get the database schema
    schema_str = get_table_schema_string()
    
    # Get conversation context
    context_str = get_conversation_context_string(user_id)
    
    system_prompt = f"""You are an expert SQL assistant that translates natural language questions into SQL queries.

{schema_str}
{context_str}

Your task is to generate a valid SQLite SQL query based on the user's question.
- Only generate a valid SQL query, don't include any explanations or commentary
- Don't use any tables or columns not listed in the schema
- If the user input is vague or you can't create a valid query, explain why clearly instead of attempting to generate SQL
- Be explicit when you need more information to create a proper query
- For general questions like "tell me about X" where X is in the database, try to generate a query that shows relevant information about X
- Reject requests that are not database queries with a polite explanation
- Ensure your queries are efficient and properly formatted
- Use appropriate joins, aggregations, or filters as needed
- The generated SQL should be directly executable without modification
- Make sure to follow SQLite syntax (not PostgreSQL)
- Use conversation history to understand context and references to previous queries
- When the user refers to previous results or queries, use that context to generate the new query
- If the user asks to refine a previous query or uses phrases like "show me more", "filter this by", etc., use the previous SQL as a base"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    
    try:
        response = llm(messages)
        generated_sql = response.content.strip()
        
        logger.info(f"Generated SQL: {generated_sql}")
        return generated_sql
        
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        logger.error(error_msg)
        return error_msg


def generate_answer(question: str, sql_query: str, query_results: List[Dict], user_id: Optional[str] = None) -> str:
    """Generate a natural language explanation of the SQL query results
    
    Args:
        question: The natural language question
        sql_query: The generated SQL query
        query_results: The results of the SQL query
        user_id: Optional user identifier for maintaining conversation history
    
    Returns:
        Natural language explanation of the results
    """
    logger.info("Generating answer explanation...")
    
    llm = get_llm()
    
    system_prompt = """You are an expert SQL assistant that explains SQL queries and their results in simple, clear language.

Your task is to provide a helpful explanation of:
1. What the SQL query does
2. What the results show
3. Any insights from the data

Guidelines:
- Use clear, non-technical language
- Explain the business meaning of the results
- Highlight interesting patterns or insights
- Keep explanations concise but informative
- If there are no results, explain why that might be the case"""

    # Truncate results if they're too long for the prompt
    results_str = json.dumps(query_results, indent=2)
    if len(results_str) > 1000:
        results_str = results_str[:1000] + "... (truncated)"

    explanation_prompt = f"""
Question: {question}
SQL Query: {sql_query}
Results: {results_str}

Please provide a clear explanation of what this query does and what insights can be drawn from the results.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=explanation_prompt)
    ]
    
    try:
        response = llm(messages)
        explanation = response.content.strip()
        
        # Update conversation context
        if user_id:
            update_conversation_context(user_id, question, sql_query, query_results, explanation)
        
        logger.info("Generated explanation successfully")
        return explanation
        
    except Exception as e:
        error_msg = f"Error generating explanation: {str(e)}"
        logger.error(error_msg)
        return error_msg
