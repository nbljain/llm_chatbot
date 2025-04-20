import json
import logging
import os
import sys

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from src.database.db import get_all_table_schemas

# Dictionary to store conversation histories by user ID
user_memories = {}

# Dictionary to store previous questions and their SQL/results by user ID
user_conversation_context = {}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"


# Initialize the LLM
def get_llm():
    """Initialize and return the language model"""
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    return ChatOpenAI(temperature=0, model=MODEL_NAME, api_key=api_key)


def get_table_schema_string():
    """Get database schema as a formatted string for the LLM prompt"""
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

    return schema_str


def setup_sql_chain(user_id=None):
    """Set up the LangChain chain for SQL generation with optional memory

    Args:
        user_id: Optional user identifier for maintaining conversation history
    """
    llm = get_llm()

    # Get database schema
    schema_str = get_table_schema_string()

    # Get conversation context if available
    context_str = ""
    if user_id and user_id in user_conversation_context:
        context = user_conversation_context[user_id]
        if context:
            context_str = "\nPrevious conversation history:\n"
            for i, interaction in enumerate(
                context[-3:]
            ):  # Only include the last 3 interactions
                context_str += f"Question {i+1}: {interaction['question']}\n"
                context_str += f"SQL: {interaction['sql']}\n"
                if interaction.get("results"):
                    context_str += f"Results: {json.dumps(interaction['results'], indent=2)[:200]}...\n"  # Truncate long results

    # Define the system prompt with the database schema and conversation context
    system_template = """You are an expert SQL assistant that translates natural language questions into SQL queries.
    
{schema}

{context}

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
- If the user asks to refine a previous query or uses phrases like "show me more", "filter this by", etc., use the previous SQL as a base
"""

    # Create the chat prompt template
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_template
    )
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Set up memory if user_id is provided
    memory = None
    if user_id is not None:
        # Create or retrieve conversation memory for this user
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="question",  # Specify the input key for memory
            )
        memory = user_memories[user_id]

    # Create the chain with or without memory
    chain_kwargs = {"llm": llm, "prompt": chat_prompt, "verbose": True}

    if memory:
        chain_kwargs["memory"] = memory

    chain = LLMChain(**chain_kwargs)

    return chain


def update_conversation_context(
    user_id, question, sql_query, query_results, explanation
):
    """Update the conversation context for a user

    Args:
        user_id: User identifier
        question: The natural language question
        sql_query: The generated SQL query
        query_results: The results of the SQL query
        explanation: The generated explanation
    """
    if user_id is None:
        return

    # Create user context if not exists
    if user_id not in user_conversation_context:
        user_conversation_context[user_id] = []

    # Add the current interaction to the context
    user_conversation_context[user_id].append(
        {
            "question": question,
            "sql": sql_query,
            "results": (
                query_results[:5] if query_results else []
            ),  # Store only first 5 results to save memory
            "explanation": explanation,
        }
    )

    # Keep only the last 5 interactions to limit memory usage
    if len(user_conversation_context[user_id]) > 5:
        user_conversation_context[user_id] = user_conversation_context[
            user_id
        ][-5:]

    # Add to the conversation memory system
    if user_id in user_memories:
        # Add the interaction as AI and human messages in the memory
        memory = user_memories[user_id]
        if hasattr(memory, "chat_memory"):
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(
                f"SQL: {sql_query}\n\nExplanation: {explanation}"
            )


def generate_answer(question, sql_query, query_results, user_id=None):
    """Generate a natural language explanation of the SQL query results

    Args:
        question: The natural language question
        sql_query: The generated SQL query
        query_results: The results of the SQL query
        user_id: Optional user identifier for maintaining conversation history
    """
    try:
        llm = get_llm()

        # Check if there are any results to explain
        if not query_results:
            return {
                "success": True,
                "explanation": "I didn't find any data matching your query. This may mean there are no records that meet your criteria or the query needs to be modified.",
            }

        # Convert query results to a string representation
        results_str = json.dumps(query_results, indent=2)

        # Add conversation context if available
        context_str = ""
        if user_id and user_id in user_conversation_context:
            context = user_conversation_context[user_id]
            if context:
                context_str = "Previous conversation context:\n"
                for i, interaction in enumerate(
                    context[-3:]
                ):  # Only include the last 3 interactions
                    context_str += (
                        f"Question {i+1}: {interaction['question']}\n"
                    )
                    context_str += f"SQL: {interaction['sql']}\n"
                    context_str += (
                        f"Explanation: {interaction['explanation']}\n\n"
                    )

        # Define the prompt for generating the explanation
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question in a clear, concise way.\n\n"
        )

        if context_str:
            prompt += f"{context_str}\n\n"

        prompt += (
            f"Current Question: {question}\n\n"
            f"SQL Query: {sql_query}\n\n"
            f"SQL Result: {results_str}\n\n"
            "Provide a clear explanation that answers the user's question based on the data. "
            "Include key insights, patterns, or notable findings. "
            "Format numbers appropriately (e.g., with commas for thousands) and use precise language. "
            "If there are trends or comparisons worth noting, mention them. "
            "If the current question refers to previous questions, use the conversation context to provide a more relevant answer. "
            "If you cannot draw meaningful insights from the data or the query results are unclear, "
            "simply state that you cannot provide a complete answer based on the available data. "
            "Never make up facts that aren't supported by the data. "
            "Keep your answer concise but informative."
        )

        # Generate the explanation - create proper chat message
        from langchain.prompts.chat import HumanMessage

        # Create a proper message object
        messages = [HumanMessage(content=prompt)]
        response = llm(messages)
        explanation = response.content.strip()

        # Check if the explanation indicates uncertainty or inability to answer
        uncertainty_phrases = [
            "i don't have enough information",
            "cannot provide a complete answer",
            "insufficient data",
            "not clear from the data",
        ]

        if any(
            phrase in explanation.lower() for phrase in uncertainty_phrases
        ):
            explanation = (
                f"{explanation}\n\nPlease try asking a more specific question or "
                "providing more details about what you're looking for."
            )

        return {"success": True, "explanation": explanation}
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": "I'm unable to generate an explanation for these results right now. Please focus on the data table for insights.",
        }


def generate_sql_query(user_question, user_id=None):
    """Generate SQL from natural language question

    Args:
        user_question: The natural language question to translate to SQL
        user_id: Optional user identifier for maintaining conversation history
    """
    try:
        # Build the context string for SQL generation
        context_str = ""
        if user_id and user_id in user_conversation_context:
            context = user_conversation_context[user_id]
            if context:
                context_str = "\nPrevious conversation history:\n"
                for i, interaction in enumerate(
                    context[-3:]
                ):  # Only include the last 3 interactions
                    context_str += (
                        f"Question {i+1}: {interaction['question']}\n"
                    )
                    context_str += f"SQL: {interaction['sql']}\n"
                    if interaction.get("results"):
                        context_str += f"Results: {json.dumps(interaction['results'], indent=2)[:200]}...\n"  # Truncate long results

        # Set up the chain with user_id for memory
        chain = setup_sql_chain(user_id)
        schema = get_table_schema_string()

        if "No tables found" in schema:
            return {
                "success": False,
                "error": "No database tables found. Please ensure your database is properly configured and contains data.",
            }

        # Generate SQL query with context

        if user_id and user_id in user_memories:
            # For chains with memory, just pass the question directly
            # The schema and context are included in the system message template
            result = chain.predict(
                question=user_question, schema=schema, context=context_str
            )
        else:
            # No memory available, use the run method
            result = chain.run(
                question=user_question, schema=schema, context=context_str
            )

        generated_sql = result.strip()

        # Clean up the SQL - remove markdown formatting if present
        if generated_sql.startswith("```"):
            # Remove markdown code block syntax
            lines = generated_sql.split("\n")
            # Remove the first line if it contains ```
            if "```" in lines[0]:
                lines = lines[1:]
            # Remove the last line if it contains ```
            if lines and "```" in lines[-1]:
                lines = lines[:-1]
            generated_sql = "\n".join(lines).strip()

        # Further cleanup: remove "sql" if it appears alone on the first line
        lines = generated_sql.split("\n")
        if lines and lines[0].strip().lower() in ["sql", "postgresql", "psql"]:
            lines = lines[1:]
            generated_sql = "\n".join(lines).strip()

        # Check if it's a valid SQL query by looking for SQL keywords at the beginning
        # Common SQL query beginnings
        sql_keywords = [
            "select",
            "with",
            "insert",
            "update",
            "delete",
            "create",
            "alter",
            "drop",
            "explain",
        ]
        first_word = (
            generated_sql.split()[0].lower() if generated_sql.split() else ""
        )

        # If the response doesn't start with a SQL keyword, it's probably an explanation
        if first_word not in sql_keywords or generated_sql.startswith("To"):
            friendly_message = (
                "I'm not sure how to translate that into a SQL query. "
                "Could you try rephrasing your question to be more specific? "
                "For example, you can ask about specific tables like employees, projects, "
                "or their relationships. Questions like 'Show me the average salary by department' "
                "or 'List all projects with their budgets' work well."
            )

            # If the LLM provided its own explanation, use that instead
            if (
                "cannot generate" in generated_sql.lower()
                or "unable to create" in generated_sql.lower()
                or "can't create" in generated_sql.lower()
                or "i can't" in generated_sql.lower()
                or "please specify" in generated_sql.lower()
                or "need more information" in generated_sql.lower()
            ):
                friendly_message = generated_sql

            return {"success": False, "error": friendly_message}

        return {"success": True, "sql": generated_sql}

    except Exception as e:
        logger.error(f"Error generating SQL query: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"I encountered an error while processing your query. Please try asking a different question.",
        }
