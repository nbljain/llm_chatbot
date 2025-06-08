"""
Database relationships configuration for SQL Chatbot.
Define table relationships to help the AI generate proper JOIN queries.
"""

import os
import json
from typing import Dict, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TableRelationships:
    """Manages table relationships for better SQL generation"""

    def __init__(self):
        self.relationships = {}
        self.load_relationships()

    def load_relationships(self):
        """Load relationships from configuration file or environment"""
        # Try to load from file first
        config_file = "table_relationships.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    self.relationships = json.load(f)
                logger.info(f"Loaded table relationships from {config_file}")
                return
            except Exception as e:
                logger.warning(f"Error loading relationships from file: {e}")

        # Fallback to environment variable
        relationships_json = os.environ.get("TABLE_RELATIONSHIPS", "{}")
        try:
            self.relationships = json.loads(relationships_json)
            logger.info("Loaded table relationships from environment variable")
        except Exception as e:
            logger.warning(f"Error parsing TABLE_RELATIONSHIPS: {e}")
            self.relationships = {}

    def add_relationship(
        self,
        table1: str,
        table2: str,
        join_condition: str,
        relationship_type: str = "inner",
    ):
        """
        Add a relationship between two tables

        Args:
            table1: First table name
            table2: Second table name
            join_condition: JOIN condition (e.g., "table1.id = table2.table1_id")
            relationship_type: Type of join (inner, left, right, full)
        """
        if table1 not in self.relationships:
            self.relationships[table1] = {}

        self.relationships[table1][table2] = {
            "join_condition": join_condition,
            "relationship_type": relationship_type,
        }

        # Add reverse relationship for bidirectional joins
        if table2 not in self.relationships:
            self.relationships[table2] = {}

        self.relationships[table2][table1] = {
            "join_condition": join_condition,
            "relationship_type": relationship_type,
        }

    def get_relationship(self, table1: str, table2: str) -> Optional[Dict]:
        """Get relationship between two tables"""
        return self.relationships.get(table1, {}).get(table2)

    def get_related_tables(self, table: str) -> List[str]:
        """Get all tables related to the given table"""
        return list(self.relationships.get(table, {}).keys())

    def get_relationship_description(self) -> str:
        """Get a formatted description of all relationships for the LLM"""
        if not self.relationships:
            return "No table relationships defined."

        description = "Table Relationships:\n"
        for table1, related_tables in self.relationships.items():
            for table2, relationship in related_tables.items():
                join_condition = relationship["join_condition"]
                join_type = relationship["relationship_type"]
                description += f"- {table1} {join_type.upper()} JOIN {table2} ON {join_condition}\n"

        return description

    def save_relationships(self, filename: str = "table_relationships.json"):
        """Save relationships to a JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.relationships, f, indent=2)
            logger.info(f"Saved table relationships to {filename}")
        except Exception as e:
            logger.error(f"Error saving relationships: {e}")


# Global instance
table_relationships = TableRelationships()


def setup_default_relationships():
    """Setup some common relationship patterns"""
    # Example relationships - these will be overridden by user configuration
    table_relationships.add_relationship(
        "customers", "orders", "customers.customer_id = orders.customer_id", "left"
    )
    table_relationships.add_relationship(
        "orders", "order_items", "orders.order_id = order_items.order_id", "left"
    )
    table_relationships.add_relationship(
        "products",
        "order_items",
        "products.product_id = order_items.product_id",
        "inner",
    )


def get_relationships_for_llm() -> str:
    """Get relationship description formatted for LLM context"""
    return table_relationships.get_relationship_description()
