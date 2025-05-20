"""
Natural Language to SQL RAG Engine

This package provides functionality to translate natural language questions into SQL queries,
execute them against a Snowflake database, and present the results in a user-friendly format.
"""

from .core.engine import NLSQLEngine
from .core.snowflake_connector import SnowflakeConnector
from .core.prompt_templates import SQL_GENERATION_PROMPT, RESULT_SUMMARIZATION_PROMPT

__version__ = "0.1.0"
