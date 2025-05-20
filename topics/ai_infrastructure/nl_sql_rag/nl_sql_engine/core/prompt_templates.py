"""
Prompt templates for the NL-SQL-NL RAG engine.
"""

SQL_GENERATION_PROMPT = """
You are a SQL expert tasked with translating natural language questions into SQL queries for a Snowflake database.

DATABASE SCHEMA:
{schema_description}

TABLES:
{table_descriptions}

TASK:
Convert the following natural language question into a valid SQL query that will run on Snowflake.
Only return the SQL query without any explanations or markdown formatting.
Ensure your query is complete, correct, and optimized.

QUESTION:
{user_question}

SQL QUERY:
"""

RESULT_SUMMARIZATION_PROMPT = """
You are a data analyst tasked with explaining query results in natural language.

ORIGINAL QUESTION:
{user_question}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{query_results}

TASK:
Provide a clear, concise summary of the query results that directly answers the original question.
Include key insights, patterns, or notable findings from the data.
Be specific and reference actual values from the results.
Keep your explanation conversational and user-friendly.

SUMMARY:
"""
