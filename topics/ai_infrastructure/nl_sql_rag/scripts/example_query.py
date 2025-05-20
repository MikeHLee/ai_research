#!/usr/bin/env python3
"""
Example script demonstrating how to use the NL-SQL-NL RAG engine.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the nl_sql_engine package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl_sql_engine.core.engine import NLSQLEngine
from nl_sql_engine.core.snowflake_connector import SnowflakeConnector


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Example natural language question
    question = "What are the top 5 products by sales in the last month?"
    
    print(f"Question: {question}\n")
    
    try:
        # Initialize Snowflake connector
        connector = SnowflakeConnector()
        
        # Initialize the NL-SQL-NL engine
        # By default, it will use OpenAI's API with the OPENAI_API_KEY from environment variables
        engine = NLSQLEngine(snowflake_connector=connector)
        
        # Process the query
        print("Processing query...\n")
        result = engine.process_query(question)
        
        # Print the results
        print("SQL Query:")
        print(result['sql_query'])
        print("\nSummary:")
        print(result['summary'])
        print("\nRaw Results (CSV):")
        print(result['raw_results'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
