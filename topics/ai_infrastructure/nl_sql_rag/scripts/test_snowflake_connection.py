#!/usr/bin/env python3
"""
Test script to verify Snowflake connection and retrieve schema information.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the nl_sql_engine package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nl_sql_engine.core.snowflake_connector import SnowflakeConnector


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = [
        'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_USER',
        'SNOWFLAKE_PASSWORD',
        'SNOWFLAKE_DATABASE',
        'SNOWFLAKE_SCHEMA',
        'SNOWFLAKE_WAREHOUSE'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Error: The following environment variables are missing: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        sys.exit(1)
    
    try:
        # Initialize Snowflake connector
        connector = SnowflakeConnector()
        
        # Test connection
        print("Testing Snowflake connection...")
        connector.connect()
        print("Connection successful!")
        
        # Get schema description
        print("\nRetrieving schema description...")
        schema_description = connector.get_schema_description()
        print(schema_description)
        
        # Get table descriptions
        print("\nRetrieving table descriptions...")
        table_descriptions = connector.get_table_descriptions()
        
        # Print table names
        print(f"Found {len(table_descriptions)} tables:")
        for table_name in table_descriptions.keys():
            print(f"- {table_name}")
        
        # Print detailed information for the first table
        if table_descriptions:
            first_table = next(iter(table_descriptions))
            print(f"\nDetailed information for table '{first_table}':")
            print(json.dumps(table_descriptions[first_table], indent=2))
        
        # Close connection
        connector.disconnect()
        print("\nConnection closed.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
