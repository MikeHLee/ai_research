"""
Snowflake connector for executing SQL queries and retrieving schema information.
"""

import os
import pandas as pd
import snowflake.connector
from typing import Dict, List, Optional, Tuple, Union
import json


class SnowflakeConnector:
    """
    Handles connections to Snowflake and executes queries.
    """
    
    def __init__(
        self,
        account: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        schema: str = None,
        warehouse: str = None,
        role: str = None
    ):
        """
        Initialize the Snowflake connector with connection parameters.
        
        If parameters are not provided, they will be read from environment variables.
        """
        self.account = account or os.environ.get('SNOWFLAKE_ACCOUNT')
        self.user = user or os.environ.get('SNOWFLAKE_USER')
        self.password = password or os.environ.get('SNOWFLAKE_PASSWORD')
        self.database = database or os.environ.get('SNOWFLAKE_DATABASE')
        self.schema = schema or os.environ.get('SNOWFLAKE_SCHEMA')
        self.warehouse = warehouse or os.environ.get('SNOWFLAKE_WAREHOUSE')
        self.role = role or os.environ.get('SNOWFLAKE_ROLE')
        
        self._conn = None
        self._schema_cache = None
    
    def connect(self) -> None:
        """
        Establish a connection to Snowflake.
        """
        if self._conn is None:
            conn_params = {
                'account': self.account,
                'user': self.user,
                'password': self.password,
                'database': self.database,
                'schema': self.schema,
                'warehouse': self.warehouse
            }
            
            if self.role:
                conn_params['role'] = self.role
                
            self._conn = snowflake.connector.connect(**conn_params)
    
    def disconnect(self) -> None:
        """
        Close the Snowflake connection.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing query results
        """
        self.connect()
        
        try:
            cursor = self._conn.cursor()
            cursor.execute(query)
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=column_names)
            
            cursor.close()
            return df
            
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")
    
    def get_schema_description(self) -> str:
        """
        Get a description of the database schema.
        
        Returns:
            String describing the database schema
        """
        self.connect()
        
        try:
            cursor = self._conn.cursor()
            
            # Get database info
            cursor.execute(f"SELECT DATABASE_NAME, DATABASE_OWNER, CREATED FROM INFORMATION_SCHEMA.DATABASES WHERE DATABASE_NAME = '{self.database}'")
            db_info = cursor.fetchone()
            
            description = f"Database: {db_info[0]}, Owner: {db_info[1]}, Created: {db_info[2]}\n"
            description += f"Schema: {self.schema}\n"
            
            cursor.close()
            return description
            
        except Exception as e:
            raise Exception(f"Error getting schema description: {str(e)}")
    
    def get_table_descriptions(self) -> Dict[str, Dict]:
        """
        Get descriptions of all tables in the current schema.
        
        Returns:
            Dictionary mapping table names to their descriptions
        """
        if self._schema_cache is not None:
            return self._schema_cache
            
        self.connect()
        
        try:
            cursor = self._conn.cursor()
            
            # Get list of tables
            cursor.execute(f"SHOW TABLES IN {self.database}.{self.schema}")
            tables = cursor.fetchall()
            
            table_descriptions = {}
            
            for table_info in tables:
                table_name = table_info[1]
                
                # Get column information
                cursor.execute(f"DESCRIBE TABLE {self.database}.{self.schema}.{table_name}")
                columns = cursor.fetchall()
                
                column_info = []
                for col in columns:
                    column_info.append({
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[3] == "Y"
                    })
                
                # Get primary key information
                cursor.execute(f"""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                        ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                    WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                    AND tc.TABLE_SCHEMA = '{self.schema}'
                    AND tc.TABLE_NAME = '{table_name}'
                """)
                
                primary_keys = [row[0] for row in cursor.fetchall()]
                
                # Get sample data (first 5 rows)
                try:
                    cursor.execute(f"SELECT * FROM {self.database}.{self.schema}.{table_name} LIMIT 5")
                    sample_data = cursor.fetchall()
                    
                    sample_rows = []
                    for row in sample_data:
                        sample_row = {}
                        for i, col in enumerate([c[0] for c in cursor.description]):
                            sample_row[col] = str(row[i])
                        sample_rows.append(sample_row)
                        
                except:
                    sample_rows = []
                
                table_descriptions[table_name] = {
                    "columns": column_info,
                    "primary_keys": primary_keys,
                    "sample_data": sample_rows
                }
            
            cursor.close()
            
            self._schema_cache = table_descriptions
            return table_descriptions
            
        except Exception as e:
            raise Exception(f"Error getting table descriptions: {str(e)}")
    
    def format_schema_for_prompt(self) -> str:
        """
        Format the schema information for inclusion in a prompt.
        
        Returns:
            Formatted schema string
        """
        table_descriptions = self.get_table_descriptions()
        
        result = []
        for table_name, table_info in table_descriptions.items():
            table_str = f"Table: {table_name}\n"
            table_str += "Columns:\n"
            
            for col in table_info["columns"]:
                pk_marker = " (PK)" if col["name"] in table_info["primary_keys"] else ""
                nullable = "" if col["nullable"] else " NOT NULL"
                table_str += f"  - {col['name']}: {col['type']}{pk_marker}{nullable}\n"
            
            if table_info["sample_data"]:
                table_str += "Sample Data:\n"
                for i, row in enumerate(table_info["sample_data"][:3]):  # Limit to 3 samples to keep prompt size reasonable
                    table_str += f"  Row {i+1}: {json.dumps(row)}\n"
            
            result.append(table_str)
        
        return "\n".join(result)
    
    def execute_query_to_csv(self, query: str) -> str:
        """
        Execute a SQL query and return the results as a CSV string.
        
        Args:
            query: SQL query to execute
            
        Returns:
            CSV string containing query results
        """
        df = self.execute_query(query)
        return df.to_csv(index=False)
