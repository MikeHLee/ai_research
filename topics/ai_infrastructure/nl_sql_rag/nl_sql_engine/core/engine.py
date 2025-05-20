"""
Core engine for the NL-SQL-NL RAG system.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import outlines
import outlines.models as models
import boto3
from botocore.config import Config
import openai

from .snowflake_connector import SnowflakeConnector
from .prompt_templates import SQL_GENERATION_PROMPT, RESULT_SUMMARIZATION_PROMPT


class NLSQLEngine:
    """
    Core engine for translating natural language to SQL and back.
    """
    
    def __init__(
        self,
        snowflake_connector: Optional[SnowflakeConnector] = None,
        use_bedrock: bool = False,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        temperature: float = 0.0,
        openai_api_key: Optional[str] = None,
        aws_region: Optional[str] = None
    ):
        """
        Initialize the NL-SQL-NL RAG engine.
        
        Args:
            snowflake_connector: SnowflakeConnector instance for database access
            use_bedrock: Whether to use AWS Bedrock (True) or OpenAI API (False)
            model_id: Model ID to use with AWS Bedrock
            temperature: Temperature parameter for the LLM
            openai_api_key: OpenAI API key (required if use_bedrock is False)
            aws_region: AWS region for Bedrock (required if use_bedrock is True)
        """
        self.snowflake = snowflake_connector or SnowflakeConnector()
        self.use_bedrock = use_bedrock
        self.model_id = model_id
        self.temperature = temperature
        
        # Set up the appropriate client based on configuration
        if use_bedrock:
            self.aws_region = aws_region or os.environ.get('AWS_REGION', 'us-west-2')
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.aws_region,
                config=Config(
                    retries={'max_attempts': 3, 'mode': 'standard'}
                )
            )
        else:
            self.openai_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required when use_bedrock is False")
            openai.api_key = self.openai_api_key
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The model's response
        """
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    def _call_bedrock(self, prompt: str) -> str:
        """
        Call the AWS Bedrock API with the given prompt.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The model's response
        """
        # Determine if we're using Claude or other model and format accordingly
        if "anthropic.claude" in self.model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": self.temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        else:
            # Default format for other models
            body = json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 4096,
                "temperature": self.temperature
            })
        
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        
        # Extract the content based on model type
        if "anthropic.claude" in self.model_id:
            return response_body.get('content')[0].get('text')
        else:
            return response_body.get('completion')
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the appropriate LLM based on configuration.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The model's response
        """
        if self.use_bedrock:
            return self._call_bedrock(prompt)
        else:
            return self._call_openai(prompt)
    
    def generate_sql_with_outlines(self, user_question: str, schema_description: str, table_descriptions: str) -> str:
        """
        Generate SQL using the Outlines library for structured output.
        
        Args:
            user_question: The natural language question
            schema_description: Description of the database schema
            table_descriptions: Descriptions of the tables
            
        Returns:
            The generated SQL query
        """
        # Define the SQL grammar using Outlines
        sql_grammar = outlines.generate.regex(r"SELECT[\s\S]*?;")
        
        # Set up the model
        if self.use_bedrock:
            # For Bedrock, we need to use a custom model implementation
            # This is a simplified example and may need adjustment
            model = models.Custom(self._call_bedrock)
        else:
            # For OpenAI, we can use the built-in model
            model = models.OpenAI(model="gpt-4o", api_key=self.openai_api_key)
        
        # Format the prompt
        prompt = SQL_GENERATION_PROMPT.format(
            schema_description=schema_description,
            table_descriptions=table_descriptions,
            user_question=user_question
        )
        
        # Generate SQL with constraints
        sql_query = outlines.generate.text(model, prompt, grammar=sql_grammar)
        
        return sql_query
    
    def process_query(self, user_question: str) -> Dict:
        """
        Process a natural language query end-to-end.
        
        Args:
            user_question: The natural language question from the user
            
        Returns:
            Dictionary containing the original question, SQL query, raw results, and summary
        """
        # Get schema information
        schema_description = self.snowflake.get_schema_description()
        table_descriptions = self.snowflake.format_schema_for_prompt()
        
        # Generate SQL query
        sql_query = self.generate_sql_with_outlines(
            user_question=user_question,
            schema_description=schema_description,
            table_descriptions=table_descriptions
        )
        
        # Execute the query
        try:
            results_df = self.snowflake.execute_query(sql_query)
            csv_results = results_df.to_csv(index=False)
            
            # Convert DataFrame to a readable format for the prompt
            if len(results_df) > 10:
                # If more than 10 rows, include first 5 and last 5
                display_df = pd.concat([results_df.head(5), results_df.tail(5)])
                results_str = display_df.to_string(index=False)
                results_str += f"\n\n[Note: Showing 10 of {len(results_df)} rows]"
            else:
                results_str = results_df.to_string(index=False)
            
            # Generate summary
            summary_prompt = RESULT_SUMMARIZATION_PROMPT.format(
                user_question=user_question,
                sql_query=sql_query,
                query_results=results_str
            )
            
            summary = self._call_llm(summary_prompt)
            
            return {
                "question": user_question,
                "sql_query": sql_query,
                "raw_results": csv_results,
                "summary": summary,
                "success": True
            }
            
        except Exception as e:
            error_message = str(e)
            
            # Generate an error summary
            error_prompt = f"""
            The following SQL query failed to execute:
            
            {sql_query}
            
            The error was:
            {error_message}
            
            Please provide a user-friendly explanation of what went wrong and how it might be fixed.
            """
            
            error_summary = self._call_llm(error_prompt)
            
            return {
                "question": user_question,
                "sql_query": sql_query,
                "raw_results": None,
                "summary": error_summary,
                "success": False,
                "error": error_message
            }
