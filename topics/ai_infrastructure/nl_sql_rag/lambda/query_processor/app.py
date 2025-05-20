import json
import os
import sys
import boto3
import logging
from botocore.exceptions import ClientError

# Add the nl_sql_engine package to the path
sys.path.append('/opt/python')

# Import the NLSQLEngine
from nl_sql_engine.core.engine import NLSQLEngine
from nl_sql_engine.core.snowflake_connector import SnowflakeConnector

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
secrets_client = boto3.client('secretsmanager')

def get_snowflake_credentials():
    """
    Get Snowflake credentials from AWS Secrets Manager.
    
    Returns:
        dict: Snowflake credentials
    """
    secret_name = os.environ.get('SNOWFLAKE_CREDENTIALS_SECRET')
    
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret_string = response['SecretString']
        return json.loads(secret_string)
    except ClientError as e:
        logger.error(f"Error retrieving Snowflake credentials: {str(e)}")
        raise

def lambda_handler(event, context):
    """
    Lambda function handler for processing natural language queries.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        dict: Response object with query results
    """
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
            
        # Get user question
        user_question = body.get('question')
        if not user_question:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: question'
                })
            }
        
        # Get Snowflake credentials
        snowflake_credentials = get_snowflake_credentials()
        
        # Initialize Snowflake connector
        snowflake_connector = SnowflakeConnector(
            account=snowflake_credentials.get('account'),
            user=snowflake_credentials.get('user'),
            password=snowflake_credentials.get('password'),
            database=snowflake_credentials.get('database'),
            schema=snowflake_credentials.get('schema'),
            warehouse=snowflake_credentials.get('warehouse')
        )
        
        # Initialize NLSQLEngine
        use_bedrock = os.environ.get('USE_BEDROCK', 'true').lower() == 'true'
        model_id = os.environ.get('MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        temperature = float(os.environ.get('TEMPERATURE', '0.0'))
        
        engine = NLSQLEngine(
            snowflake_connector=snowflake_connector,
            use_bedrock=use_bedrock,
            model_id=model_id,
            temperature=temperature,
            aws_region=os.environ.get('AWS_REGION', 'us-west-2')
        )
        
        # Process the query
        result = engine.process_query(user_question)
        
        # Return the response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
