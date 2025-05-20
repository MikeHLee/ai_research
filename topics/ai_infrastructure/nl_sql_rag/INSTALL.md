# Installation and Setup Guide

This guide will walk you through setting up the Natural Language to SQL RAG Engine.

## Prerequisites

- Python 3.9+
- AWS Account with appropriate permissions
- Snowflake Account with database access
- AWS CLI configured on your local machine

## Local Development Setup

1. Clone the repository and navigate to the project directory:

```bash
cd /path/to/nl_sql_rag
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with the following variables:

```
# AWS Configuration
AWS_REGION=us-west-2
AWS_PROFILE=default

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse

# OpenAI Configuration (for local development)
OPENAI_API_KEY=your_openai_api_key
```

## AWS Infrastructure Deployment

1. Deploy the AWS resources:

```bash
cd infrastructure
./deploy.sh
```

This script will:
- Create necessary IAM roles and policies
- Deploy Lambda functions
- Set up API Gateway
- Configure Bedrock model access

2. After deployment, you'll receive the API endpoint URL for your chat interface.

## Running the Chat Interface Locally

1. Start the local development server:

```bash
cd chat_interface
python app.py
```

2. Open your browser and navigate to `http://localhost:8000`

## Connecting to Snowflake

1. Ensure your Snowflake credentials are correctly set in the `.env` file
2. Run the test connection script:

```bash
python scripts/test_snowflake_connection.py
```

If successful, you should see a confirmation message and a list of available tables.

## Troubleshooting

If you encounter issues:

1. Check the logs in AWS CloudWatch
2. Verify your AWS credentials are correctly configured
3. Ensure your Snowflake credentials are correct
4. Check that the required ports are open in your firewall

For more detailed troubleshooting, refer to the documentation in the `docs` directory.
