# Natural Language to SQL RAG Engine

This project implements a Retrieval Augmented Generation (RAG) system that allows users to query Snowflake databases using natural language. The system translates natural language questions into SQL queries, executes them against a Snowflake database, and then presents the results in a user-friendly natural language format.

## Architecture

1. **Input**: Natural language question from the user
2. **Processing**:
   - SQL Generation: Translates the natural language question to a SQL query using GPT-4o
   - Query Validation: Validates and constrains the SQL output using the Outlines library
   - Query Execution: Executes the query against a Snowflake database
   - Result Processing: Formats and summarizes the query results using GPT-4o
3. **Output**: Natural language summary of the query results with the raw data appended

## Components

- **Core RAG Engine**: Python library for NL-SQL-NL conversion
- **AWS Infrastructure**: Bedrock for LLM access, Lambda for serverless execution, API Gateway for web access
- **Web Chat Interface**: Simple chat interface for interacting with the system
- **Snowflake Integration**: Connector for executing SQL queries against Snowflake

## Example Snowflake Dataset

For demonstration purposes, you can use Snowflake's sample datasets. Here's how to access them:

1. Log in to your Snowflake account
2. Run the following SQL commands:

```sql
-- Set up the sample database
USE ROLE ACCOUNTADMIN;
CREATE DATABASE IF NOT EXISTS SNOWFLAKE_SAMPLE_DATA FROM SHARE SFC_SAMPLES.SAMPLE_DATA;

-- Switch to the sample database
USE DATABASE SNOWFLAKE_SAMPLE_DATA;
USE SCHEMA TPCDS_SF10TCL;

-- Example query to verify access
SELECT * FROM ITEM LIMIT 10;
```

This will give you access to the TPC-DS dataset, which includes tables like:
- `CUSTOMER`: Customer information
- `STORE_SALES`: Sales transactions
- `ITEM`: Product information
- `DATE_DIM`: Date dimension table

## Example Questions

Once set up, you can ask questions like:

- "What are the top 5 selling items in the last quarter?"
- "How many customers made purchases in California last month?"
- "What is the average sales amount by store type?"
- "Which product category has shown the highest growth this year?"

## How It Works

1. **Prompt Engineering**: The system uses carefully crafted prompts to instruct the LLM on how to generate SQL
2. **SQL Validation**: The Outlines library ensures that only valid SQL is generated
3. **Query Execution**: The SQL is executed against Snowflake
4. **Result Summarization**: Another prompt transforms the raw results into natural language

## Local Development

```bash
# Clone the repository
git clone <repository-url>
cd nl_sql_rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Test Snowflake connection
python scripts/test_snowflake_connection.py

# Run an example query
python scripts/example_query.py
```

## AWS Deployment

See the installation and setup instructions in the [INSTALL.md](./INSTALL.md) file for detailed deployment steps.

## License

MIT
