#!/bin/bash

set -e

# Configuration
PROJECT_NAME="nl-sql-rag"
REGION="us-west-2"
STACK_NAME="${PROJECT_NAME}-stack"
CF_TEMPLATE="./cloudformation/template.yaml"
LAMBDA_CODE_DIR="../lambda"
LAYERS_DIR="../layers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq is not installed. Some features may not work correctly.${NC}"
fi

# Load environment variables from .env file if it exists
if [ -f "../.env" ]; then
    echo -e "${GREEN}Loading environment variables from .env file...${NC}"
    source "../.env"
fi

# Create S3 bucket for Lambda code and layers if it doesn't exist
S3_BUCKET="${PROJECT_NAME}-lambda-code-$(aws sts get-caller-identity --query 'Account' --output text)"
echo -e "${GREEN}Checking if S3 bucket ${S3_BUCKET} exists...${NC}"
if ! aws s3api head-bucket --bucket "${S3_BUCKET}" 2>/dev/null; then
    echo -e "${GREEN}Creating S3 bucket ${S3_BUCKET}...${NC}"
    aws s3api create-bucket \
        --bucket "${S3_BUCKET}" \
        --region "${REGION}" \
        --create-bucket-configuration LocationConstraint="${REGION}"
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "${S3_BUCKET}" \
        --versioning-configuration Status=Enabled
fi

# Build and package Lambda functions
echo -e "${GREEN}Building and packaging Lambda functions...${NC}"

# Create temporary directory for packaging
TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT

# Package query processor Lambda function
echo -e "${GREEN}Packaging query processor Lambda function...${NC}"
mkdir -p "${TMP_DIR}/query_processor"
cp -r "${LAMBDA_CODE_DIR}/query_processor/"* "${TMP_DIR}/query_processor/"
cd "${TMP_DIR}/query_processor"
zip -r "../query_processor.zip" .
cd -

# Package web interface Lambda function
echo -e "${GREEN}Packaging web interface Lambda function...${NC}"
mkdir -p "${TMP_DIR}/web_interface"
cp -r "../web_interface/lambda_app.py" "${TMP_DIR}/web_interface/"
cd "${TMP_DIR}/web_interface"
zip -r "../web_interface.zip" .
cd -

# Upload Lambda function code to S3
echo -e "${GREEN}Uploading Lambda function code to S3...${NC}"
aws s3 cp "${TMP_DIR}/query_processor.zip" "s3://${S3_BUCKET}/functions/query_processor.zip"
aws s3 cp "${TMP_DIR}/web_interface.zip" "s3://${S3_BUCKET}/functions/web_interface.zip"

# Package Lambda layers
echo -e "${GREEN}Packaging Lambda layers...${NC}"
mkdir -p "${TMP_DIR}/layers/python"
cp -r "${LAYERS_DIR}/dependencies/python/"* "${TMP_DIR}/layers/python/"
cd "${TMP_DIR}/layers"
zip -r "../dependencies.zip" .
cd -

# Upload Lambda layers to S3
echo -e "${GREEN}Uploading Lambda layers to S3...${NC}"
aws s3 cp "${TMP_DIR}/dependencies.zip" "s3://${S3_BUCKET}/layers/dependencies.zip"

# Prompt for Snowflake credentials if not in environment variables
if [ -z "${SNOWFLAKE_ACCOUNT}" ]; then
    read -p "Enter Snowflake account: " SNOWFLAKE_ACCOUNT
fi

if [ -z "${SNOWFLAKE_USER}" ]; then
    read -p "Enter Snowflake user: " SNOWFLAKE_USER
fi

if [ -z "${SNOWFLAKE_PASSWORD}" ]; then
    read -sp "Enter Snowflake password: " SNOWFLAKE_PASSWORD
    echo ""
fi

if [ -z "${SNOWFLAKE_DATABASE}" ]; then
    read -p "Enter Snowflake database: " SNOWFLAKE_DATABASE
fi

if [ -z "${SNOWFLAKE_SCHEMA}" ]; then
    read -p "Enter Snowflake schema: " SNOWFLAKE_SCHEMA
fi

if [ -z "${SNOWFLAKE_WAREHOUSE}" ]; then
    read -p "Enter Snowflake warehouse: " SNOWFLAKE_WAREHOUSE
fi

# Check if stack exists
STACK_EXISTS=$(aws cloudformation describe-stacks --region "${REGION}" --stack-name "${STACK_NAME}" 2>/dev/null || echo "")

if [ -z "${STACK_EXISTS}" ]; then
    # Create CloudFormation stack
    echo -e "${GREEN}Creating CloudFormation stack ${STACK_NAME}...${NC}"
    aws cloudformation create-stack \
        --stack-name "${STACK_NAME}" \
        --template-body "file://${CF_TEMPLATE}" \
        --parameters \
            ParameterKey=ProjectName,ParameterValue="${PROJECT_NAME}" \
            ParameterKey=SnowflakeAccount,ParameterValue="${SNOWFLAKE_ACCOUNT}" \
            ParameterKey=SnowflakeUser,ParameterValue="${SNOWFLAKE_USER}" \
            ParameterKey=SnowflakePassword,ParameterValue="${SNOWFLAKE_PASSWORD}" \
            ParameterKey=SnowflakeDatabase,ParameterValue="${SNOWFLAKE_DATABASE}" \
            ParameterKey=SnowflakeSchema,ParameterValue="${SNOWFLAKE_SCHEMA}" \
            ParameterKey=SnowflakeWarehouse,ParameterValue="${SNOWFLAKE_WAREHOUSE}" \
        --capabilities CAPABILITY_IAM \
        --region "${REGION}"

    echo -e "${GREEN}Waiting for stack creation to complete...${NC}"
    aws cloudformation wait stack-create-complete \
        --stack-name "${STACK_NAME}" \
        --region "${REGION}"
else
    # Update CloudFormation stack
    echo -e "${GREEN}Updating CloudFormation stack ${STACK_NAME}...${NC}"
    aws cloudformation update-stack \
        --stack-name "${STACK_NAME}" \
        --template-body "file://${CF_TEMPLATE}" \
        --parameters \
            ParameterKey=ProjectName,ParameterValue="${PROJECT_NAME}" \
            ParameterKey=SnowflakeAccount,ParameterValue="${SNOWFLAKE_ACCOUNT}" \
            ParameterKey=SnowflakeUser,ParameterValue="${SNOWFLAKE_USER}" \
            ParameterKey=SnowflakePassword,ParameterValue="${SNOWFLAKE_PASSWORD}" \
            ParameterKey=SnowflakeDatabase,ParameterValue="${SNOWFLAKE_DATABASE}" \
            ParameterKey=SnowflakeSchema,ParameterValue="${SNOWFLAKE_SCHEMA}" \
            ParameterKey=SnowflakeWarehouse,ParameterValue="${SNOWFLAKE_WAREHOUSE}" \
        --capabilities CAPABILITY_IAM \
        --region "${REGION}" || echo -e "${YELLOW}No updates to be performed.${NC}"

    # Only wait for stack update if an update was started
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Waiting for stack update to complete...${NC}"
        aws cloudformation wait stack-update-complete \
            --stack-name "${STACK_NAME}" \
            --region "${REGION}"
    fi
fi

# Get stack outputs
echo -e "${GREEN}Getting stack outputs...${NC}"
OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs" \
    --output json)

# Extract API endpoint and web interface URL
API_ENDPOINT=$(echo "${OUTPUTS}" | jq -r '.[] | select(.OutputKey=="ApiEndpoint") | .OutputValue')
WEB_INTERFACE_URL=$(echo "${OUTPUTS}" | jq -r '.[] | select(.OutputKey=="WebInterfaceURL") | .OutputValue')

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "API Endpoint: ${YELLOW}${API_ENDPOINT}${NC}"
echo -e "Web Interface URL: ${YELLOW}${WEB_INTERFACE_URL}${NC}"

# Deploy web interface
echo -e "${GREEN}Deploying web interface...${NC}"
WEB_BUCKET=$(echo "${WEB_INTERFACE_URL}" | sed 's/http:\/\///g' | sed 's/\.s3-website.*//g')
aws s3 sync "../web_interface/build/" "s3://${WEB_BUCKET}/" --delete

echo -e "${GREEN}Web interface deployed successfully!${NC}"
echo -e "You can access the web interface at: ${YELLOW}${WEB_INTERFACE_URL}${NC}"
