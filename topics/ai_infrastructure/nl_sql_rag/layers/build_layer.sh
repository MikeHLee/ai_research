#!/bin/bash

set -e

# Configuration
LAYER_DIR="dependencies"
PYTHON_DIR="${LAYER_DIR}/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install it first.${NC}"
    exit 1
fi

# Create the layer directory structure
echo -e "${GREEN}Creating layer directory structure...${NC}"
mkdir -p "${PYTHON_DIR}"

# Install dependencies to the layer directory
echo -e "${GREEN}Installing dependencies to the layer directory...${NC}"
pip install -r ../requirements.txt -t "${PYTHON_DIR}" --upgrade

# Copy the nl_sql_engine package to the layer directory
echo -e "${GREEN}Copying nl_sql_engine package to the layer directory...${NC}"
cp -r ../nl_sql_engine "${PYTHON_DIR}/"

# Remove unnecessary files to reduce the layer size
echo -e "${GREEN}Cleaning up unnecessary files...${NC}"
find "${PYTHON_DIR}" -name "__pycache__" -type d -exec rm -rf {} +
find "${PYTHON_DIR}" -name "*.pyc" -delete
find "${PYTHON_DIR}" -name "*.pyo" -delete
find "${PYTHON_DIR}" -name "*.dist-info" -type d -exec rm -rf {} +
find "${PYTHON_DIR}" -name "*.egg-info" -type d -exec rm -rf {} +
find "${PYTHON_DIR}" -name "tests" -type d -exec rm -rf {} +

echo -e "${GREEN}Layer build completed successfully!${NC}"
echo -e "Layer directory: ${YELLOW}${LAYER_DIR}${NC}"
