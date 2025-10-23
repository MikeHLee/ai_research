# MCP Tool Use Fine-tuning Dataset

This dataset contains training examples for fine-tuning LLMs to use MCP (Model Context Protocol) tools effectively.

## Overview

- **Format**: JSONL (JSON Lines)
- **Examples**: 28 training samples
- **System Prompt Length**: ~3,845 characters
- **Tool Coverage**: 11 MCP tools

## Dataset Structure

Each line in `mcp_finetuning_dataset.jsonl` contains a JSON object with:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<System prompt with all tool definitions>"
    },
    {
      "role": "user",
      "content": "<User request>"
    },
    {
      "role": "assistant",
      "content": "",
      "toolCalls": [
        {
          "id": "call_X",
          "type": "function",
          "function": {
            "name": "<tool_name>",
            "arguments": "<JSON string of arguments>"
          }
        }
      ]
    }
  ]
}
```

## Tools Covered

### File Operations
- **read_file**: Read file contents with optional line ranges
- **write_to_file**: Create new files with content
- **edit**: Perform exact string replacements in files

### Search & Discovery
- **grep_search**: Search for text patterns using ripgrep
- **find_by_name**: Find files/directories by name patterns
- **list_dir**: List directory contents
- **find_code_context**: Natural language code search

### Command Execution
- **run_command**: Execute terminal commands with safety controls

### Web Access
- **search_web**: Perform web searches
- **read_url_content**: Fetch content from URLs

### Memory
- **create_memory**: Save context to persistent memory

## Key Features

1. **Complete Tool Signatures**: Every example includes full tool definitions in the system prompt
2. **Diverse Use Cases**: Covers single-tool and multi-tool scenarios
3. **Realistic Requests**: Natural language user requests
4. **Proper JSON Format**: Arguments are JSON-encoded strings
5. **Safety Considerations**: Includes SafeToAutoRun flags for commands

## Example Categories

- **File Reading** (2 examples): Basic and range-based file reading
- **Search Operations** (3 examples): Text search, regex patterns, filtered searches
- **File Discovery** (3 examples): Pattern matching, extension filtering
- **Directory Listing** (2 examples): Basic directory exploration
- **File Creation** (3 examples): Code files, components, empty files
- **File Editing** (2 examples): Single and bulk replacements
- **Command Execution** (4 examples): npm commands, git operations
- **Web Operations** (4 examples): Web search and URL reading
- **Code Context** (2 examples): Natural language code search
- **Memory** (1 example): Context persistence
- **Multi-tool** (2 examples): Combined tool usage

## Usage

### For Fine-tuning

This dataset is designed for supervised fine-tuning of LLMs to learn:
1. When to use specific tools based on user intent
2. How to construct proper tool call arguments
3. How to handle multi-tool scenarios

### Training Format

The format is compatible with:
- OpenAI fine-tuning API
- Hugging Face transformers
- Custom training pipelines

### Generation Script

The dataset is generated using `generate_dataset.py`, which:
- Loads tool definitions from `tool_definitions.json`
- Formats them into readable system prompts
- Creates training examples with proper JSON encoding
- Outputs to `mcp_finetuning_dataset.jsonl`

## Validation

To validate the dataset format:

```bash
# Check JSON validity
head -1 mcp_finetuning_dataset.jsonl | python3 -m json.tool

# Count examples
wc -l mcp_finetuning_dataset.jsonl

# View system prompt length
python3 -c "import json; print(len(json.loads(open('mcp_finetuning_dataset.jsonl').readline())['messages'][0]['content']))"
```

## Extending the Dataset

To add more examples:

1. Edit `generate_dataset.py`
2. Add new examples to the `examples` list
3. Run: `../venv/bin/python generate_dataset.py`

To add new tools:

1. Update `tool_definitions.json` with the new tool schema
2. Add examples using the new tool to `generate_dataset.py`
3. Regenerate the dataset

## Notes

- All file paths use `/workspace` as the base directory
- Tool arguments are JSON-encoded strings (not objects)
- The system prompt is identical across all examples for consistency
- Boolean values in Python (True/False) are converted to JSON (true/false) via `json.dumps()`

## License

This dataset is created for research and fine-tuning purposes based on Windsurf/Cascade MCP tool definitions.
