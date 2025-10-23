#!/usr/bin/env python3
"""Generate MCP tool use fine-tuning dataset with tool signatures."""

import json

# System prompt with tool definitions
SYSTEM_PROMPT_TEMPLATE = """You are Cascade, an AI coding assistant with access to MCP tools. You help developers by using the available tools to complete tasks.

Available tools:
{tool_definitions}

Use these tools to help the user with their request."""

# Load tool definitions
with open('tool_definitions.json', 'r') as f:
    TOOLS = json.load(f)

def format_tool_for_prompt(tool_name, tool_def):
    """Format a single tool definition for the system prompt."""
    params = tool_def['parameters']['properties']
    required = tool_def['parameters'].get('required', [])
    
    param_list = []
    for param_name, param_info in params.items():
        req_marker = " (required)" if param_name in required else " (optional)"
        param_type = param_info.get('type', 'any')
        param_desc = param_info.get('description', '')
        param_list.append(f"  - {param_name} ({param_type}){req_marker}: {param_desc}")
    
    return f"""- {tool_name}: {tool_def['description']}
{chr(10).join(param_list)}"""

def get_system_prompt():
    """Generate system prompt with all tool definitions."""
    tool_defs = "\n\n".join([format_tool_for_prompt(name, defn) for name, defn in TOOLS.items()])
    return SYSTEM_PROMPT_TEMPLATE.format(tool_definitions=tool_defs)

# Training examples
examples = [
    # File reading examples
    {
        "user": "Read the contents of src/main.py",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"file_path": "/workspace/src/main.py"})
            }
        }]
    },
    {
        "user": "Show me lines 50-100 of config.json",
        "tool_calls": [{
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"file_path": "/workspace/config.json", "offset": 50, "limit": 50})
            }
        }]
    },
    
    # Search examples
    {
        "user": "Search for all TODO comments in the codebase",
        "tool_calls": [{
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "grep_search",
                "arguments": json.dumps({
                    "Query": "TODO",
                    "SearchPath": "/workspace",
                    "CaseInsensitive": True
                })
            }
        }]
    },
    {
        "user": "Find all async function definitions in Python files",
        "tool_calls": [{
            "id": "call_4",
            "type": "function",
            "function": {
                "name": "grep_search",
                "arguments": json.dumps({
                    "Query": "async def",
                    "SearchPath": "/workspace",
                    "Includes": ["*.py"],
                    "IsRegex": False
                })
            }
        }]
    },
    {
        "user": "Search for error handling patterns using regex",
        "tool_calls": [{
            "id": "call_5",
            "type": "function",
            "function": {
                "name": "grep_search",
                "arguments": json.dumps({
                    "Query": "try\\s*{|catch\\s*\\(|except ",
                    "SearchPath": "/workspace",
                    "IsRegex": True
                })
            }
        }]
    },
    
    # Find by name examples
    {
        "user": "Find all TypeScript configuration files",
        "tool_calls": [{
            "id": "call_6",
            "type": "function",
            "function": {
                "name": "find_by_name",
                "arguments": json.dumps({
                    "SearchDirectory": "/workspace",
                    "Pattern": "tsconfig*.json",
                    "Type": "file"
                })
            }
        }]
    },
    {
        "user": "Find all test files in the project",
        "tool_calls": [{
            "id": "call_7",
            "type": "function",
            "function": {
                "name": "find_by_name",
                "arguments": json.dumps({
                    "SearchDirectory": "/workspace",
                    "Pattern": "*.test.*",
                    "Extensions": ["ts", "js", "tsx", "jsx"],
                    "Type": "file"
                })
            }
        }]
    },
    {
        "user": "Find all Python files in the src directory",
        "tool_calls": [{
            "id": "call_8",
            "type": "function",
            "function": {
                "name": "find_by_name",
                "arguments": json.dumps({
                    "SearchDirectory": "/workspace/src",
                    "Extensions": ["py"],
                    "Type": "file"
                })
            }
        }]
    },
    
    # Directory listing examples
    {
        "user": "List all files in the components directory",
        "tool_calls": [{
            "id": "call_9",
            "type": "function",
            "function": {
                "name": "list_dir",
                "arguments": json.dumps({"DirectoryPath": "/workspace/components"})
            }
        }]
    },
    {
        "user": "What's in the current directory?",
        "tool_calls": [{
            "id": "call_10",
            "type": "function",
            "function": {
                "name": "list_dir",
                "arguments": json.dumps({"DirectoryPath": "/workspace"})
            }
        }]
    },
    
    # File creation examples
    {
        "user": "Create a utility file with date formatting functions",
        "tool_calls": [{
            "id": "call_11",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": json.dumps({
                    "TargetFile": "/workspace/utils/dateFormatter.ts",
                    "CodeContent": "export function formatDate(date: Date): string {\n  return date.toISOString().split('T')[0];\n}\n\nexport function formatDateTime(date: Date): string {\n  return date.toLocaleString('en-US', {\n    year: 'numeric',\n    month: '2-digit',\n    day: '2-digit',\n    hour: '2-digit',\n    minute: '2-digit'\n  });\n}"
                })
            }
        }]
    },
    {
        "user": "Create a React Button component",
        "tool_calls": [{
            "id": "call_12",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": json.dumps({
                    "TargetFile": "/workspace/components/Button.tsx",
                    "CodeContent": "import React from 'react';\n\ninterface ButtonProps {\n  label: string;\n  onClick: () => void;\n  variant?: 'primary' | 'secondary';\n  disabled?: boolean;\n}\n\nexport const Button: React.FC<ButtonProps> = ({\n  label,\n  onClick,\n  variant = 'primary',\n  disabled = false\n}) => {\n  return (\n    <button\n      className={`btn btn-${variant}`}\n      onClick={onClick}\n      disabled={disabled}\n    >\n      {label}\n    </button>\n  );\n};"
                })
            }
        }]
    },
    {
        "user": "Create an empty .gitkeep file in the logs directory",
        "tool_calls": [{
            "id": "call_13",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": json.dumps({
                    "TargetFile": "/workspace/logs/.gitkeep",
                    "EmptyFile": True
                })
            }
        }]
    },
    
    # File editing examples
    {
        "user": "Change the port from 3000 to 8080 in server.js",
        "tool_calls": [{
            "id": "call_14",
            "type": "function",
            "function": {
                "name": "edit",
                "arguments": json.dumps({
                    "file_path": "/workspace/server.js",
                    "old_string": "const PORT = 3000;",
                    "new_string": "const PORT = 8080;",
                    "explanation": "Update server port from 3000 to 8080"
                })
            }
        }]
    },
    {
        "user": "Rename all occurrences of oldFunction to newFunction in utils.ts",
        "tool_calls": [{
            "id": "call_15",
            "type": "function",
            "function": {
                "name": "edit",
                "arguments": json.dumps({
                    "file_path": "/workspace/utils.ts",
                    "old_string": "oldFunction",
                    "new_string": "newFunction",
                    "replace_all": True,
                    "explanation": "Rename oldFunction to newFunction throughout file"
                })
            }
        }]
    },
    
    # Command execution examples
    {
        "user": "Run npm install",
        "tool_calls": [{
            "id": "call_16",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({
                    "CommandLine": "npm install",
                    "Cwd": "/workspace",
                    "Blocking": False,
                    "SafeToAutoRun": False
                })
            }
        }]
    },
    {
        "user": "Run the test suite",
        "tool_calls": [{
            "id": "call_17",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({
                    "CommandLine": "npm test",
                    "Cwd": "/workspace",
                    "Blocking": False,
                    "SafeToAutoRun": False
                })
            }
        }]
    },
    {
        "user": "Check git status",
        "tool_calls": [{
            "id": "call_18",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({
                    "CommandLine": "git status",
                    "Cwd": "/workspace",
                    "Blocking": True,
                    "SafeToAutoRun": True
                })
            }
        }]
    },
    {
        "user": "Show the last 10 git commits",
        "tool_calls": [{
            "id": "call_19",
            "type": "function",
            "function": {
                "name": "run_command",
                "arguments": json.dumps({
                    "CommandLine": "git log --oneline -10",
                    "Cwd": "/workspace",
                    "Blocking": True,
                    "SafeToAutoRun": True
                })
            }
        }]
    },
    
    # Web search examples
    {
        "user": "Search for React hooks best practices",
        "tool_calls": [{
            "id": "call_20",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({
                    "query": "React hooks best practices 2024"
                })
            }
        }]
    },
    {
        "user": "Look up TypeScript generics documentation",
        "tool_calls": [{
            "id": "call_21",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({
                    "query": "TypeScript generics documentation",
                    "domain": "typescriptlang.org"
                })
            }
        }]
    },
    
    # URL reading examples
    {
        "user": "Fetch the content from the GitHub API for this repo",
        "tool_calls": [{
            "id": "call_22",
            "type": "function",
            "function": {
                "name": "read_url_content",
                "arguments": json.dumps({
                    "Url": "https://api.github.com/repos/owner/repo"
                })
            }
        }]
    },
    {
        "user": "Read the MDN documentation for Array.map",
        "tool_calls": [{
            "id": "call_23",
            "type": "function",
            "function": {
                "name": "read_url_content",
                "arguments": json.dumps({
                    "Url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map"
                })
            }
        }]
    },
    
    # Code context search examples
    {
        "user": "Find where authentication is handled in the codebase",
        "tool_calls": [{
            "id": "call_24",
            "type": "function",
            "function": {
                "name": "find_code_context",
                "arguments": json.dumps({
                    "search_term": "Find where user authentication and login is handled",
                    "search_folder_absolute_uri": "/workspace"
                })
            }
        }]
    },
    {
        "user": "Locate the database connection setup code",
        "tool_calls": [{
            "id": "call_25",
            "type": "function",
            "function": {
                "name": "find_code_context",
                "arguments": json.dumps({
                    "search_term": "Find database connection initialization and configuration",
                    "search_folder_absolute_uri": "/workspace/src"
                })
            }
        }]
    },
    
    # Memory creation examples
    {
        "user": "Remember that we're using PostgreSQL for the database",
        "tool_calls": [{
            "id": "call_26",
            "type": "function",
            "function": {
                "name": "create_memory",
                "arguments": json.dumps({
                    "Action": "create",
                    "Title": "Database Technology",
                    "Content": "This project uses PostgreSQL as the primary database",
                    "Tags": ["database", "postgresql", "infrastructure"],
                    "CorpusNames": ["workspace"]
                })
            }
        }]
    },
    
    # Multi-tool examples
    {
        "user": "Find and read the main configuration file",
        "tool_calls": [
            {
                "id": "call_27a",
                "type": "function",
                "function": {
                    "name": "find_by_name",
                    "arguments": json.dumps({
                        "SearchDirectory": "/workspace",
                        "Pattern": "config.*",
                        "Type": "file"
                    })
                }
            }
        ]
    },
    {
        "user": "Search for all API endpoints and list the routes directory",
        "tool_calls": [
            {
                "id": "call_28a",
                "type": "function",
                "function": {
                    "name": "grep_search",
                    "arguments": json.dumps({
                        "Query": "@app.route|@router.get|@router.post|app.get\\(|app.post\\(",
                        "SearchPath": "/workspace",
                        "IsRegex": True
                    })
                }
            },
            {
                "id": "call_28b",
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "arguments": json.dumps({
                        "DirectoryPath": "/workspace/routes"
                    })
                }
            }
        ]
    }
]

# Generate dataset
system_prompt = get_system_prompt()

output_file = 'mcp_finetuning_dataset.jsonl'
with open(output_file, 'w') as f:
    for example in examples:
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": "", "toolCalls": example["tool_calls"]}
            ]
        }
        f.write(json.dumps(entry) + '\n')

print(f"Generated {len(examples)} training examples in {output_file}")
print(f"System prompt length: {len(system_prompt)} characters")
