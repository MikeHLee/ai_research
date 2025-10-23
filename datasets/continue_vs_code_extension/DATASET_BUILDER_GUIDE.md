# Continue Editor Assistant Fine-tuning Dataset Builder Guide

## Purpose
This guide provides a systematic process for an LLM to iteratively build and enhance the `continue_finetuning_dataset.jsonl` file. The goal is to create comprehensive training examples that teach open-source models (like Gemma) to effectively use the Continue VSCode extension's tool-calling capabilities.

## Dataset Overview

**Current State:**
- File: `continue_finetuning_dataset.jsonl`
- Format: JSONL (one JSON object per line)
- Total examples: 230+ (as of last update)
- Target model: Open-source LLMs (Gemma, Llama, etc.)

**Key Insight from Source Code Analysis:**
The Continue editor uses Claude-optimized tool calling patterns. Our dataset must teach open-source models to:
1. Make parallel tool calls when gathering independent information
2. Chain tool calls sequentially for dependent operations
3. Use proper JSON formatting in function arguments
4. Handle multi-step debugging and refactoring workflows
5. Understand when to use each of the available tools

## Available Tools (from source code)

**IMPORTANT: Before generating examples, examine the Continue source code to discover:**
1. **Planning and agentic commands** - Look for tools like `write_checklist`, `update_plan`, task management features
2. **Advanced workflow patterns** - Multi-step reasoning, checkpoint creation, progress tracking
3. **Tool combinations** - How tools are chained together in the actual implementation
4. **Error handling patterns** - Retry logic, validation, fallback strategies
5. **Context management** - How the assistant maintains state across conversations

**Where to look in the source code:**
- Tool definitions and implementations
- System prompts and instructions
- Agentic workflow examples
- Planning and task decomposition logic
- Multi-turn conversation patterns

### Core Built-in Tools:
1. **read_file** - Read entire file contents
2. **read_file_range** - Read specific line ranges (for large files)
3. **edit** - Make targeted edits to files
4. **multi_edit** - Make multiple edits to a single file (for capable models)
5. **write_file** - Create new files
6. **list_files** / **ls** - List directory contents
7. **search_code** / **grep_search** - Search for patterns in code
8. **file_glob_search** - Find files matching patterns
9. **run_terminal_command** - Execute shell commands
10. **fetch** / **fetch_url_content** - Fetch web content
11. **view_diff** - View git diff
12. **read_currently_open_file** - Read the active file in IDE
13. **write_checklist** - Create task checklists (for planning)

### Planning & Agentic Tools (check source code for latest):
- **update_plan** - Update task plans with steps and status
- **create_checkpoint** - Save progress for long-running tasks
- **write_notes** - Maintain context across sessions
- **status** - Report current status and progress

### Dynamic Tools (context-dependent):
- **exit** - Exit in headless mode
- **status** - Report status (beta feature)

### Tool Capability Filtering:
- Models deemed "capable" get `multi_edit` instead of `edit`
- Capability determined by provider/model name matching

## Dataset Structure

Each example follows this format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension. You help developers write, understand, and debug code through natural conversation and tool usage."
    },
    {
      "role": "user",
      "content": "<user request>"
    },
    {
      "role": "assistant",
      "content": "<optional explanation>",
      "toolCalls": [
        {
          "id": "<unique_id>",
          "type": "function",
          "function": {
            "name": "<tool_name>",
            "arguments": "<JSON_string_of_args>"
          }
        }
      ]
    }
  ]
}
```

## Current Dataset Patterns (Analyzed)

### Pattern Categories:

1. **Simple Single Tool Calls (60% of dataset)**
   - Direct file reading
   - Simple searches
   - Terminal commands
   - Directory listings

2. **Conceptual Explanations (20% of dataset)**
   - No tool calls
   - Pure educational content
   - Programming concepts
   - Best practices

3. **Multi-turn Conversations (15% of dataset)**
   - Sequential tool calls with user feedback
   - Debugging workflows
   - Refactoring tasks
   - Feature implementation

4. **Parallel Tool Calls (5% of dataset)**
   - Multiple independent information gathering
   - Simultaneous file reads
   - Concurrent searches

## Gap Analysis - What's Missing

Based on source code analysis and current dataset review:

### Critical Gaps:

1. **Advanced Multi-Edit Examples**
   - The `multi_edit` tool is for capable models but has minimal examples
   - Need examples showing multiple sequential edits to same file
   - Import statement additions separate from code changes

2. **Complex Workflow Patterns**
   - Limited examples of 5+ step workflows
   - Few examples of error recovery and retry logic
   - Missing examples of tool call validation failures

3. **Parallel Tool Call Patterns**
   - Only ~5% show parallel calls
   - Need more examples of gathering multiple independent pieces of info
   - Batch file reading patterns

4. **Context-Aware Tool Selection**
   - Missing examples showing when NOT to use certain tools
   - No examples of choosing between `edit` vs `multi_edit`
   - Limited guidance on `read_file` vs `read_file_range`

5. **Real-World Scenarios**
   - Few examples of actual debugging sessions
   - Limited refactoring workflows
   - Missing migration/upgrade scenarios
   - No performance optimization examples

6. **Tool Argument Complexity**
   - Most examples use simple arguments
   - Need more regex patterns in grep_search
   - Complex glob patterns underrepresented
   - Multi-line file content in write_file

## Iterative Building Process

### Phase 1: Read and Analyze
```
1. Read the entire dataset file (use offset/limit for large files)
2. Categorize existing examples by:
   - Tool usage patterns
   - Complexity level
   - Domain (frontend, backend, DevOps, etc.)
3. Identify gaps and underrepresented patterns
4. Create a priority list of examples to add
```

### Phase 2: Generate New Examples
```
For each gap identified:
1. Create 3-5 variations of the pattern
2. Ensure proper JSON formatting
3. Use realistic file paths and code snippets
4. Vary the programming languages/frameworks
5. Include both simple and complex versions
```

### Phase 3: Append to Dataset
```
1. Format new examples as valid JSONL
2. Append to the file (don't overwrite)
3. Verify JSON validity
4. Count total examples
5. Document what was added
```

### Phase 4: Validate and Iterate
```
1. Re-read portions of the dataset
2. Check for redundancy
3. Ensure diversity
4. Return to Phase 1 with refined focus
```

## Example Generation Templates

### Template 1: Parallel Tool Calls
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension. You help developers write, understand, and debug code through natural conversation and tool usage."
    },
    {
      "role": "user",
      "content": "<Request requiring multiple independent pieces of information>"
    },
    {
      "role": "assistant",
      "content": "I'll gather the necessary information from multiple sources.",
      "toolCalls": [
        {"id": "call_X1", "type": "function", "function": {"name": "tool1", "arguments": "{...}"}},
        {"id": "call_X2", "type": "function", "function": {"name": "tool2", "arguments": "{...}"}},
        {"id": "call_X3", "type": "function", "function": {"name": "tool3", "arguments": "{...}"}}
      ]
    }
  ]
}
```

### Template 2: Multi-Step Workflow
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension. You help developers write, understand, and debug code through natural conversation and tool usage."
    },
    {
      "role": "user",
      "content": "<Complex task requiring multiple steps>"
    },
    {
      "role": "assistant",
      "content": "I'll start by <step 1 explanation>.",
      "toolCalls": [{"id": "call_Y1", "type": "function", "function": {"name": "tool1", "arguments": "{...}"}}]
    },
    {
      "role": "user",
      "content": "<Tool result or feedback>"
    },
    {
      "role": "assistant",
      "content": "Now I'll <step 2 explanation>.",
      "toolCalls": [{"id": "call_Y2", "type": "function", "function": {"name": "tool2", "arguments": "{...}"}}]
    }
  ]
}
```

### Template 3: Multi-Edit Pattern
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension. You help developers write, understand, and debug code through natural conversation and tool usage."
    },
    {
      "role": "user",
      "content": "<Request requiring multiple edits to same file>"
    },
    {
      "role": "assistant",
      "content": "I'll make multiple changes to this file.",
      "toolCalls": [{
        "id": "call_Z1",
        "type": "function",
        "function": {
          "name": "multi_edit",
          "arguments": "{\"filepath\": \"...\", \"edits\": [{\"old_string\": \"...\", \"new_string\": \"...\"}, {...}]}"
        }
      }]
    }
  ]
}
```

### Template 4: Planning & Agentic Workflow
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension. You help developers write, understand, and debug code through natural conversation and tool usage."
    },
    {
      "role": "user",
      "content": "<Complex multi-step request requiring planning>"
    },
    {
      "role": "assistant",
      "content": "I'll break this down into steps and track progress.",
      "toolCalls": [{
        "id": "call_PLAN1",
        "type": "function",
        "function": {
          "name": "update_plan",
          "arguments": "{\"plan\": [{\"step\": \"Step 1 description\", \"status\": \"in_progress\"}, {\"step\": \"Step 2 description\", \"status\": \"pending\"}, {\"step\": \"Step 3 description\", \"status\": \"pending\"}], \"explanation\": \"Breaking down the task...\"}"
        }
      }]
    },
    {
      "role": "user",
      "content": "<Tool result or continuation>"
    },
    {
      "role": "assistant",
      "content": "Completing step 1 and moving to step 2.",
      "toolCalls": [
        {"id": "call_PLAN2", "type": "function", "function": {"name": "edit", "arguments": "{...}"}},
        {"id": "call_PLAN3", "type": "function", "function": {"name": "update_plan", "arguments": "{\"plan\": [{\"step\": \"Step 1\", \"status\": \"completed\"}, {\"step\": \"Step 2\", \"status\": \"in_progress\"}, {\"step\": \"Step 3\", \"status\": \"pending\"}]}"}}
      ]
    }
  ]
}
```

## Priority Areas for Next Additions

### Critical Priority (Add 30-40 examples each):
1. **Planning & Agentic Workflows** ⭐ NEW
   - Using `update_plan` to track multi-step tasks
   - Breaking down complex requests into subtasks
   - Using `write_checklist` for task management
   - Creating checkpoints for long-running work
   - Maintaining context with notes across sessions
   - Reporting progress with status updates
   - Adaptive planning (updating plans based on discoveries)

### High Priority (Add 20-30 examples each):
1. **Debugging Workflows**
   - Finding error sources
   - Analyzing stack traces
   - Testing fixes
   - Verifying solutions

2. **Refactoring Patterns**
   - Extract function/component
   - Rename across files
   - Move code between files
   - Update imports

3. **Parallel Information Gathering**
   - Project structure analysis
   - Dependency audits
   - Configuration reviews
   - Multi-file searches

4. **Complex Search Patterns**
   - Regex-based code searches
   - Multi-pattern searches
   - Negative lookups
   - Context-aware searches

### Medium Priority (Add 10-15 examples each):
1. **Migration Scenarios**
   - Framework upgrades
   - Language migrations
   - API version changes
   - Dependency updates

2. **Performance Optimization**
   - Identifying bottlenecks
   - Code profiling
   - Bundle analysis
   - Query optimization

3. **Security Audits**
   - Finding vulnerabilities
   - Checking for hardcoded secrets
   - Validating input sanitization
   - Reviewing authentication

4. **Testing Workflows**
   - Writing unit tests
   - Creating integration tests
   - E2E test setup
   - Test coverage analysis

### Low Priority (Add 5-10 examples each):
1. **Documentation Generation**
   - API docs
   - README creation
   - Code comments
   - Architecture diagrams

2. **DevOps Tasks**
   - CI/CD setup
   - Docker configuration
   - Kubernetes manifests
   - Infrastructure as code

## Quality Guidelines

### Each Example Should:
1. ✅ Use realistic file paths and names
2. ✅ Include proper error handling context when relevant
3. ✅ Show natural language that developers actually use
4. ✅ Demonstrate correct JSON escaping in arguments
5. ✅ Vary programming languages and frameworks
6. ✅ Include both simple and complex variations
7. ✅ Show proper tool selection reasoning
8. ✅ Use unique call IDs (call_XXX format)

### Avoid:
1. ❌ Overly generic or vague requests
2. ❌ Unrealistic file structures
3. ❌ Repetitive patterns without variation
4. ❌ Invalid JSON formatting
5. ❌ Tool calls that don't match the request
6. ❌ Missing required tool arguments
7. ❌ Inconsistent system messages

## Execution Loop for LLM

```
LOOP:
  0. **FIRST TIME ONLY**: Examine Continue source code for:
     - All available tools (especially planning/agentic ones)
     - Tool argument structures and examples
     - Multi-step workflow patterns
     - System prompts and behavioral guidelines
     - Update this guide with any newly discovered tools
  1. Read dataset (use offset/limit, read 50-100 lines at a time)
  2. Analyze current patterns and gaps
  3. Select 1-2 priority areas from above
  4. Generate 10-20 new examples (include planning/agentic tool usage)
  5. Write to temporary file
  6. Validate JSON format
  7. Append to main dataset
  8. Document additions
  9. GOTO LOOP (or stop after 5-10 iterations)
```

**Source Code Examination Checklist:**
- [ ] Found all tool definitions and their argument schemas
- [ ] Identified planning/agentic commands (update_plan, write_checklist, etc.)
- [ ] Reviewed example workflows from the codebase
- [ ] Noted any special tool combinations or patterns
- [ ] Checked for error handling and retry mechanisms
- [ ] Examined multi-turn conversation examples

## Validation Checklist

Before appending new examples:
- [ ] All JSON is valid (no syntax errors)
- [ ] Each line is a complete JSON object
- [ ] System message is consistent
- [ ] Tool names match available tools
- [ ] Arguments are properly JSON-stringified
- [ ] Call IDs are unique
- [ ] Examples add new patterns (not duplicates)
- [ ] File paths are realistic
- [ ] Code snippets are syntactically valid

## Success Metrics

Track these as you build:
- Total examples: Target 500+
- Tool coverage: Each tool should have 15+ examples
- **Planning/Agentic tools: 30+ examples each** ⭐ NEW
- Parallel calls: Should be 15-20% of dataset
- Multi-turn: Should be 25-30% of dataset
- **Multi-step workflows with planning: 20% of dataset** ⭐ NEW
- Language diversity: 10+ programming languages
- Framework coverage: React, Vue, Angular, Express, FastAPI, Django, etc.
- Complexity levels: 40% simple, 40% medium, 20% complex

## Notes for Open-Source Model Training

The key difference between Claude and open-source models:
- Claude has built-in understanding of tool calling patterns
- Open-source models need explicit examples of:
  - When to make parallel vs sequential calls
  - How to format complex JSON arguments
  - When to use which tool
  - How to chain operations
  - Error recovery patterns

This dataset bridges that gap by providing comprehensive examples that teach these patterns explicitly.

## Next Steps

1. **⭐ FIRST: Examine the Continue source code** to discover all tools, especially planning/agentic ones
   - Search for tool definitions and implementations
   - Look for `update_plan`, `write_checklist`, `create_checkpoint`, etc.
   - Review actual workflow examples from the codebase
   - Note argument structures and usage patterns
2. **Read the current dataset systematically** (50-100 lines at a time)
3. **Identify the top 3 gaps** from the priority list (prioritize planning/agentic workflows)
4. **Generate 15-20 examples** for each gap (include planning tool usage)
5. **Validate and append** to the dataset
6. **Repeat** until target metrics are met

---

**Last Updated:** October 2025
**Dataset Version:** 1.0
**Target Models:** Gemma 2, Llama 3, Mistral, Qwen, etc.
