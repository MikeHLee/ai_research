# Continue VSCode Extension Fine-Tuning Guide

This guide covers fine-tuning a language model for the Continue VSCode extension using MLX and Ollama, with emphasis on tool calling capabilities.

## Overview

The Continue extension requires models that can:
1. Understand natural language coding requests
2. Make appropriate tool calls in JSON format
3. Chain multiple tool calls for complex tasks
4. Provide helpful coding assistance

## Dataset

**Location:** `/datasets/continue_finetuning_dataset.jsonl`

**Size:** 205 training examples
- 201 single-step tool interactions
- 4 multi-step tool chain examples

**Format:** JSONL with chat messages including `toolCalls` structure

### Example Entry
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant integrated into the Continue VSCode extension..."
    },
    {
      "role": "user",
      "content": "Read the package.json file"
    },
    {
      "role": "assistant",
      "content": "",
      "toolCalls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "read_file",
            "arguments": "{\"filepath\": \"package.json\"}"
          }
        }
      ]
    }
  ]
}
```

### Tools Covered

The dataset includes examples for all Continue built-in tools:

**File Operations:**
- `read_file` - Read entire file contents
- `read_file_range` - Read specific line ranges
- `read_currently_open_file` - Read active editor file
- `create_new_file` - Create files with content

**Search & Discovery:**
- `grep_search` - Regex pattern search in files
- `file_glob_search` - Find files by glob patterns
- `ls` - List directory contents
- `view_repo_map` - View repository structure

**Execution:**
- `run_terminal_command` - Execute shell commands

**Version Control:**
- `view_diff` - View git changes

**Web:**
- `search_web` - Web search
- `fetch_url_content` - Fetch URL content

### Coverage Areas

- **Languages:** JavaScript/TypeScript, Python, Go, Rust, Java, Kotlin, Swift, Ruby, PHP, C++, Scala, Elixir, Dart, Haskell
- **Frameworks:** React, Express, FastAPI, Next.js, Vite
- **Tools:** Docker, Kubernetes, Terraform, CI/CD, Git
- **Concepts:** Debugging, testing, refactoring, authentication, API design

## Fine-Tuning Process

### 1. Prerequisites

```bash
# Install MLX
pip install mlx-lm

# Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Prepare Training Data

The dataset is already prepared at `/datasets/continue_finetuning_dataset.jsonl`

Verify format:
```bash
head -n 1 datasets/continue_finetuning_dataset.jsonl | jq .
```

### 3. Fine-Tune with MLX

```bash
# Pull base model (choose one with tool support)
ollama pull llama3.1:8b

# Export model for MLX
ollama show llama3.1:8b --modelfile > base_modelfile.txt

# Fine-tune with MLX
python -m mlx_lm.lora \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train \
  --data datasets/continue_finetuning_dataset.jsonl \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-layers 16 \
  --output-dir ./finetuned_model

# Convert to GGUF
python -m mlx_lm.convert \
  --model ./finetuned_model \
  --output ./continue-assistant-finetuned.gguf
```

### 4. Create Ollama Model

Use the provided Modelfile:

```bash
cd topics/efficient_fine_tuning
ollama create continue-assistant -f Modelfile.continue-assistant
```

### 5. Test the Model

```bash
# Basic test
ollama run continue-assistant "Read the file main.py"

# Tool calling test
ollama run continue-assistant "Find all TypeScript files"

# Multi-step test
ollama run continue-assistant "Debug why the API is failing"
```

Expected output should include `toolCalls` JSON structure:
```json
{
  "toolCalls": [{
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{\"filepath\": \"main.py\"}"
    }
  }]
}
```

## Model Configuration

### Critical Parameters (in Modelfile)

```dockerfile
PARAMETER num_ctx 8192              # Large context for code
PARAMETER temperature 0             # Deterministic tool calls
PARAMETER top_p 0.9                 # Focused sampling
PARAMETER repeat_penalty 1.1        # Prevent repetition
```

### Why Temperature = 0?

Tool calling requires **exact JSON format**. Temperature 0 ensures:
- Consistent tool call structure
- Valid JSON output
- Deterministic behavior
- Reliable function arguments

## Integration with Continue

### 1. Configure Continue

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Continue Assistant",
      "provider": "ollama",
      "model": "continue-assistant",
      "apiBase": "http://localhost:11434"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Codestral",
    "provider": "ollama",
    "model": "codestral"
  }
}
```

### 2. Restart VSCode

The model should now appear in Continue's model selector.

### 3. Test in Continue

Try these prompts:
- "Read the package.json file"
- "Find all React components"
- "Show me the git diff"
- "Run the tests"
- "Search for TODO comments"

## Troubleshooting

### Model doesn't output tool calls

**Issue:** Model responds with text instead of tool calls

**Solutions:**
1. Verify training data format is correct
2. Ensure base model supports tool calling (llama3.1, mistral, qwen2.5)
3. Check temperature is set to 0
4. Increase training iterations
5. Verify system prompt in Modelfile

### Invalid JSON in tool calls

**Issue:** Tool calls have malformed JSON

**Solutions:**
1. Lower temperature (should be 0)
2. Add more training examples with correct format
3. Use a base model with better instruction following
4. Increase LoRA layers for better learning

### Model is too slow

**Issue:** Responses take too long

**Solutions:**
1. Use quantized model (Q4_K_M or Q5_K_M)
2. Reduce context window if not needed
3. Use smaller base model (8B instead of 70B)
4. Enable GPU acceleration in Ollama

### Tool calls work but responses are poor

**Issue:** Correct tool calls but unhelpful explanations

**Solutions:**
1. Add more diverse training examples
2. Include examples with explanatory text
3. Fine-tune for more iterations
4. Use a larger base model

## Advanced: Custom Tools

To add custom tools to the dataset:

```python
import json

custom_example = {
    "messages": [
        {
            "role": "system",
            "content": "You are an AI coding assistant..."
        },
        {
            "role": "user",
            "content": "Use my custom tool"
        },
        {
            "role": "assistant",
            "content": "",
            "toolCalls": [{
                "id": "call_custom",
                "type": "function",
                "function": {
                    "name": "my_custom_tool",
                    "arguments": json.dumps({"param": "value"})
                }
            }]
        }
    ]
}

with open('datasets/continue_finetuning_dataset.jsonl', 'a') as f:
    f.write(json.dumps(custom_example) + '\n')
```

## Performance Benchmarks

Expected performance after fine-tuning:

| Metric | Target | Notes |
|--------|--------|-------|
| Tool Call Accuracy | >95% | Correct tool selection |
| JSON Validity | >99% | Valid JSON format |
| Argument Accuracy | >90% | Correct tool arguments |
| Response Time | <2s | On M1/M2 Mac with 8B model |
| Context Understanding | >85% | Understands coding context |

## Best Practices

### Training Data
- ✅ Include diverse tool combinations
- ✅ Cover common coding scenarios
- ✅ Add multi-step examples
- ✅ Include error handling cases
- ❌ Don't duplicate similar examples
- ❌ Don't use inconsistent formats

### Model Selection
- ✅ Use models with native tool support
- ✅ Start with 8B models for speed
- ✅ Consider 70B for complex reasoning
- ❌ Don't use models without instruction tuning
- ❌ Don't use models <7B for tool calling

### Deployment
- ✅ Set temperature to 0 for tools
- ✅ Use large context window (8K+)
- ✅ Test thoroughly before production
- ✅ Monitor tool call success rate
- ❌ Don't skip validation testing
- ❌ Don't use default model parameters

## Resources

- [MLX Documentation](https://github.com/ml-explore/mlx)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Continue Extension](https://github.com/continuedev/continue)
- [Fine-Tuning Guide](https://dzone.com/articles/fine-tuning-llms-locally-using-mlx-lm-guide)

## Next Steps

1. **Expand Dataset:** Add more domain-specific examples
2. **Evaluate:** Test on held-out examples
3. **Iterate:** Refine based on performance
4. **Deploy:** Integrate with Continue extension
5. **Monitor:** Track tool call success rates
6. **Improve:** Add examples for failure cases

## License

Dataset and Modelfile: Apache 2.0
