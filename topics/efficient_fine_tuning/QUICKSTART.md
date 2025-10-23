# Continue Fine-Tuning Quick Start

## TL;DR - Complete Workflow

```bash
# 1. Install dependencies
pip install mlx-lm
brew install ollama  # or appropriate for your OS

# 2. Verify dataset
head -n 1 ../../datasets/continue_finetuning_dataset.jsonl | jq .

# 3. Fine-tune with MLX (choose base model)
python -m mlx_lm.lora \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train \
  --data ../../datasets/continue_finetuning_dataset.jsonl \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-layers 16 \
  --output-dir ./finetuned_model

# 4. Convert to GGUF
python -m mlx_lm.convert \
  --model ./finetuned_model \
  --output ./continue-assistant-finetuned.gguf

# 5. Create Ollama model
ollama create continue-assistant -f Modelfile.continue-assistant

# 6. Test
ollama run continue-assistant "Read the file main.py"

# 7. Configure Continue (~/.continue/config.json)
{
  "models": [{
    "title": "Continue Assistant",
    "provider": "ollama",
    "model": "continue-assistant"
  }]
}
```

## Dataset Stats

- **Total Examples:** 205
- **Single-Step:** 201
- **Multi-Step:** 4
- **Tools Covered:** 12 built-in Continue tools
- **Languages:** 14+ programming languages

## Recommended Base Models

| Model | Size | Tool Support | Speed | Quality |
|-------|------|--------------|-------|---------|
| llama3.1:8b | 8B | ✅ Excellent | Fast | Good |
| llama3.1:70b | 70B | ✅ Excellent | Slow | Excellent |
| mistral:7b | 7B | ✅ Good | Fast | Good |
| qwen2.5:7b | 7B | ✅ Good | Fast | Good |
| codellama:13b | 13B | ⚠️ Limited | Medium | Good |

**Recommendation:** Start with `llama3.1:8b` for best balance.

## Key Parameters

### Training
```bash
--iters 1000              # More for better quality
--batch-size 4            # Adjust based on RAM
--learning-rate 1e-5      # Conservative for stability
--lora-layers 16          # More layers = better learning
```

### Inference (Modelfile)
```dockerfile
PARAMETER num_ctx 8192        # Large context for code
PARAMETER temperature 0       # CRITICAL for tool calls
PARAMETER top_p 0.9          # Focused sampling
```

## Testing Checklist

- [ ] Model outputs valid JSON tool calls
- [ ] Tool names match Continue's built-in tools
- [ ] Arguments are properly formatted JSON strings
- [ ] Model handles multi-step requests
- [ ] Responses are concise and helpful
- [ ] No hallucinated tool names
- [ ] Temperature is set to 0

## Common Issues

### "Model outputs text instead of tool calls"
→ Check temperature is 0, verify training data format

### "Invalid JSON in tool calls"
→ Lower temperature, add more training examples

### "Model is slow"
→ Use smaller model or quantized version

### "Tool calls correct but responses poor"
→ Add more diverse examples, train longer

## File Structure

```
ai_research/
├── datasets/
│   └── continue_finetuning_dataset.jsonl  # 205 examples
└── topics/efficient_fine_tuning/
    ├── Modelfile.continue-assistant        # Ollama config
    ├── README_CONTINUE_FINETUNING.md      # Full guide
    ├── QUICKSTART.md                       # This file
    └── ollama_mlx_fine_tuning             # Original notes
```

## Next Steps

1. Run the fine-tuning command
2. Create the Ollama model
3. Test with simple prompts
4. Configure Continue extension
5. Test in VSCode
6. Iterate based on results

## Support

- MLX Issues: https://github.com/ml-explore/mlx/issues
- Ollama Issues: https://github.com/ollama/ollama/issues
- Continue Issues: https://github.com/continuedev/continue/issues
