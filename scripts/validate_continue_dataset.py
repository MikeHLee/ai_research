#!/usr/bin/env python3
"""
Validate Continue fine-tuning dataset for quality and correctness.

Usage:
    python scripts/validate_continue_dataset.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Continue built-in tools
VALID_TOOLS = {
    "read_file",
    "read_file_range",
    "read_currently_open_file",
    "create_new_file",
    "grep_search",
    "file_glob_search",
    "ls",
    "run_terminal_command",
    "view_diff",
    "search_web",
    "fetch_url_content",
    "view_repo_map",
}

def validate_dataset(dataset_path: str):
    """Validate the Continue fine-tuning dataset."""
    
    print(f"🔍 Validating dataset: {dataset_path}\n")
    
    errors = []
    warnings = []
    stats = {
        "total_examples": 0,
        "single_step": 0,
        "multi_step": 0,
        "tool_usage": Counter(),
        "invalid_json": 0,
        "invalid_tools": 0,
        "missing_fields": 0,
    }
    
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats["total_examples"] += 1
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                stats["invalid_json"] += 1
                continue
            
            # Validate structure
            if "messages" not in entry:
                errors.append(f"Line {line_num}: Missing 'messages' field")
                stats["missing_fields"] += 1
                continue
            
            messages = entry["messages"]
            
            # Check for system message
            if not messages or messages[0].get("role") != "system":
                warnings.append(f"Line {line_num}: Missing or incorrect system message")
            
            # Count tool calls
            tool_calls_count = 0
            for msg in messages:
                if msg.get("role") == "assistant" and "toolCalls" in msg:
                    tool_calls_count += 1
                    
                    # Validate tool calls
                    for tool_call in msg["toolCalls"]:
                        # Check structure
                        if "function" not in tool_call:
                            errors.append(f"Line {line_num}: Tool call missing 'function' field")
                            continue
                        
                        func = tool_call["function"]
                        tool_name = func.get("name")
                        
                        # Validate tool name
                        if tool_name not in VALID_TOOLS:
                            errors.append(f"Line {line_num}: Invalid tool '{tool_name}'")
                            stats["invalid_tools"] += 1
                        else:
                            stats["tool_usage"][tool_name] += 1
                        
                        # Validate arguments are valid JSON
                        try:
                            args = func.get("arguments", "{}")
                            json.loads(args)
                        except json.JSONDecodeError:
                            errors.append(f"Line {line_num}: Invalid JSON in tool arguments")
            
            # Categorize example
            if tool_calls_count == 1:
                stats["single_step"] += 1
            elif tool_calls_count > 1:
                stats["multi_step"] += 1
    
    # Print results
    print("=" * 60)
    print("📊 DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Examples:     {stats['total_examples']}")
    print(f"Single-Step:        {stats['single_step']}")
    print(f"Multi-Step:         {stats['multi_step']}")
    print(f"Invalid JSON:       {stats['invalid_json']}")
    print(f"Invalid Tools:      {stats['invalid_tools']}")
    print(f"Missing Fields:     {stats['missing_fields']}")
    print()
    
    print("=" * 60)
    print("🔧 TOOL USAGE DISTRIBUTION")
    print("=" * 60)
    for tool, count in stats["tool_usage"].most_common():
        bar = "█" * (count // 2)
        print(f"{tool:30s} {count:3d} {bar}")
    print()
    
    # Print errors
    if errors:
        print("=" * 60)
        print(f"❌ ERRORS ({len(errors)})")
        print("=" * 60)
        for error in errors[:10]:  # Show first 10
            print(f"  • {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
    
    # Print warnings
    if warnings:
        print("=" * 60)
        print(f"⚠️  WARNINGS ({len(warnings)})")
        print("=" * 60)
        for warning in warnings[:10]:  # Show first 10
            print(f"  • {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
        print()
    
    # Final verdict
    print("=" * 60)
    if errors:
        print("❌ VALIDATION FAILED")
        print(f"   Found {len(errors)} errors that need to be fixed")
        return False
    elif warnings:
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
        print(f"   Found {len(warnings)} warnings (non-critical)")
        return True
    else:
        print("✅ VALIDATION PASSED")
        print("   Dataset is ready for fine-tuning!")
        return True
    print("=" * 60)

def main():
    dataset_path = Path(__file__).parent.parent / "datasets" / "continue_finetuning_dataset.jsonl"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    success = validate_dataset(str(dataset_path))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
