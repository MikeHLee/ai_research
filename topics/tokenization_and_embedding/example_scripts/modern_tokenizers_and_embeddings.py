#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modern Tokenizers and Embeddings for LLMs

This script demonstrates how to use modern tokenizers from popular libraries like
HuggingFace Transformers and their application with language models. It also shows
how tokenization relates to embeddings and model input preparation.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install transformers torch sentence-transformers
"""

import torch
from typing import List, Dict, Any, Tuple

# Try to import required libraries, but provide fallbacks if not available
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Some examples will be simulated.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Some examples will be simulated.")

# Sample text for demonstration
sample_text = """
Tokenization is a fundamental preprocessing step in natural language processing (NLP). 
It involves breaking down text into smaller units called tokens. These tokens can be 
characters, words, subwords, or even sentences, depending on the specific tokenization 
approach used. Effective tokenization is crucial for many NLP tasks, including text 
classification, machine translation, and sentiment analysis.

Large Language Models (LLMs) like GPT-4 and BERT use sophisticated tokenization methods 
such as Byte-Pair Encoding (BPE) or WordPiece. These methods strike a balance between 
character-level and word-level tokenization, allowing the models to handle rare words 
and out-of-vocabulary terms more effectively.
"""

# Sample sentences for embedding examples
sample_sentences = [
    "Tokenization is a fundamental preprocessing step in NLP.",
    "Large language models use sophisticated tokenization methods.",
    "Effective tokenization is crucial for many NLP tasks.",
    "Python is a popular programming language for machine learning."
]


def demonstrate_bert_tokenizer():
    """
    Demonstrate the BERT tokenizer from Hugging Face Transformers.
    
    Returns:
        Dict: Tokenization results or a simulated example
    """
    if not TRANSFORMERS_AVAILABLE:
        # Provide a simulated example if transformers is not available
        return {
            "tokens": ["[CLS]", "token", "##ization", "is", "a", "fundamental", "preprocessing", "step", "in", "nlp", ".", "[SEP]"],
            "input_ids": [101, 19204, 6254, 2003, 1037, 4125, 16057, 3458, 1999, 2361, 1012, 102],
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    
    # Load the BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize the first sentence of the sample text
    first_sentence = sample_sentences[0]
    encoding = tokenizer(first_sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    return {
        "tokens": tokens,
        "input_ids": encoding["input_ids"][0].tolist(),
        "token_type_ids": encoding["token_type_ids"][0].tolist() if "token_type_ids" in encoding else None,
        "attention_mask": encoding["attention_mask"][0].tolist()
    }


def demonstrate_gpt2_tokenizer():
    """
    Demonstrate the GPT-2 tokenizer from Hugging Face Transformers.
    
    Returns:
        Dict: Tokenization results or a simulated example
    """
    if not TRANSFORMERS_AVAILABLE:
        # Provide a simulated example if transformers is not available
        return {
            "tokens": ["Token", "ization", " is", " a", " fundamental", " pre", "processing", " step", " in", " NLP", "."],
            "input_ids": [13745, 11, 318, 257, 4686, 1348, 16155, 1314, 287, 17953, 13],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    
    # Load the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Tokenize the first sentence of the sample text
    first_sentence = sample_sentences[0]
    encoding = tokenizer(first_sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    return {
        "tokens": tokens,
        "input_ids": encoding["input_ids"][0].tolist(),
        "attention_mask": encoding["attention_mask"][0].tolist()
    }


def compare_tokenizers_on_special_cases():
    """
    Compare how different tokenizers handle special cases like rare words,
    numbers, and punctuation.
    
    Returns:
        Dict: Comparison results or a simulated example
    """
    if not TRANSFORMERS_AVAILABLE:
        # Provide a simulated example if transformers is not available
        return {
            "special_text": "COVID-19 affects 7.8 billion people worldwide! 😷",
            "bert": ["[CLS]", "covid", "-", "19", "affects", "7", ".", "8", "billion", "people", "worldwide", "!", "[UNK]", "[SEP]"],
            "gpt2": ["COVID", "-", "19", " affects", " 7", ".", "8", " billion", " people", " worldwide", "!", " 😷"],
            "roberta": ["<s>", "COVID", "-", "19", "Ġaffects", "Ġ7", ".", "8", "Ġbillion", "Ġpeople", "Ġworldwide", "!", "Ġ😷", "</s>"]
        }
    
    # Special text with rare words, numbers, and emoji
    special_text = "COVID-19 affects 7.8 billion people worldwide! 😷"
    
    # Load tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Tokenize with each tokenizer
    bert_tokens = bert_tokenizer.tokenize(bert_tokenizer.cls_token + special_text + bert_tokenizer.sep_token)
    gpt2_tokens = gpt2_tokenizer.tokenize(special_text)
    roberta_tokens = roberta_tokenizer.tokenize(roberta_tokenizer.bos_token + special_text + roberta_tokenizer.eos_token)
    
    return {
        "special_text": special_text,
        "bert": bert_tokens,
        "gpt2": gpt2_tokens,
        "roberta": roberta_tokens
    }


def demonstrate_sentence_embeddings():
    """
    Demonstrate how tokenization relates to sentence embeddings.
    
    Returns:
        Dict: Embedding results or a simulated example
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Provide a simulated example if sentence-transformers is not available
        import numpy as np
        np.random.seed(42)
        
        # Simulate embeddings with random vectors
        simulated_embeddings = [np.random.randn(384) for _ in range(len(sample_sentences))]
        simulated_similarities = [[np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)) 
                                  for e2 in simulated_embeddings] for e1 in simulated_embeddings]
        
        return {
            "sentences": sample_sentences,
            "embedding_dim": 384,
            "similarity_matrix": simulated_similarities
        }
    
    # Load a sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Generate embeddings for sample sentences
    embeddings = model.encode(sample_sentences)
    
    # Calculate cosine similarities between all pairs of sentences
    similarity_matrix = [[torch.nn.functional.cosine_similarity(
                          torch.tensor(e1).unsqueeze(0), 
                          torch.tensor(e2).unsqueeze(0)).item() 
                         for e2 in embeddings] for e1 in embeddings]
    
    return {
        "sentences": sample_sentences,
        "embedding_dim": embeddings[0].shape[0],
        "similarity_matrix": similarity_matrix
    }


def demonstrate_tokenization_for_model_input():
    """
    Demonstrate how tokenization is used to prepare inputs for language models.
    
    Returns:
        Dict: Model input preparation results or a simulated example
    """
    if not TRANSFORMERS_AVAILABLE:
        # Provide a simulated example if transformers is not available
        return {
            "model_name": "bert-base-uncased (simulated)",
            "input_text": sample_sentences[0],
            "tokenizer_output": {
                "input_ids": [[101, 19204, 6254, 2003, 1037, 4125, 16057, 3458, 1999, 2361, 1012, 102]],
                "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            },
            "model_output_shape": [1, 12, 768]
        }
    
    # Load a pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare input text
    input_text = sample_sentences[0]
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    # Get model outputs (without gradient calculation)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert tensors to lists for display
    inputs_dict = {k: v.tolist() for k, v in inputs.items()}
    
    return {
        "model_name": model_name,
        "input_text": input_text,
        "tokenizer_output": inputs_dict,
        "model_output_shape": list(outputs.last_hidden_state.shape)
    }


def print_tokenizer_results(results: Dict[str, Any], title: str) -> None:
    """
    Print tokenizer results in a readable format.
    
    Args:
        results (Dict[str, Any]): Tokenization results
        title (str): Title for the results section
    """
    print(f"\n--- {title} ---\n")
    
    if "tokens" in results:
        print("Tokens:")
        for i, token in enumerate(results["tokens"]):
            print(f"  {i}: '{token}'")
    
    if "input_ids" in results:
        print("\nInput IDs:")
        print(f"  {results['input_ids']}")
    
    if "token_type_ids" in results and results["token_type_ids"] is not None:
        print("\nToken Type IDs:")
        print(f"  {results['token_type_ids']}")
    
    if "attention_mask" in results:
        print("\nAttention Mask:")
        print(f"  {results['attention_mask']}")


def print_comparison_results(results: Dict[str, Any]) -> None:
    """
    Print tokenizer comparison results.
    
    Args:
        results (Dict[str, Any]): Comparison results
    """
    print(f"\n--- Tokenizer Comparison on Special Cases ---\n")
    print(f"Text: {results['special_text']}\n")
    
    print("BERT tokenization:")
    print(f"  {results['bert']}\n")
    
    print("GPT-2 tokenization:")
    print(f"  {results['gpt2']}\n")
    
    print("RoBERTa tokenization:")
    print(f"  {results['roberta']}")


def print_embedding_results(results: Dict[str, Any]) -> None:
    """
    Print sentence embedding results.
    
    Args:
        results (Dict[str, Any]): Embedding results
    """
    print(f"\n--- Sentence Embeddings ---\n")
    print(f"Embedding dimension: {results['embedding_dim']}\n")
    
    print("Sentences:")
    for i, sentence in enumerate(results['sentences']):
        print(f"  {i}: {sentence}")
    
    print("\nSimilarity Matrix:")
    for i, row in enumerate(results['similarity_matrix']):
        formatted_row = [f"{sim:.2f}" for sim in row]
        print(f"  {i}: {formatted_row}")


def print_model_input_results(results: Dict[str, Any]) -> None:
    """
    Print model input preparation results.
    
    Args:
        results (Dict[str, Any]): Model input results
    """
    print(f"\n--- Model Input Preparation with {results['model_name']} ---\n")
    print(f"Input text: {results['input_text']}\n")
    
    print("Tokenizer output:")
    for key, value in results['tokenizer_output'].items():
        print(f"  {key}: {value}")
    
    print(f"\nModel output shape: {results['model_output_shape']}")
    print(f"  [batch_size, sequence_length, hidden_size]")


def main():
    print("\n=== Modern Tokenizers and Embeddings for LLMs ===\n")
    
    # Example 1: BERT tokenizer
    bert_results = demonstrate_bert_tokenizer()
    print_tokenizer_results(bert_results, "BERT Tokenizer Example")
    
    # Example 2: GPT-2 tokenizer
    gpt2_results = demonstrate_gpt2_tokenizer()
    print_tokenizer_results(gpt2_results, "GPT-2 Tokenizer Example")
    
    # Example 3: Compare tokenizers on special cases
    comparison_results = compare_tokenizers_on_special_cases()
    print_comparison_results(comparison_results)
    
    # Example 4: Sentence embeddings
    embedding_results = demonstrate_sentence_embeddings()
    print_embedding_results(embedding_results)
    
    # Example 5: Tokenization for model input
    model_input_results = demonstrate_tokenization_for_model_input()
    print_model_input_results(model_input_results)


if __name__ == "__main__":
    main()
