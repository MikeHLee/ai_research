#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Tokenization Methods for NLP and LLMs

This script demonstrates various basic tokenization methods used in natural language processing
and large language models, including character-level, word-level, subword, and BPE tokenization.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install nltk transformers
"""

import re
import nltk
from typing import List, Dict, Any, Tuple
from collections import Counter

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

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


def character_tokenization(text: str) -> List[str]:
    """
    Tokenize text into individual characters.
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        List[str]: List of character tokens
    """
    return list(text)


def word_tokenization(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into words using NLTK's word_tokenize.
    
    Args:
        text (str): The input text to tokenize
        lowercase (bool): Whether to convert tokens to lowercase
        
    Returns:
        List[str]: List of word tokens
    """
    tokens = nltk.word_tokenize(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def regex_word_tokenization(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenize text into words using regex pattern matching.
    
    Args:
        text (str): The input text to tokenize
        lowercase (bool): Whether to convert tokens to lowercase
        
    Returns:
        List[str]: List of word tokens
    """
    # Pattern to match words, including contractions and hyphenated words
    pattern = r"[\w]+-[\w]+|[\w]+\'[\w]+|[\w]+|[^\w\s]"
    tokens = re.findall(pattern, text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def whitespace_tokenization(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    
    Args:
        text (str): The input text to tokenize
        lowercase (bool): Whether to convert tokens to lowercase
        
    Returns:
        List[str]: List of tokens
    """
    tokens = text.split()
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def sentence_tokenization(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK's sent_tokenize.
    
    Args:
        text (str): The input text to tokenize
        
    Returns:
        List[str]: List of sentence tokens
    """
    return nltk.sent_tokenize(text)


def simple_bpe_tokenization(text: str, vocab_size: int = 100) -> Tuple[List[str], Dict[str, int]]:
    """
    A simplified implementation of Byte-Pair Encoding tokenization.
    
    Args:
        text (str): The input text to tokenize
        vocab_size (int): Maximum vocabulary size
        
    Returns:
        Tuple[List[str], Dict[str, int]]: Tokenized text and vocabulary
    """
    # Start with character-level tokens
    words = whitespace_tokenization(text, lowercase=True)
    
    # Add space prefix to words (except the first one) to handle word boundaries
    words = [words[0]] + [' ' + word for word in words[1:]]
    
    # Initialize each word as a sequence of characters
    splits = [[c for c in word] for word in words]
    vocab = set(c for word in splits for c in word)
    
    # Merge most frequent pairs until vocab_size is reached
    while len(vocab) < vocab_size:
        # Count pair frequencies across all words
        pairs = Counter()
        for word in splits:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        
        # If no more pairs to merge, break
        if not pairs:
            break
        
        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        # Create a new token by merging the pair
        new_token = ''.join(best_pair)
        vocab.add(new_token)
        
        # Replace all occurrences of the pair with the new token
        for i, word in enumerate(splits):
            j = 0
            while j < len(word) - 1:
                if word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                    word[j:j + 2] = [new_token]
                else:
                    j += 1
    
    # Create vocabulary with token IDs
    vocab_dict = {token: i for i, token in enumerate(sorted(vocab))}
    
    # Flatten the tokens back into a single list
    tokens = [token for word in splits for token in word]
    
    return tokens, vocab_dict


def print_tokens(tokens: List[str], method_name: str) -> None:
    """
    Print tokens with their indices.
    
    Args:
        tokens (List[str]): List of tokens
        method_name (str): Name of the tokenization method
    """
    print(f"\n--- {method_name} ---")
    print(f"Total tokens: {len(tokens)}\n")
    
    # Print tokens in a readable format
    for i, token in enumerate(tokens[:50]):
        # Replace whitespace with visible characters for display
        display_token = token.replace(' ', '␣').replace('\n', '\\n').replace('\t', '\\t')
        print(f"{i}: '{display_token}'")
    
    if len(tokens) > 50:
        print(f"... and {len(tokens) - 50} more tokens")


def main():
    print("\n=== Basic Tokenization Methods for NLP and LLMs ===\n")
    print("Sample text:\n", sample_text)
    
    # Example 1: Character tokenization
    char_tokens = character_tokenization(sample_text)
    print_tokens(char_tokens[:100], "Character tokenization (first 100 tokens)")
    
    # Example 2: Word tokenization with NLTK
    word_tokens = word_tokenization(sample_text)
    print_tokens(word_tokens, "Word tokenization (NLTK)")
    
    # Example 3: Regex word tokenization
    regex_tokens = regex_word_tokenization(sample_text)
    print_tokens(regex_tokens, "Regex word tokenization")
    
    # Example 4: Whitespace tokenization
    whitespace_tokens = whitespace_tokenization(sample_text)
    print_tokens(whitespace_tokens, "Whitespace tokenization")
    
    # Example 5: Sentence tokenization
    sentence_tokens = sentence_tokenization(sample_text)
    print_tokens(sentence_tokens, "Sentence tokenization")
    
    # Example 6: Simple BPE tokenization
    bpe_tokens, bpe_vocab = simple_bpe_tokenization(sample_text, vocab_size=150)
    print_tokens(bpe_tokens, "Simple BPE tokenization")
    print(f"\nBPE Vocabulary size: {len(bpe_vocab)}")
    print("Sample vocabulary items:")
    for i, (token, idx) in enumerate(list(bpe_vocab.items())[:20]):
        display_token = token.replace(' ', '␣').replace('\n', '\\n').replace('\t', '\\t')
        print(f"{idx}: '{display_token}'")
    
    if len(bpe_vocab) > 20:
        print(f"... and {len(bpe_vocab) - 20} more vocabulary items")


if __name__ == "__main__":
    main()
