#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embeddings Visualization with PCA and 3D Plotting

This script demonstrates how to generate word/sentence embeddings, reduce their dimensionality
using PCA, and visualize them in 3D space using Plotly. It provides reusable helper functions
for embedding generation, dimensionality reduction, and interactive visualization.

Before running, make sure to:
1. Activate the project venv at the project root
2. Install the required packages: pip install sentence-transformers scikit-learn plotly pandas numpy
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Try to import sentence-transformers, but provide a fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embedding generation will be simulated.")

# Sample sentences for embedding examples
sample_sentences = [
    "Embeddings are vector representations of text.",
    "Tokenization is a preprocessing step for embeddings.",
    "Large language models use embeddings to understand semantics.",
    "Word2Vec is an early word embedding technique.",
    "GloVe is another popular word embedding method.",
    "BERT embeddings are contextual, unlike Word2Vec.",
    "Sentence embeddings capture meaning at the sentence level.",
    "Embeddings can be visualized after dimensionality reduction.",
    "PCA is a common technique for reducing embedding dimensions.",
    "t-SNE is another method for visualizing high-dimensional embeddings.",
    "Python is a popular programming language for NLP.",
    "Machine learning models often use embeddings as input features.",
    "Semantic search relies on embedding similarity.",
    "Cosine similarity measures the angle between embedding vectors.",
    "Euclidean distance is another way to compare embeddings."
]


def generate_embeddings(texts: List[str], model_name: str = 'paraphrase-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate embeddings for a list of texts using a sentence transformer model.
    
    Args:
        texts (List[str]): List of texts to generate embeddings for
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        # Provide simulated embeddings if sentence-transformers is not available
        print(f"Simulating embeddings for {len(texts)} texts with a 384-dimensional model")
        np.random.seed(42)  # For reproducibility
        return np.random.randn(len(texts), 384)
    
    print(f"Generating embeddings using model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 3, method: str = 'pca') -> np.ndarray:
    """
    Reduce the dimensionality of embeddings for visualization.
    
    Args:
        embeddings (np.ndarray): Embedding vectors to reduce
        n_components (int): Number of dimensions to reduce to
        method (str): Dimensionality reduction method ('pca' currently supported)
        
    Returns:
        np.ndarray: Reduced embeddings with shape (len(embeddings), n_components)
    """
    if method.lower() == 'pca':
        print(f"Reducing {embeddings.shape[1]}-dimensional embeddings to {n_components}D using PCA")
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        explained_variance = pca.explained_variance_ratio_.sum() * 100
        print(f"Explained variance: {explained_variance:.2f}%")
        return reduced_embeddings
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")


def plot_embeddings_3d(reduced_embeddings: np.ndarray, labels: List[str], 
                     title: str = 'Embedding Visualization', 
                     color_by: Optional[List[Any]] = None,
                     hover_data: Optional[Dict[str, List[Any]]] = None) -> go.Figure:
    """
    Create an interactive 3D scatter plot of embeddings using Plotly.
    
    Args:
        reduced_embeddings (np.ndarray): Reduced embeddings with shape (n_samples, 3)
        labels (List[str]): Text labels for each point
        title (str): Title for the plot
        color_by (Optional[List[Any]]): Values to color points by (defaults to labels)
        hover_data (Optional[Dict[str, List[Any]]]): Additional data to show on hover
        
    Returns:
        go.Figure: Plotly figure object for the 3D scatter plot
    """
    if reduced_embeddings.shape[1] != 3:
        raise ValueError(f"Expected 3D embeddings, got {reduced_embeddings.shape[1]}D")
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'z': reduced_embeddings[:, 2],
        'text': labels,
        'color': color_by if color_by is not None else labels
    })
    
    # Add any additional hover data
    if hover_data is not None:
        for key, values in hover_data.items():
            df[key] = values
    
    # Create the 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='color',
        text='text',
        hover_name='text',
        title=title,
        opacity=0.8,
        height=800
    )
    
    # Improve the layout
    fig.update_layout(
        scene={
            'xaxis_title': 'Component 1',
            'yaxis_title': 'Component 2',
            'zaxis_title': 'Component 3'
        },
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


def calculate_similarity_matrix(embeddings: np.ndarray, method: str = 'cosine') -> np.ndarray:
    """
    Calculate a similarity matrix between all pairs of embeddings.
    
    Args:
        embeddings (np.ndarray): Embedding vectors
        method (str): Similarity method ('cosine' or 'euclidean')
        
    Returns:
        np.ndarray: Similarity matrix with shape (len(embeddings), len(embeddings))
    """
    if method.lower() == 'cosine':
        # Normalize the embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        return similarity_matrix
    
    elif method.lower() == 'euclidean':
        # Calculate pairwise euclidean distances
        n_samples = embeddings.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # Calculate Euclidean distance
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                # Convert distance to similarity (closer = more similar)
                similarity_matrix[i, j] = 1 / (1 + dist)
        
        return similarity_matrix
    
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def plot_similarity_heatmap(similarity_matrix: np.ndarray, labels: List[str], 
                           title: str = 'Embedding Similarity Matrix') -> go.Figure:
    """
    Create an interactive heatmap of the similarity matrix using Plotly.
    
    Args:
        similarity_matrix (np.ndarray): Similarity matrix
        labels (List[str]): Text labels for each embedding
        title (str): Title for the plot
        
    Returns:
        go.Figure: Plotly figure object for the heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        title=title,
        height=800,
        width=800,
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=0)
    )
    
    return fig


def demonstrate_word_embeddings():
    """
    Demonstrate word embeddings visualization with a set of related terms.
    """
    # Sample words related to NLP and ML
    words = [
        "embedding", "vector", "token", "language", "model", "neural", "network",
        "transformer", "attention", "bert", "gpt", "nlp", "semantic", "syntax",
        "word", "sentence", "document", "classification", "clustering", "similarity"
    ]
    
    print(f"\n--- Word Embeddings Demonstration ---\n")
    print(f"Generating embeddings for {len(words)} words")
    
    # Generate embeddings
    embeddings = generate_embeddings(words)
    
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings)
    
    # Create categories for coloring
    categories = [
        "representation", "representation", "processing", "concept", "concept", "architecture", "architecture",
        "architecture", "mechanism", "model", "model", "field", "property", "property",
        "unit", "unit", "unit", "task", "task", "measure"
    ]
    
    # Create 3D visualization
    fig = plot_embeddings_3d(
        reduced_embeddings, 
        words, 
        title="Word Embeddings in 3D Space",
        color_by=categories,
        hover_data={"category": categories}
    )
    
    # Save the figure
    fig.write_html("word_embeddings_3d.html")
    print("Saved 3D visualization to 'word_embeddings_3d.html'")
    
    # Calculate and visualize similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)
    sim_fig = plot_similarity_heatmap(similarity_matrix, words, "Word Embedding Similarities")
    sim_fig.write_html("word_embedding_similarities.html")
    print("Saved similarity heatmap to 'word_embedding_similarities.html'")


def demonstrate_sentence_embeddings():
    """
    Demonstrate sentence embeddings visualization with sample sentences.
    """
    print(f"\n--- Sentence Embeddings Demonstration ---\n")
    print(f"Generating embeddings for {len(sample_sentences)} sentences")
    
    # Generate embeddings
    embeddings = generate_embeddings(sample_sentences)
    
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings)
    
    # Create categories for coloring
    categories = [
        "concept", "process", "application", "technique", "technique",
        "comparison", "concept", "visualization", "technique", "technique",
        "tool", "application", "application", "measure", "measure"
    ]
    
    # Create shortened labels for better visualization
    short_labels = [s[:30] + "..." if len(s) > 30 else s for s in sample_sentences]
    
    # Create 3D visualization
    fig = plot_embeddings_3d(
        reduced_embeddings, 
        short_labels, 
        title="Sentence Embeddings in 3D Space",
        color_by=categories,
        hover_data={"category": categories, "full_text": sample_sentences}
    )
    
    # Save the figure
    fig.write_html("sentence_embeddings_3d.html")
    print("Saved 3D visualization to 'sentence_embeddings_3d.html'")
    
    # Calculate and visualize similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)
    sim_fig = plot_similarity_heatmap(similarity_matrix, short_labels, "Sentence Embedding Similarities")
    sim_fig.write_html("sentence_embedding_similarities.html")
    print("Saved similarity heatmap to 'sentence_embedding_similarities.html'")


def main():
    print("\n=== Embeddings Visualization with PCA and 3D Plotting ===\n")
    
    # Example 1: Word embeddings
    demonstrate_word_embeddings()
    
    # Example 2: Sentence embeddings
    demonstrate_sentence_embeddings()
    
    print("\nAll visualizations have been saved as HTML files.")
    print("You can open these files in a web browser to interact with the 3D plots and heatmaps.")


if __name__ == "__main__":
    main()
