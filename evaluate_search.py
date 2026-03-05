#!/usr/bin/env python3
"""
Visual Product Similarity Search Evaluation Script
===================================================

This script evaluates the performance of a visual product similarity search system
implemented using Vision Transformer (ViT) embeddings and FAISS vector search.

Features:
- FAISS-based approximate nearest neighbor search
- Brute-force cosine similarity search for comparison
- Performance benchmarking with timing statistics
- Detailed similarity score reporting

Compatible with CPU execution (no GPU required).
"""

import time
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import os
import random

# ML Libraries
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Configuration
# ============================================================================

# Model checkpoint (same as used during indexing)
MODEL_CKPT = 'google/vit-base-patch16-224'

# Dataset repository containing precomputed embeddings
DATASET_REPO = "Gauravannad/fashion-products-embeddings"

# Evaluation parameters
TOP_K = 5  # Number of similar products to retrieve
NUM_QUERIES = 10  # Number of queries for benchmarking
RANDOM_SEED = 42  # For reproducibility


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class RetrievalResult:
    """Stores the result of a single retrieval query."""
    query_index: int
    faiss_time_ms: float
    bruteforce_time_ms: float
    faiss_scores: List[float]
    bruteforce_scores: List[float]
    faiss_indices: List[int]
    bruteforce_indices: List[int]


@dataclass
class BenchmarkSummary:
    """Stores the overall benchmark summary."""
    num_queries: int
    top_k: int
    total_embeddings: int
    embedding_dim: int
    avg_faiss_time_ms: float
    std_faiss_time_ms: float
    min_faiss_time_ms: float
    max_faiss_time_ms: float
    avg_bruteforce_time_ms: float
    std_bruteforce_time_ms: float
    min_bruteforce_time_ms: float
    max_bruteforce_time_ms: float
    speedup_factor: float
    recall_at_k: float  # Percentage of brute-force results found by FAISS


# ============================================================================
# Core Functions
# ============================================================================

def load_model_and_extractor(model_ckpt: str) -> Tuple[Any, Any]:
    """
    Load the pre-trained Vision Transformer model and image processor.
    
    Args:
        model_ckpt: Hugging Face model checkpoint name
        
    Returns:
        Tuple of (image_processor, model)
    """
    print(f"Loading model: {model_ckpt}")
    extractor = AutoImageProcessor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    print(f"Model loaded successfully. Hidden dimension: {model.config.hidden_size}")
    return extractor, model


def extract_embedding(image: Image.Image, extractor: Any, model: Any) -> np.ndarray:
    """
    Extract embedding vector from a PIL Image using the ViT model.
    
    Args:
        image: PIL Image object
        extractor: Feature extractor from Hugging Face
        model: ViT model from Hugging Face
        
    Returns:
        Embedding vector as numpy array
    """
    # Ensure image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess image
    image_pp = extractor(image, return_tensors="pt")
    
    # Extract features (CLS token embedding)
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    
    return features.squeeze()


def load_dataset_with_faiss(dataset_repo: str) -> Tuple[Any, np.ndarray]:
    """
    Load the dataset with precomputed embeddings and add FAISS index.
    
    Args:
        dataset_repo: Hugging Face dataset repository name
        
    Returns:
        Tuple of (dataset_with_faiss_index, embeddings_matrix)
    """
    print(f"\nLoading dataset from: {dataset_repo}")
    dataset = load_dataset(dataset_repo, split="train")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Extract all embeddings as a numpy matrix for brute-force search
    print("Extracting embeddings matrix...")
    embeddings = np.array(dataset["embeddings"])
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Add FAISS index for efficient similarity search
    print("Building FAISS index...")
    dataset.add_faiss_index(column="embeddings")
    print("FAISS index built successfully.")
    
    return dataset, embeddings


def faiss_search(
    dataset: Any, 
    query_embedding: np.ndarray, 
    k: int
) -> Tuple[np.ndarray, List[int], float]:
    """
    Perform similarity search using FAISS index.
    
    Args:
        dataset: Dataset with FAISS index
        query_embedding: Query embedding vector
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Tuple of (distances, indices, time_ms)
    """
    start_time = time.perf_counter()
    
    # Use search method which returns scores and indices directly
    # Note: returns flat arrays for single query
    scores, indices = dataset.search("embeddings", query_embedding.reshape(1, -1), k=k)
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    
    # Ensure proper numpy array types
    scores = np.array(scores).flatten()
    indices = [int(i) for i in np.array(indices).flatten()]
    
    return scores, indices, time_ms


def bruteforce_cosine_search(
    embeddings: np.ndarray, 
    query_embedding: np.ndarray, 
    k: int
) -> Tuple[np.ndarray, List[int], float]:
    """
    Perform brute-force cosine similarity search over all embeddings.
    
    Args:
        embeddings: Matrix of all embeddings (N x D)
        query_embedding: Query embedding vector (D,)
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Tuple of (similarity_scores, indices, time_ms)
    """
    start_time = time.perf_counter()
    
    # Compute cosine similarity between query and all embeddings
    query_reshaped = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_reshaped, embeddings)[0]
    
    # Get top-K indices (highest similarity)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_scores = similarities[top_k_indices]
    
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    
    return top_k_scores, top_k_indices.tolist(), time_ms


def convert_faiss_distance_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Convert FAISS L2 distances to similarity scores (0-1 range).
    
    FAISS returns L2 (Euclidean) distances by default.
    Lower distance = higher similarity.
    
    For normalized embeddings: L2_distance = sqrt(2 * (1 - cosine_similarity))
    So: cosine_similarity = 1 - (L2_distance^2 / 2)
    
    For non-normalized embeddings, we use exponential decay with adaptive scaling.
    
    Args:
        distances: Array of L2 distances
        
    Returns:
        Array of similarity scores (0-1 range)
    """
    # For ViT embeddings (non-normalized), typical L2 distances range from 0 to ~600
    # Using exponential decay with scale based on typical distances
    # scale = 100 gives: dist=0 -> 1.0, dist=100 -> 0.37, dist=300 -> 0.05
    scale = 100.0
    similarities = np.exp(-distances / scale)
    return similarities


def run_benchmark(
    dataset: Any,
    embeddings: np.ndarray,
    extractor: Any,
    model: Any,
    num_queries: int,
    top_k: int,
    seed: int = 42
) -> Tuple[List[RetrievalResult], BenchmarkSummary]:
    """
    Run the complete benchmark comparing FAISS and brute-force search.
    
    Args:
        dataset: Dataset with FAISS index
        embeddings: Matrix of all embeddings
        extractor: Feature extractor
        model: ViT model
        num_queries: Number of queries to run
        top_k: Number of results to retrieve
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of RetrievalResult, BenchmarkSummary)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Select random query indices
    total_samples = len(dataset)
    query_indices = random.sample(range(total_samples), min(num_queries, total_samples))
    
    results = []
    faiss_times = []
    bruteforce_times = []
    
    print(f"\n{'='*70}")
    print("RUNNING BENCHMARK")
    print(f"{'='*70}")
    print(f"Number of queries: {num_queries}")
    print(f"Top-K results: {top_k}")
    print(f"Total embeddings in index: {total_samples}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"{'='*70}\n")
    
    for i, query_idx in enumerate(query_indices):
        print(f"\n--- Query {i+1}/{num_queries} (Dataset Index: {query_idx}) ---")
        
        # Get query image and extract embedding
        query_image = dataset[query_idx]["image"]
        query_embedding = extract_embedding(query_image, extractor, model)
        
        # FAISS search
        faiss_distances, faiss_indices, faiss_time = faiss_search(
            dataset, query_embedding, top_k
        )
        faiss_similarities = convert_faiss_distance_to_similarity(faiss_distances)
        faiss_times.append(faiss_time)
        
        # Brute-force cosine similarity search
        bf_scores, bf_indices, bf_time = bruteforce_cosine_search(
            embeddings, query_embedding, top_k
        )
        bruteforce_times.append(bf_time)
        
        # Store result
        result = RetrievalResult(
            query_index=query_idx,
            faiss_time_ms=faiss_time,
            bruteforce_time_ms=bf_time,
            faiss_scores=faiss_similarities.tolist(),
            bruteforce_scores=bf_scores.tolist(),
            faiss_indices=faiss_indices,
            bruteforce_indices=bf_indices
        )
        results.append(result)
        
        # Print per-query results
        print(f"  FAISS retrieval time:      {faiss_time:8.3f} ms")
        print(f"  Brute-force retrieval time:{bf_time:8.3f} ms")
        print(f"  Speedup factor:            {bf_time/faiss_time:8.2f}x")
        
        # Calculate recall for this query (how many of BF results are in FAISS results)
        overlap = len(set(faiss_indices) & set(bf_indices))
        query_recall = overlap / top_k
        print(f"  Recall@{top_k}:              {query_recall:8.2%}")
        
        print(f"\n  Top-{top_k} FAISS Results (Index: Similarity Score):")
        for j, (idx, score) in enumerate(zip(faiss_indices, faiss_similarities)):
            print(f"    {j+1}. Index {idx:6d}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"\n  Top-{top_k} Brute-Force Cosine Results (Index: Similarity Score):")
        for j, (idx, score) in enumerate(zip(bf_indices, bf_scores)):
            print(f"    {j+1}. Index {idx:6d}: {score:.4f} ({score*100:.2f}%)")
    
    # Calculate overall recall
    all_recalls = []
    for result in results:
        overlap = len(set(result.faiss_indices) & set(result.bruteforce_indices))
        all_recalls.append(overlap / top_k)
    avg_recall = np.mean(all_recalls)
    
    # Compute summary statistics
    summary = BenchmarkSummary(
        num_queries=num_queries,
        top_k=top_k,
        total_embeddings=total_samples,
        embedding_dim=embeddings.shape[1],
        avg_faiss_time_ms=np.mean(faiss_times),
        std_faiss_time_ms=np.std(faiss_times),
        min_faiss_time_ms=np.min(faiss_times),
        max_faiss_time_ms=np.max(faiss_times),
        avg_bruteforce_time_ms=np.mean(bruteforce_times),
        std_bruteforce_time_ms=np.std(bruteforce_times),
        min_bruteforce_time_ms=np.min(bruteforce_times),
        max_bruteforce_time_ms=np.max(bruteforce_times),
        speedup_factor=np.mean(bruteforce_times) / np.mean(faiss_times),
        recall_at_k=avg_recall
    )
    
    return results, summary


def print_summary(summary: BenchmarkSummary) -> None:
    """
    Print a formatted summary of the benchmark results.
    Suitable for inclusion in research papers.
    """
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL CONFIGURATION                       │
├────────────────────────────────────────────────────────────────────┤
│  Number of Queries:          {summary.num_queries:>8d}                             │
│  Top-K Results:              {summary.top_k:>8d}                             │
│  Total Embeddings:           {summary.total_embeddings:>8d}                             │
│  Embedding Dimension:        {summary.embedding_dim:>8d}                             │
│  Model:                      ViT-Base-Patch16-224                   │
├────────────────────────────────────────────────────────────────────┤
│                       RETRIEVAL PERFORMANCE                         │
├────────────────────────────────────────────────────────────────────┤
│  FAISS Search (L2 Distance):                                        │
│    Average Time:             {summary.avg_faiss_time_ms:>8.3f} ms (± {summary.std_faiss_time_ms:.3f} ms)        │
│    Min / Max Time:           {summary.min_faiss_time_ms:>8.3f} ms / {summary.max_faiss_time_ms:.3f} ms          │
│                                                                     │
│  Brute-Force Cosine Search:                                         │
│    Average Time:             {summary.avg_bruteforce_time_ms:>8.3f} ms (± {summary.std_bruteforce_time_ms:.3f} ms)        │
│    Min / Max Time:           {summary.min_bruteforce_time_ms:>8.3f} ms / {summary.max_bruteforce_time_ms:.3f} ms          │
│                                                                     │
│  Speedup Factor (BF/FAISS):  {summary.speedup_factor:>8.2f}x                            │
│  Recall@{summary.top_k} (FAISS vs BF):    {summary.recall_at_k:>8.2%}                            │
└────────────────────────────────────────────────────────────────────┘
""")
    
    # Print LaTeX-friendly table format
    print("\n--- LaTeX Table Format (for research papers) ---")
    print(r"""
\begin{table}[h]
\centering
\caption{Retrieval Performance Comparison}
\label{tab:retrieval_performance}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Avg. (ms)} & \textbf{Std. (ms)} & \textbf{Min (ms)} & \textbf{Max (ms)} & \textbf{Recall@K} \\
\midrule""")
    print(f"FAISS (L2) & {summary.avg_faiss_time_ms:.3f} & {summary.std_faiss_time_ms:.3f} & {summary.min_faiss_time_ms:.3f} & {summary.max_faiss_time_ms:.3f} & {summary.recall_at_k:.2%} \\\\")
    print(f"Brute-Force (Cosine) & {summary.avg_bruteforce_time_ms:.3f} & {summary.std_bruteforce_time_ms:.3f} & {summary.min_bruteforce_time_ms:.3f} & {summary.max_bruteforce_time_ms:.3f} & 100.00\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")


def print_detailed_results(results: List[RetrievalResult], top_k: int) -> None:
    """
    Print detailed per-query results in a tabular format.
    """
    print(f"\n{'='*70}")
    print("DETAILED RESULTS PER QUERY")
    print(f"{'='*70}")
    
    print("\n--- FAISS Similarity Scores ---")
    print(f"{'Query':>6} | " + " | ".join([f"Top-{i+1:d}" for i in range(top_k)]))
    print("-" * (8 + top_k * 9))
    
    for result in results:
        scores_str = " | ".join([f"{s:.4f}" for s in result.faiss_scores])
        print(f"{result.query_index:>6} | {scores_str}")
    
    print("\n--- Brute-Force Cosine Similarity Scores ---")
    print(f"{'Query':>6} | " + " | ".join([f"Top-{i+1:d}" for i in range(top_k)]))
    print("-" * (8 + top_k * 9))
    
    for result in results:
        scores_str = " | ".join([f"{s:.4f}" for s in result.bruteforce_scores])
        print(f"{result.query_index:>6} | {scores_str}")
    
    # Compute average scores across queries
    print("\n--- Average Similarity Scores Across All Queries ---")
    avg_faiss_scores = np.mean([r.faiss_scores for r in results], axis=0)
    avg_bf_scores = np.mean([r.bruteforce_scores for r in results], axis=0)
    
    print(f"{'Method':>20} | " + " | ".join([f"Top-{i+1:d}" for i in range(top_k)]))
    print("-" * (22 + top_k * 9))
    print(f"{'FAISS':>20} | " + " | ".join([f"{s:.4f}" for s in avg_faiss_scores]))
    print(f"{'Brute-Force Cosine':>20} | " + " | ".join([f"{s:.4f}" for s in avg_bf_scores]))


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run the evaluation."""
    print("="*70)
    print("VISUAL PRODUCT SIMILARITY SEARCH EVALUATION")
    print("="*70)
    print(f"Model: {MODEL_CKPT}")
    print(f"Dataset: {DATASET_REPO}")
    print(f"Top-K: {TOP_K}")
    print(f"Number of Queries: {NUM_QUERIES}")
    print("="*70)
    
    # Load model and feature extractor
    extractor, model = load_model_and_extractor(MODEL_CKPT)
    
    # Load dataset with FAISS index
    dataset, embeddings = load_dataset_with_faiss(DATASET_REPO)
    
    # Run benchmark
    results, summary = run_benchmark(
        dataset=dataset,
        embeddings=embeddings,
        extractor=extractor,
        model=model,
        num_queries=NUM_QUERIES,
        top_k=TOP_K,
        seed=RANDOM_SEED
    )
    
    # Print summary
    print_summary(summary)
    
    # Print detailed results
    print_detailed_results(results, TOP_K)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    return results, summary


if __name__ == "__main__":
    main()
