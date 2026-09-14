#!/usr/bin/env python3
"""
Multi-agent document similarity system.
Compares PDFs and Pages documents against an indicator PDF.
"""

import sys
import time
from pathlib import Path
import json
import argparse

from document_parser import batch_parse_documents
from embedding_agent import EmbeddingAgent
from similarity_search import SimilaritySearchAgent

def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Query-by-example semantic corpus search"
    )
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="Reference document to rank the corpus against"
    )
    parser.add_argument(
        "--search-dir", type=Path, default=Path("."),
        help="Directory of documents to search"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("similarity_results.json"),
        help="Where to write the ranked results"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Minimum cosine similarity to report a match"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Maximum number of matches to return"
    )
    args = parser.parse_args()

    indicator_path = args.reference
    search_directory = args.search_dir
    output_path = args.output
    threshold = args.threshold
    
    print("=== Document Similarity Search System ===")
    print(f"Indicator document: {indicator_path}")
    print(f"Search directory: {search_directory}")
    print(f"Similarity threshold: {threshold}")
    print()
    
    # Step 1: Parse all documents
    print("Step 1: Parsing documents...")
    start_time = time.time()
    
    # Parse indicator document
    from document_parser import parse_document
    indicator_text = parse_document(indicator_path)
    if not indicator_text.strip():
        print("ERROR: Could not extract text from indicator document")
        sys.exit(1)
    
    # Parse all documents in search directory
    all_documents = batch_parse_documents(search_directory)
    all_documents[str(indicator_path)] = indicator_text
    
    parse_time = time.time() - start_time
    print(f"Parsed {len(all_documents)} documents in {parse_time:.2f} seconds")
    print()
    
    # Step 2: Generate embeddings
    print("Step 2: Generating embeddings...")
    start_time = time.time()
    
    embedding_agent = EmbeddingAgent()
    embeddings = embedding_agent.generate_embeddings(all_documents)
    
    embed_time = time.time() - start_time
    print(f"Generated embeddings in {embed_time:.2f} seconds")
    print()
    
    # Step 3: Find similar documents
    print("Step 3: Finding similar documents...")
    start_time = time.time()
    
    search_agent = SimilaritySearchAgent(threshold=threshold)
    matches = search_agent.find_matches_for_query(
        query_id=str(indicator_path),
        embeddings=embeddings,
        threshold=threshold,
        top_k=args.top_k
    )
    
    search_time = time.time() - start_time
    print(f"Found {len(matches)} matches above threshold {threshold}")
    print(f"Search completed in {search_time:.2f} seconds")
    print()
    
    # Step 4: Display results
    print("=== Top Matches ===")
    for i, (doc_path, score) in enumerate(matches[:10], 1):
        print(f"{i:2d}. {score:.3f} - {doc_path}")
    
    # Analyze distribution
    distribution = search_agent.analyze_distribution(matches)
    print(f"\n=== Score Distribution ===")
    print(f"Mean: {distribution.get('mean', 0):.3f}")
    print(f"Std: {distribution.get('std', 0):.3f}")
    print(f"Range: {distribution.get('min', 0):.3f} - {distribution.get('max', 0):.3f}")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    search_agent.save_results(matches, output_path)
    
    print("\n=== Complete ===")
    total_time = parse_time + embed_time + search_time
    print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()