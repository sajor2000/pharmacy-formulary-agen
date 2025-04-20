#!/usr/bin/env python3
"""
Check Pinecone Status
--------------------
Script to check the status of the Pinecone index and see what PDFs have been processed.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def check_pinecone_status():
    """Check the status of the Pinecone index"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # List available indexes
        indexes = pc.list_indexes()
        print(f"Available Pinecone indexes: {indexes.names()}")
        
        # Connect to the 'form' index
        index = pc.Index("form")
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"\nIndex statistics: {stats}")
        
        # Check which PDFs have been processed
        if 'namespaces' in stats and stats['namespaces'] and 'formulary' in stats['namespaces']:
            total_vectors = stats['namespaces']['formulary']['vector_count']
            print(f"\nTotal vectors in 'formulary' namespace: {total_vectors}")
        else:
            print("\nNo vectors found in 'formulary' namespace")
        
        # Query to see what sources are in the index
        query_embedding = [0.1] * 1024  # Create a dummy vector for querying
        results = index.query(
            namespace="formulary",
            vector=query_embedding,
            top_k=100,  # Get a large number to see more sources
            include_metadata=True
        )
        
        # Extract unique source files
        sources = set()
        for match in results.matches:
            if hasattr(match, 'metadata') and match.metadata and 'source' in match.metadata:
                sources.add(match.metadata['source'])
        
        # Print the sources
        print(f"\nFound {len(sources)} unique source files in Pinecone:")
        for i, source in enumerate(sorted(sources), 1):
            print(f"{i}. {source}")
        
        # Compare with files in data directory
        data_dir = "data"
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
        
        print(f"\nTotal PDF files in data directory: {len(pdf_files)}")
        
        # Find files that haven't been processed
        unprocessed = set(pdf_files) - sources
        if unprocessed:
            print(f"\nFiles not yet processed ({len(unprocessed)}):")
            for i, file in enumerate(sorted(unprocessed), 1):
                print(f"{i}. {file}")
        else:
            print("\nAll PDF files have been processed!")
        
        return True
    except Exception as e:
        print(f"Error checking Pinecone status: {e}")
        return False

if __name__ == "__main__":
    print("Checking Pinecone status...")
    check_pinecone_status()
