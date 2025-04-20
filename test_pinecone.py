#!/usr/bin/env python3
"""
Pinecone Connection Test
-----------------------
Simple script to test the Pinecone connection and API key.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def test_pinecone_connection():
    """Test the Pinecone connection with the provided API key"""
    try:
        print(f"Testing Pinecone connection with API key: {PINECONE_API_KEY[:5]}...{PINECONE_API_KEY[-5:]}")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # List available indexes
        indexes = pc.list_indexes()
        print(f"\nAvailable indexes: {indexes.names()}")
        
        # Try to connect to the 'form' index
        try:
            index = pc.Index("form")
            print(f"\nSuccessfully connected to 'form' index")
            
            # Get index stats
            stats = index.describe_index_stats()
            print(f"\nIndex stats: {stats}")
            
            # Test a simple upsert with a single vector
            test_vector = {
                'id': 'test_vector',
                'values': [0.1] * 1024,  # 1024-dimensional vector
                'metadata': {'test': 'test'}
            }
            
            print("\nAttempting to upsert a test vector...")
            index.upsert(vectors=[test_vector], namespace="test")
            print("Successfully upserted test vector")
            
            # Test a simple query
            print("\nAttempting to query the test vector...")
            results = index.query(
                namespace="test",
                vector=[0.1] * 1024,
                top_k=1,
                include_values=True,
                include_metadata=True
            )
            print(f"Query results: {results}")
            
            # Clean up the test vector
            print("\nCleaning up test vector...")
            index.delete(ids=["test_vector"], namespace="test")
            print("Test vector deleted")
            
            return True
        except Exception as e:
            print(f"\nError connecting to 'form' index: {e}")
            return False
            
    except Exception as e:
        print(f"\nError initializing Pinecone: {e}")
        return False

if __name__ == "__main__":
    success = test_pinecone_connection()
    if success:
        print("\n✅ Pinecone connection test successful!")
    else:
        print("\n❌ Pinecone connection test failed. Please check your API key and index configuration.")
