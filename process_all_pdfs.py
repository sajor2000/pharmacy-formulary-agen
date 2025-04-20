#!/usr/bin/env python3
"""
Process All PDFs
---------------
Script to process all PDF files in the data directory and store their embeddings in Pinecone.
"""

import os
import time
from document_processor import DocumentProcessor

def process_pdfs_in_batches(batch_size=5, delay_between_batches=5):
    """Process PDFs in batches with a delay between batches"""
    processor = DocumentProcessor()
    
    # Get list of all PDF files
    pdf_files = [f for f in os.listdir(processor.pdf_dir) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to process")
    
    # Process in batches
    all_embeddings = []
    for i in range(0, total_files, batch_size):
        batch = pdf_files[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"\n=== Processing Batch {batch_num}/{total_batches} ===")
        print(f"Files: {batch}")
        
        # Process this batch
        batch_embeddings = []
        for filename in batch:
            pdf_path = os.path.join(processor.pdf_dir, filename)
            print(f"\nProcessing {filename}...")
            
            try:
                # Extract text
                text = processor.extract_text_from_pdf(pdf_path)
                
                # Extract tables
                tables = processor.extract_tables_from_pdf(pdf_path)
                
                # Create embeddings
                file_embeddings = []
                
                # Embedding for full text
                if text:
                    print(f"Creating embedding for full text...")
                    text_embedding = processor.get_embedding(text)
                    if isinstance(text_embedding, list) and len(text_embedding) > 0:
                        file_embeddings.append({
                            'content': text[:1000] + '...',  # Just store preview
                            'embedding': text_embedding,
                            'metadata': {'source': filename, 'type': 'full_text', 'insurance': get_insurance_from_filename(filename)}
                        })
                
                # Embeddings for tables
                for i, table in enumerate(tables):
                    if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                        print(f"Creating embedding for table {i+1}...")
                        table_str = table['data'].to_string()
                        table_embedding = processor.get_embedding(table_str)
                        if isinstance(table_embedding, list) and len(table_embedding) > 0:
                            file_embeddings.append({
                                'content': f"Table from page {table['page']}:\n{table_str[:500]}...",
                                'embedding': table_embedding,
                                'metadata': {
                                    'source': filename,
                                    'type': 'table',
                                    'page': table['page'],
                                    'insurance': get_insurance_from_filename(filename)
                                }
                            })
                
                batch_embeddings.extend(file_embeddings)
                print(f"Created {len(file_embeddings)} embeddings for {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Store batch embeddings in Pinecone
        if batch_embeddings:
            print(f"\nStoring {len(batch_embeddings)} embeddings from batch {batch_num} in Pinecone...")
            success = processor.store_in_pinecone(batch_embeddings)
            if success:
                print(f"Successfully stored batch {batch_num} embeddings in Pinecone")
                all_embeddings.extend(batch_embeddings)
            else:
                print(f"Failed to store batch {batch_num} embeddings in Pinecone")
        
        # Delay between batches to avoid rate limits
        if i + batch_size < total_files:
            print(f"\nWaiting {delay_between_batches} seconds before processing next batch...")
            time.sleep(delay_between_batches)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total embeddings created and stored: {len(all_embeddings)}")
    return all_embeddings

def get_insurance_from_filename(filename):
    """Extract insurance provider from filename"""
    # Common insurance providers in the filenames
    providers = {
        "UHC": "UnitedHealthcare",
        "BCBS": "Blue Cross Blue Shield",
        "Cigna": "Cigna",
        "Express Scripts": "Express Scripts",
        "Humana": "Humana",
        "Meridian": "Meridian",
        "Wellcare": "Wellcare",
        "County Care": "County Care"
    }
    
    for key, value in providers.items():
        if key in filename:
            # Check for specific plan types
            if "HMO" in filename:
                return f"{value} HMO"
            elif "PPO" in filename:
                return f"{value} PPO"
            elif "Medicare" in filename:
                return f"{value} Medicare"
            else:
                return value
    
    return "Unknown Insurance"

if __name__ == "__main__":
    import pandas as pd
    
    print("=== Processing All Formulary PDFs ===")
    print("This script will process all PDF files in the data directory and store their embeddings in Pinecone.")
    print("Processing will be done in batches to avoid overwhelming the system.\n")
    
    # Process PDFs in batches of 5 with a 5-second delay between batches
    all_embeddings = process_pdfs_in_batches(batch_size=5, delay_between_batches=5)
    
    print("\nAll PDFs have been processed and their embeddings stored in Pinecone.")
    print("You can now use the query interface to search for medications across all formularies.")
