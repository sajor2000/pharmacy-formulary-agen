#!/usr/bin/env python3
"""
Pharmacy Formulary Query Interface
----------------------------------
A simple interface for querying the formulary database using the document processor
and formulary agent components.
"""

import os
import sys
from typing import List, Dict, Any
import openai
import pinecone
import pandas as pd
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from formulary_agent import FormularyAgent, FormularyResponse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

class FormularyQueryInterface:
    def __init__(self):
        """Initialize the query interface with document processor and formulary agent"""
        self.processor = DocumentProcessor()
        self.agent = FormularyAgent()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def direct_query(self, query_text: str) -> str:
        """
        Process a direct natural language query using GPT-4o
        
        This bypasses the structured query interface and allows for free-form questions
        about medications, insurance coverage, etc.
        """
        try:
            # First, search Pinecone for relevant context
            context = self._retrieve_context(query_text)
            
            # Then use GPT-4o to generate a response based on the context
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are a pharmacy formulary specialist assistant. 
                    You help healthcare providers and patients understand medication coverage under different insurance plans.
                    Focus on providing accurate, specific information about respiratory medications.
                    Always cite your sources and be clear about coverage details including tier, prior authorization, and quantity limits."""},
                    {"role": "user", "content": f"Question: {query_text}\n\nRelevant context from formulary documents:\n{context}"}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing query: {e}"
    
    def _retrieve_context(self, query: str, top_k: int = 5, namespace="formulary") -> str:
        """Retrieve relevant context from Pinecone using the existing index"""
        try:
            # Get embedding for the query
            query_embedding = self.processor.get_embedding(query)
            
            # Query Pinecone using the latest API
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index("form")  # Use the existing index name
            
            # Query in the exact format from the documentation
            results = index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
            
            # Format context from results
            context = ""
            for match in results.matches:
                metadata = match.metadata
                context += f"Source: {metadata.get('source', 'Unknown')}\n"
                context += f"Type: {metadata.get('type', 'Unknown')}\n"
                if 'page' in metadata:
                    context += f"Page: {metadata.get('page')}\n"
                context += f"Content: {metadata.get('content', '')}\n\n"
            
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No relevant context found."
    
    def structured_query(self, 
                        insurance_name: str,
                        medication_class: str,
                        brand_preference: str,
                        patient_age: int = None,
                        patient_conditions: List[str] = None) -> str:
        """Process a structured query using the formulary agent"""
        return self.agent.process_query(
            insurance_name=insurance_name,
            medication_class=medication_class,
            brand_preference=brand_preference,
            patient_age=patient_age,
            patient_conditions=patient_conditions
        )
    
    def run_interactive_interface(self):
        """Run an interactive command-line interface"""
        print("\n===== Pharmacy Formulary AI Assistant =====\n")
        print("This assistant helps you find information about medication coverage under different insurance plans.")
        
        while True:
            print("\nSelect query type:")
            print("1. Structured query (guided)")
            print("2. Direct question (free-form)")
            print("3. Process new formulary documents")
            print("4. Exit")
            
            try:
                choice = int(input("\nEnter choice (1-4): "))
                
                if choice == 1:
                    self._run_structured_query()
                elif choice == 2:
                    self._run_direct_query()
                elif choice == 3:
                    self._process_new_documents()
                elif choice == 4:
                    print("\nExiting. Thank you for using the Pharmacy Formulary AI Assistant.")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except KeyboardInterrupt:
                print("\n\nExiting. Thank you for using the Pharmacy Formulary AI Assistant.")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _run_structured_query(self):
        """Run the structured query interface"""
        try:
            print("\n=== Structured Formulary Query ===\n")
            
            insurance_name = input("Insurance name (e.g., UnitedHealthcare, Aetna): ")
            
            print("\nSelect medication class (enter number):")
            for i, med_class in enumerate(self.agent.get_medication_classes(), 1):
                print(f"{i}. {med_class}")
            class_choice = int(input("\nChoice: "))
            medication_class = self.agent.get_medication_classes()[class_choice-1]
            
            print("\nBrand preference:")
            print("1. Generic preferred")
            print("2. Brand preferred")
            print("3. No preference")
            brand_choice = int(input("\nChoice: "))
            brand_map = {1: "generic preferred", 2: "brand preferred", 3: "no"}
            brand_preference = brand_map[brand_choice]
            
            # Optional patient context
            include_patient = input("\nInclude patient context? (y/n): ").lower() == 'y'
            patient_age = None
            patient_conditions = None
            
            if include_patient:
                patient_age = int(input("Patient age: "))
                conditions_input = input("Patient conditions (comma separated, or leave blank): ")
                patient_conditions = [c.strip() for c in conditions_input.split(",")] if conditions_input else None
            
            print("\nQuerying formulary database...\n")
            response = self.structured_query(
                insurance_name=insurance_name,
                medication_class=medication_class,
                brand_preference=brand_preference,
                patient_age=patient_age,
                patient_conditions=patient_conditions
            )
            
            print("\n=== Recommendation ===\n")
            print(response)
            
        except KeyboardInterrupt:
            print("\nReturning to main menu.")
        except Exception as e:
            print(f"\nError: {e}")
    
    def _run_direct_query(self):
        """Run the direct query interface"""
        try:
            print("\n=== Direct Query Interface ===")
            print("Ask any question about medication coverage, formularies, or insurance plans.")
            print("Examples:")
            print("- What respiratory medications are covered by UnitedHealthcare?")
            print("- Does Aetna require prior authorization for Advair?")
            print("- What are the tier 1 asthma medications for Medicare Part D?")
            
            query = input("\nYour question: ")
            print("\nSearching formulary database...\n")
            
            response = self.direct_query(query)
            print("\n=== Answer ===\n")
            print(response)
            
        except KeyboardInterrupt:
            print("\nReturning to main menu.")
        except Exception as e:
            print(f"\nError: {e}")
    
    def _process_new_documents(self):
        """Process new formulary documents"""
        try:
            print("\n=== Process New Formulary Documents ===")
            print("This will scan the 'data' directory for PDF files and process them.")
            confirm = input("Continue? (y/n): ").lower()
            
            if confirm == 'y':
                print("\nProcessing documents...")
                # Get list of PDF files
                pdf_files = [f for f in os.listdir(self.processor.pdf_dir) if f.endswith('.pdf')]
                
                if not pdf_files:
                    print("No PDF files found in the 'data' directory.")
                    return
                
                print(f"Found {len(pdf_files)} PDF files.")
                
                # Process files
                all_embeddings = []
                for filename in pdf_files:
                    pdf_path = os.path.join(self.processor.pdf_dir, filename)
                    print(f"Processing {filename}...")
                    
                    # Extract text and tables
                    text = self.processor.extract_text_from_pdf(pdf_path)
                    tables = self.processor.extract_tables_from_pdf(pdf_path)
                    
                    # Create embeddings
                    if text:
                        print(f"Creating embedding for full text...")
                        text_embedding = self.processor.get_embedding(text)
                        if text_embedding:
                            all_embeddings.append({
                                'content': text[:1000] + '...',
                                'embedding': text_embedding,
                                'metadata': {'source': filename, 'type': 'full_text'}
                            })
                    
                    # Embeddings for tables
                    for i, table in enumerate(tables):
                        if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                            print(f"Creating embedding for table {i+1}...")
                            table_str = table['data'].to_string()
                            table_embedding = self.processor.get_embedding(table_str)
                            if table_embedding:
                                all_embeddings.append({
                                    'content': f"Table from page {table['page']}:\n{table_str[:500]}...",
                                    'embedding': table_embedding,
                                    'metadata': {
                                        'source': filename,
                                        'type': 'table',
                                        'page': table['page']
                                    }
                                })
                
                # Store in Pinecone
                if all_embeddings:
                    print(f"\nStoring {len(all_embeddings)} embeddings in Pinecone...")
                    success = self.processor.store_in_pinecone(all_embeddings)
                    if success:
                        print(f"Successfully stored {len(all_embeddings)} embeddings in Pinecone")
                    else:
                        print("Failed to store embeddings in Pinecone")
                else:
                    print("No embeddings created.")
            else:
                print("Operation cancelled.")
                
        except KeyboardInterrupt:
            print("\nReturning to main menu.")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    # Check if environment variables are set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set this in your .env file.")
        sys.exit(1)
    
    if not PINECONE_API_KEY or not PINECONE_ENV:
        print("Error: Pinecone API key or environment not found in environment variables.")
        print("Please set PINECONE_API_KEY and PINECONE_ENVIRONMENT in your .env file.")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF formulary documents to this directory.")
    
    # Run the interface
    interface = FormularyQueryInterface()
    interface.run_interactive_interface()
