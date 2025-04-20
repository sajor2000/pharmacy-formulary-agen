import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

from torchvision import transforms

from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import fitz  # PyMuPDF
import numpy as np
from sklearn.decomposition import PCA

# We'll use a simpler approach without LlamaIndex document objects
# since the import structure may vary between versions

# Set device for PyTorch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load API keys from environment variables
from dotenv import load_dotenv
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Get Pinecone API key from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

class DocumentProcessor:
    def __init__(self, pdf_dir="data"):
        self.pdf_dir = pdf_dir
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def get_embedding(self, text):
        """Get embedding for text and resize to match Pinecone index dimensions (1024)"""
        try:
            # Truncate text if too long (embedding model has token limits)
            max_tokens = 8000  # embedding models typically have 8k token limit
            # Simple truncation - in production you'd want to chunk properly
            if len(text) > max_tokens * 4:  # rough estimate of chars to tokens
                text = text[:max_tokens * 4]
            
            # Use OpenAI's standard embedding model
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Get the embedding vector
            embedding = response.data[0].embedding
            
            # Resize to match Pinecone index dimensions (1024)
            resized_embedding = self._resize_embedding(embedding, target_dim=1024)
            
            return resized_embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables from a PDF file using both PyMuPDF and LlamaIndex"""
        tables = []
        try:
            # Method 1: Extract tables using PyMuPDF
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                tab = page.find_tables()
                if tab.tables:
                    for i, table in enumerate(tab.tables):
                        df = table.to_pandas()
                        tables.append({
                            'page': page_num + 1,
                            'table_num': i + 1,
                            'source': 'pymupdf',
                            'data': df
                        })
            
            # Method 2: Use standard PDFReader from LlamaIndex as a fallback
            try:
                pdf_reader = PDFReader()
                llama_docs = pdf_reader.load_data(pdf_path)
                
                # Look for potential tables in the text
                for i, doc in enumerate(llama_docs):
                    text = doc.text
                    # Simple heuristic to detect tables: look for text with multiple lines and consistent delimiters
                    lines = text.strip().split('\n')
                    
                    # Check if this might be a table (has multiple lines with similar structure)
                    if len(lines) > 3:  # Need at least a few lines for a table
                        # Try to detect table-like structures
                        delimiter_counts = [line.count('|') for line in lines[:5] if line.strip()]
                        space_pattern = [len(line.split()) for line in lines[:5] if line.strip()]
                        
                        # If consistent pattern of delimiters or spaces, might be a table
                        if (len(set(delimiter_counts)) <= 2 and max(delimiter_counts) > 1) or \
                           (len(set(space_pattern)) <= 2 and min(space_pattern) >= 3):
                            
                            # Try to create a DataFrame
                            try:
                                # For pipe-delimited tables
                                if '|' in text:
                                    # Get non-empty lines
                                    table_lines = [line for line in lines if line.strip()]
                                    if len(table_lines) > 1:
                                        # First line as header
                                        headers = [h.strip() for h in table_lines[0].split('|') if h.strip()]
                                        data = []
                                        for line in table_lines[1:]:  # Skip header
                                            if '|' in line:
                                                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                                if row:
                                                    data.append(row)
                                        
                                        if headers and data:
                                            # Handle mismatched columns
                                            max_cols = max(len(headers), max(len(row) for row in data))
                                            headers = headers + [''] * (max_cols - len(headers))
                                            data = [row + [''] * (max_cols - len(row)) for row in data]
                                            df = pd.DataFrame(data, columns=headers)
                                            
                                            tables.append({
                                                'page': doc.metadata.get('page_label', i+1),
                                                'table_num': len(tables) + 1,
                                                'source': 'llamaindex',
                                                'data': df
                                            })
                                # For space-delimited tables
                                else:
                                    # Try pandas' read_csv with string buffer
                                    import io
                                    df = pd.read_csv(io.StringIO(text), sep='\\s+', engine='python')
                                    if not df.empty:
                                        tables.append({
                                            'page': doc.metadata.get('page_label', i+1),
                                            'table_num': len(tables) + 1,
                                            'source': 'llamaindex_space',
                                            'data': df
                                        })
                            except Exception as table_e:
                                # Not a table or couldn't parse
                                pass
            except Exception as llama_e:
                print(f"Error with LlamaIndex extraction: {llama_e}")
                
            return tables
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {e}")
            return []
    
    def process_all_pdfs(self):
        """Process all PDFs in the directory"""
        results = {}
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                print(f"Processing {filename}...")
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                # Extract tables
                tables = self.extract_tables_from_pdf(pdf_path)
                
                # Create simple document structures for better querying
                docs = []
                
                # Create a document from the full text
                if text:
                    docs.append({
                        "text": text,
                        "metadata": {"source": filename, "type": "full_text"}
                    })
                
                # Create documents for each table
                for table in tables:
                    if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                        # Convert DataFrame to string representation
                        table_str = table['data'].to_string()
                        docs.append({
                            "text": f"Table from page {table['page']}, table #{table['table_num']}:\n{table_str}",
                            "metadata": {
                                "source": filename,
                                "type": "table",
                                "page": table['page'],
                                "table_num": table['table_num']
                            }
                        })
                
                # Create embeddings for text and tables
                embeddings = []
                
                # Embedding for full text
                if text:
                    text_embedding = self.get_embedding(text)
                    if text_embedding:
                        embeddings.append({
                            'content': text[:1000] + '...',  # Just store preview
                            'embedding': text_embedding,
                            'metadata': {'source': filename, 'type': 'full_text'}
                        })
                
                # Embeddings for tables
                for table in tables:
                    if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                        table_str = table['data'].to_string()
                        table_embedding = self.get_embedding(table_str)
                        if isinstance(table_embedding, list) and len(table_embedding) > 0:
                            embeddings.append({
                                'content': f"Table from page {table['page']}:\n{table_str[:500]}...",
                                'embedding': table_embedding,
                                'metadata': {
                                    'source': filename,
                                    'type': 'table',
                                    'page': table['page']
                                }
                            })
                
                results[filename] = {
                    'text': text,
                    'tables': tables,
                    'docs': docs,
                    'embeddings': embeddings
                }
        
        return results
    
    def analyze_formulary_with_gpt(self, text, tables=None, prompt=None):
        """Analyze formulary text and tables with GPT"""
        if prompt is None:
            prompt = """
            Analyze this formulary document and extract the following information:
            1. All respiratory medications listed
            2. Their formulary tiers
            3. Prior authorization requirements
            4. Quantity limits
            5. Step therapy requirements
            
            Pay special attention to information in tables, which contain most of the formulary details.
            Format the response as a structured JSON.
            """
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for improved capabilities
                messages=[
                    {"role": "system", "content": "You are a pharmacy formulary specialist who extracts and structures medication information from formulary documents. You have expertise in respiratory medications, insurance formularies, and pharmacy benefit management."},
                    {"role": "user", "content": prompt + "\n\nDocument text:\n" + text[:15000]}  # Limit text to avoid token limits
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing with GPT: {e}")
            return ""

    def _resize_embedding(self, embedding, target_dim=1024):
        """Resize an embedding vector to the target dimension using a simpler method"""
        try:
            # Convert to numpy array if it's not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Get current dimensions
            current_dim = embedding.shape[0]
            
            if current_dim == target_dim:
                return embedding.tolist()
            
            # For dimensionality reduction, use a simpler approach
            # We'll take a subset of the original dimensions and normalize
            if current_dim > target_dim:
                # Take every nth element to get target_dim elements
                indices = np.round(np.linspace(0, current_dim - 1, target_dim)).astype(int)
                reduced = embedding[indices]
                
                # Normalize to preserve the vector magnitude
                norm = np.linalg.norm(reduced)
                if norm > 0:
                    reduced = reduced / norm
                
                return reduced.tolist()
            
            # If we need to increase dimensions, pad with zeros
            if current_dim < target_dim:
                padded = np.zeros(target_dim)
                padded[:current_dim] = embedding
                
                # Normalize to preserve the vector magnitude
                norm = np.linalg.norm(padded)
                if norm > 0:
                    padded = padded / norm
                
                return padded.tolist()
            
        except Exception as e:
            print(f"Error resizing embedding: {e}")
            # If resizing fails, create a random normalized vector
            random_vec = np.random.randn(target_dim)
            random_vec = random_vec / np.linalg.norm(random_vec)
            return random_vec.tolist()
    
    def store_in_pinecone(self, embeddings, index_name="form", namespace="formulary"):
        """Store embeddings in Pinecone using the existing index"""
        try:
            from pinecone import Pinecone
            import numpy as np
            
            print(f"\nStoring {len(embeddings)} embeddings in Pinecone...")
            
            # Initialize Pinecone with the exact approach from the documentation
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Connect to the existing index
            index = pc.Index(index_name)
            print(f"Connected to existing index: {index_name}")
            
            # Prepare vectors for upsert in the exact format from the documentation
            vectors = []
            for i, item in enumerate(embeddings):
                # Convert numpy arrays to lists if needed
                embedding = item['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                vectors.append({
                    'id': f"{item['metadata']['source']}_{item['metadata']['type']}_{i}",
                    'values': embedding,
                    'metadata': {
                        'content': item['content'],
                        **item['metadata']
                    }
                })
            
            # Upsert in batches (Pinecone has limits)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                print(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            print(f"Successfully stored {len(vectors)} vectors in Pinecone index '{index_name}', namespace '{namespace}'")
            return True
        except Exception as e:
            print(f"Error storing in Pinecone: {e}")
            return False

if __name__ == "__main__":
    # Create a test processor that only processes a few files
    processor = DocumentProcessor()
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(processor.pdf_dir) if f.endswith('.pdf')]
    
    # Select just 2 files to test with
    test_files = pdf_files[:2] if len(pdf_files) >= 2 else pdf_files
    
    print(f"Testing with {len(test_files)} files: {test_files}")
    
    # Process just these files
    all_embeddings = []
    results = {}
    for filename in test_files:
        pdf_path = os.path.join(processor.pdf_dir, filename)
        print(f"Processing {filename}...")
        
        # Extract text
        text = processor.extract_text_from_pdf(pdf_path)
        
        # Extract tables
        tables = processor.extract_tables_from_pdf(pdf_path)
        
        # Create embeddings
        embeddings = []
        
        # Embedding for full text
        if text:
            print(f"Creating embedding for full text...")
            text_embedding = processor.get_embedding(text)
            if isinstance(text_embedding, list) and len(text_embedding) > 0:
                embeddings.append({
                    'content': text[:1000] + '...',  # Just store preview
                    'embedding': text_embedding,
                    'metadata': {'source': filename, 'type': 'full_text'}
                })
        
        # Embeddings for tables
        for i, table in enumerate(tables):
            if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                print(f"Creating embedding for table {i+1}...")
                table_str = table['data'].to_string()
                table_embedding = processor.get_embedding(table_str)
                if table_embedding:
                    embeddings.append({
                        'content': f"Table from page {table['page']}:\n{table_str[:500]}...",
                        'embedding': table_embedding,
                        'metadata': {
                            'source': filename,
                            'type': 'table',
                            'page': table['page']
                        }
                    })
        
        all_embeddings.extend(embeddings)
        
        results[filename] = {
            'text': text,
            'tables': tables,
            'embeddings': embeddings
        }
    
    # Analyze just the first file
    if results:
        first_pdf = list(results.keys())[0]
        text = results[first_pdf]['text']
        tables = results[first_pdf]['tables']
        embeddings = results[first_pdf]['embeddings']
        
        # Print table information
        print(f"\nFound {len(tables)} tables in {first_pdf}")
        if tables:
            for i, table in enumerate(tables[:2]):  # Show first 2 tables only
                print(f"\nTable {i+1} from page {table['page']} (source: {table['source']})")
                if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                    print(table['data'].head(3))  # Show first 3 rows
        
        # Print embedding information
        print(f"\nCreated {len(embeddings)} embeddings for {first_pdf}")
        
        # Add table information to the analysis
        table_text = ""
        for table in tables[:5]:  # Include first 5 tables in analysis
            if isinstance(table['data'], pd.DataFrame) and not table['data'].empty:
                table_text += f"\nTable from page {table['page']}:\n{table['data'].to_string(max_rows=10)}\n"
        
        print(f"\nAnalyzing {first_pdf} with GPT-4o (including table data)...")
        analysis = processor.analyze_formulary_with_gpt(text + "\n" + table_text if table_text else text)
        print(f"\nAnalysis of {first_pdf}:\n{analysis}")
        
        # Store embeddings in Pinecone
        print("\nStoring embeddings in Pinecone...")
        if all_embeddings:
            success = processor.store_in_pinecone(all_embeddings)
            if success:
                print(f"Successfully stored {len(all_embeddings)} embeddings in Pinecone")
            else:
                print("Failed to store embeddings in Pinecone")
