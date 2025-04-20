import os
from typing import List
from dotenv import load_dotenv
# Updated LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pydantic import BaseModel
import pinecone
import openai

# Load environment variables
load_dotenv()

class MedicationRecommendation(BaseModel):
    name: str
    form: str
    device_type: str
    strength: str
    tier: str
    requirements: str
    quantity_limit: str
    estimated_copay: str = None

class AlternativeOption(BaseModel):
    name: str
    key_difference: str
    requirements: str

class FormularyResponse(BaseModel):
    primary_recommendation: MedicationRecommendation
    alternative_options: List[AlternativeOption]
    coverage_notes: str
    source: str

class FormularyAgent:
    def __init__(self, data_dir: str = "data"):
        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        
        # Initialize Pinecone with the latest API
        from pinecone import Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "form"  # Use the existing index name
        
        # Connect to the existing index
        pinecone_index = pc.Index(index_name)
        print(f"Connected to existing Pinecone index: {index_name}")

        # Set up LlamaIndex with GPT-4o for enhanced capabilities
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)
        )
        
        # Load and chunk documents
        self.documents = SimpleDirectoryReader(data_dir).load_data()
        
        # Use Pinecone as the vector store
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Create the index
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            service_context=service_context,
            vector_store=vector_store
        )
        self.query_engine = self.index.as_query_engine()

    def process_query(self, 
                     insurance_name: str,
                     medication_class: str,
                     brand_preference: str,
                     patient_age: int = None,
                     patient_conditions: List[str] = None) -> FormularyResponse:
        """Process a medication query with enhanced context"""
        # Construct the query with additional context if provided
        patient_context = ""
        if patient_age:
            patient_context += f"Patient age: {patient_age}. "
        if patient_conditions:
            patient_context += f"Patient conditions: {', '.join(patient_conditions)}. "
            
        query = f"""
        For {insurance_name} insurance with {brand_preference} preference,
        find formulary coverage for {medication_class} medications.
        {patient_context}
        
        Format the response as a structured recommendation including:
        1. Primary recommendation with full coverage details (name, form, device type, strength, tier, requirements, quantity limits)
        2. Alternative options with key differences and requirements
        3. Coverage notes and restrictions
        4. Source of the information (always specify whether from uploaded docs or external source)
        
        Focus on respiratory medications and provide specific details about prior authorization requirements.
        """
        
        # Get response from the index (RAG)
        response = self.query_engine.query(query)
        
        # Parse the response into structured format
        try:
            # For now, just return the raw response
            # In a production system, we would parse this into the FormularyResponse model
            print("\n===== RAG Response =====\n")
            print(response)
            print("\n=======================\n")
            return response
        except Exception as e:
            print(f"Error parsing response: {e}")
            return response

    def get_medication_classes(self) -> List[str]:
        """Return available medication classes"""
        return [
            "SABA (Short-Acting Beta Agonists)",
            "ICS (Inhaled Corticosteroids)",
            "ICS-LABA Combinations",
            "LAMA Medications",
            "LAMA-LABA Combinations",
            "Triple Therapy (ICS-LABA-LAMA)"
        ]

def main():
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory. Please add your PDF formulary documents to this directory.")
        return

    # Initialize the agent (loads PDFs, builds Pinecone RAG DB)
    agent = FormularyAgent()
    print("\nRAG database created and ready!")

    # Example usage
    print("Available medication classes:")
    for i, med_class in enumerate(agent.get_medication_classes(), 1):
        print(f"{i}. {med_class}")
    
    # Interactive query interface
    try:
        print("\n=== Formulary Query Interface ===\n")
        print("Enter information to query formulary coverage (or press Ctrl+C to exit):\n")
        
        insurance_name = input("Insurance name (e.g., UnitedHealthcare, Aetna): ")
        
        print("\nSelect medication class (enter number):")
        for i, med_class in enumerate(agent.get_medication_classes(), 1):
            print(f"{i}. {med_class}")
        class_choice = int(input("\nChoice: "))
        medication_class = agent.get_medication_classes()[class_choice-1]
        
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
        response = agent.process_query(
            insurance_name=insurance_name,
            medication_class=medication_class,
            brand_preference=brand_preference,
            patient_age=patient_age,
            patient_conditions=patient_conditions
        )
        
        print("\n=== Recommendation ===\n")
        print(response)
        
    except KeyboardInterrupt:
        print("\nExiting query interface.")
    except Exception as e:
        print(f"\nError: {e}")
    for cls in agent.get_medication_classes():
        print(f"- {cls}")
    
    # Example query
    _ = agent.process_query(
        insurance_name="Example Insurance",
        medication_class="ICS-LABA Combinations",
        brand_preference="generic"
    )

if __name__ == "__main__":
    main()
