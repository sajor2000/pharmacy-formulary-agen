#!/usr/bin/env python3
"""
Inhaler Recommendation System
----------------------------
A specialized interface for healthcare providers to get inhaler recommendations
based on insurance formularies and patient needs.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import openai
import pandas as pd
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from formulary_agent import FormularyAgent, FormularyResponse

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class InhalerRecommender:
    def __init__(self):
        """Initialize the inhaler recommender with document processor and formulary agent"""
        self.processor = DocumentProcessor()
        self.agent = FormularyAgent()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Define medication classes
        self.medication_classes = {
            "1": "SABA (Short-Acting Beta Agonists)",
            "2": "ICS (Inhaled Corticosteroids)",
            "3": "LABA (Long-Acting Beta Agonists)",
            "4": "LAMA (Long-Acting Muscarinic Antagonists)",
            "5": "ICS-LABA Combinations",
            "6": "LAMA-LABA Combinations",
            "7": "Triple Therapy (ICS-LABA-LAMA)"
        }
        
        # Map medication classes to common examples
        self.medication_examples = {
            "SABA (Short-Acting Beta Agonists)": ["albuterol (ProAir, Ventolin, Proventil)", "levalbuterol (Xopenex)"],
            "ICS (Inhaled Corticosteroids)": ["fluticasone (Flovent)", "budesonide (Pulmicort)", "beclomethasone (QVAR)", "mometasone (Asmanex)"],
            "LABA (Long-Acting Beta Agonists)": ["salmeterol (Serevent)", "formoterol (Foradil)", "indacaterol (Arcapta)"],
            "LAMA (Long-Acting Muscarinic Antagonists)": ["tiotropium (Spiriva)", "umeclidinium (Incruse)", "aclidinium (Tudorza)", "glycopyrrolate (Seebri)"],
            "ICS-LABA Combinations": ["fluticasone-salmeterol (Advair)", "budesonide-formoterol (Symbicort)", "mometasone-formoterol (Dulera)", "fluticasone-vilanterol (Breo)"],
            "LAMA-LABA Combinations": ["umeclidinium-vilanterol (Anoro)", "tiotropium-olodaterol (Stiolto)", "glycopyrrolate-formoterol (Bevespi)"],
            "Triple Therapy (ICS-LABA-LAMA)": ["fluticasone-umeclidinium-vilanterol (Trelegy)", "budesonide-glycopyrrolate-formoterol (Breztri)"]
        }
        
        # Map insurance providers to their formulary files
        self.insurance_formularies = {
            "UnitedHealthcare": [f for f in os.listdir("data") if f.startswith("4-25 UHC")],
            "Blue Cross Blue Shield": [f for f in os.listdir("data") if f.startswith("4-25 BCBS")],
            "Cigna": [f for f in os.listdir("data") if f.startswith("4-25 Cigna")],
            "Express Scripts": [f for f in os.listdir("data") if f.startswith("4-25 Express")],
            "Humana": [f for f in os.listdir("data") if f.startswith("4-25 Humana")],
            "CountyCare": [f for f in os.listdir("data") if f.startswith("4-25 County")],
            "Meridian": [f for f in os.listdir("data") if f.startswith("4-25 Meridian")],
            "Wellcare": [f for f in os.listdir("data") if f.startswith("4-25 Wellcare")]
        }
    
    def get_inhaler_recommendation(self, 
                                  insurance_provider: str,
                                  medication_class: str,
                                  patient_age: Optional[int] = None,
                                  patient_conditions: Optional[List[str]] = None,
                                  device_preference: Optional[str] = None,
                                  brand_preference: str = "generic preferred") -> str:
        """
        Get inhaler recommendations based on insurance formulary and patient needs
        
        Args:
            insurance_provider: Name of the insurance provider
            medication_class: Type of inhaler needed
            patient_age: Optional patient age
            patient_conditions: Optional list of patient conditions
            device_preference: Optional device preference (MDI, DPI, etc.)
            brand_preference: Brand preference (default: generic preferred)
            
        Returns:
            Structured recommendation with primary and alternative options
        """
        try:
            # Construct context for the query
            context = self._get_formulary_context(insurance_provider, medication_class)
            
            # Build the prompt
            prompt = f"""
            As a pharmacy formulary specialist, provide an inhaler recommendation based on the following:
            
            PATIENT NEEDS:
            - Insurance: {insurance_provider}
            - Medication Class Needed: {medication_class}
            """
            
            if patient_age:
                prompt += f"- Patient Age: {patient_age}\n"
            
            if patient_conditions:
                prompt += f"- Patient Conditions: {', '.join(patient_conditions)}\n"
            
            if device_preference:
                prompt += f"- Device Preference: {device_preference}\n"
            
            prompt += f"- Brand Preference: {brand_preference}\n\n"
            
            prompt += f"""
            FORMULARY CONTEXT:
            {context}
            
            Please provide a structured recommendation including:
            1. PRIMARY RECOMMENDATION: Provide the best option based on the formulary with details on:
               - Medication name (brand and generic)
               - Form/device type
               - Strength
               - Formulary tier
               - Prior authorization or step therapy requirements
               - Quantity limits
               - Estimated copay (if available)
            
            2. ALTERNATIVE OPTIONS: List 1-2 alternatives with key differences and requirements
            
            3. COVERAGE NOTES: Any important notes about coverage, restrictions, or special considerations
            
            Format your response in a clear, structured way that a healthcare provider can easily understand.
            """
            
            # Define tier structure information
            tier_structure = """
            FORMULARY TIER STRUCTURE:
            Tier 1: Typically includes the lowest-cost generic drugs, often with the lowest copays.
            Tier 2: May include more expensive generic drugs, some preferred brand-name drugs, and non-preferred brand-name drugs.
            Tier 3: Often contains preferred brand-name drugs, with some generic drugs for which there are lower-cost or over-the-counter alternatives.
            Tier 4: May include non-preferred brand-name drugs and some higher-cost generic drugs, as well as preferred specialty drugs.
            Tier 5: Typically reserved for specialty drugs, which are expensive medications used to treat specific conditions.
            Tier 6: Some plans may have a Tier 6 for select medications with lower copays, often including generics and some brand-name drugs for chronic conditions.
            """
            
            # Add tier structure to the prompt
            prompt += f"\n{tier_structure}\n"
            
            # Add explicit instruction to prioritize lowest tier medications
            prompt += "\nIMPORTANT: Always prioritize medications with the lowest tier (Tier 1 if available) as your primary recommendation. Only recommend higher tier medications if there are compelling clinical reasons or if lower tier options are not available for the required medication class.\n"
            
            # Generate recommendation using GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pharmacy formulary specialist who helps healthcare providers find the best inhaler options for their patients based on insurance coverage. You ALWAYS prioritize medications with the lowest tier (Tier 1 if available) as your primary recommendation, unless there are compelling clinical reasons to choose a higher tier option. You provide clear, structured recommendations with primary and alternative options."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating recommendation: {e}"
    
    def _get_formulary_context(self, insurance_provider: str, medication_class: str) -> str:
        """Retrieve relevant formulary context from Pinecone"""
        try:
            # Construct a query that combines insurance and medication class
            query = f"{insurance_provider} formulary coverage for {medication_class}"
            
            # Get embedding for the query
            query_embedding = self.processor.get_embedding(query)
            
            # Query Pinecone
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index("form")
            
            # Filter for the specific insurance provider if possible
            filter_dict = {}
            if insurance_provider:
                # We'll use a metadata filter if the insurance field is available
                # This is a soft filter since not all vectors might have this metadata
                filter_dict = {"insurance": {"$eq": insurance_provider}}
            
            results = index.query(
                namespace="formulary",
                vector=query_embedding,
                top_k=10,
                include_values=False,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format context from results
            context = ""
            for match in results.matches:
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    context += f"Source: {metadata.get('source', 'Unknown')}\n"
                    if 'type' in metadata:
                        context += f"Type: {metadata.get('type')}\n"
                    if 'page' in metadata:
                        context += f"Page: {metadata.get('page')}\n"
                    if 'content' in metadata:
                        context += f"Content: {metadata.get('content')}\n\n"
            
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No relevant formulary information found. Providing general recommendations based on common formulary patterns."

    def run_interactive_interface(self):
        """Run an interactive command-line interface for inhaler recommendations"""
        print("\n===== Respiratory Medication Formulary Assistant =====\n")
        print("This assistant helps healthcare providers find the best inhaler options")
        print("for their patients based on insurance formulary coverage.\n")
        
        while True:
            try:
                # Get insurance provider
                print("Available Insurance Providers:")
                providers = list(self.insurance_formularies.keys())
                for i, provider in enumerate(providers, 1):
                    print(f"{i}. {provider}")
                
                provider_choice = int(input("\nSelect insurance provider (number): "))
                if provider_choice < 1 or provider_choice > len(providers):
                    print("Invalid choice. Please try again.")
                    continue
                
                insurance_provider = providers[provider_choice-1]
                print(f"\nSelected: {insurance_provider}")
                
                # Get medication class
                print("\nMedication Class Needed:")
                for key, value in self.medication_classes.items():
                    print(f"{key}. {value}")
                    examples = self.medication_examples.get(value, [])
                    if examples:
                        print(f"   Examples: {', '.join(examples)}")
                
                class_choice = input("\nSelect medication class (number): ")
                if class_choice not in self.medication_classes:
                    print("Invalid choice. Please try again.")
                    continue
                
                medication_class = self.medication_classes[class_choice]
                print(f"\nSelected: {medication_class}")
                
                # Get optional clinical details
                print("\nWould you like to provide additional clinical details? (y/n)")
                if input().lower() == 'y':
                    patient_age = input("\nPatient age (leave blank if not relevant): ")
                    patient_age = int(patient_age) if patient_age.isdigit() else None
                    
                    conditions = input("\nPatient conditions (e.g., COPD, asthma, allergies - comma separated): ")
                    patient_conditions = [c.strip() for c in conditions.split(",")] if conditions else None
                    
                    device_preference = input("\nDevice preference (e.g., MDI, DPI, Respimat, or leave blank): ")
                    device_preference = device_preference if device_preference else None
                    
                    print("\nBrand preference:")
                    print("1. Generic preferred")
                    print("2. Brand preferred")
                    print("3. No preference")
                    brand_choice = input("\nSelect (1-3): ")
                    brand_map = {
                        "1": "generic preferred", 
                        "2": "brand preferred", 
                        "3": "no preference"
                    }
                    brand_preference = brand_map.get(brand_choice, "generic preferred")
                else:
                    patient_age = None
                    patient_conditions = None
                    device_preference = None
                    brand_preference = "generic preferred"
                
                # Generate recommendation
                print("\nGenerating recommendation based on formulary data...\n")
                recommendation = self.get_inhaler_recommendation(
                    insurance_provider=insurance_provider,
                    medication_class=medication_class,
                    patient_age=patient_age,
                    patient_conditions=patient_conditions,
                    device_preference=device_preference,
                    brand_preference=brand_preference
                )
                
                print("\n===== INHALER RECOMMENDATION =====\n")
                print(recommendation)
                print("\n===================================\n")
                
                # Ask if user wants another recommendation
                print("Would you like another recommendation? (y/n)")
                if input().lower() != 'y':
                    print("\nThank you for using the Respiratory Medication Formulary Assistant.\n")
                    break
                    
            except KeyboardInterrupt:
                print("\n\nExiting. Thank you for using the Respiratory Medication Formulary Assistant.\n")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Let's try again.\n")

if __name__ == "__main__":
    # Check if environment variables are set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set this in your .env file.")
        sys.exit(1)
    
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not found in environment variables.")
        print("Please set this in your .env file.")
        sys.exit(1)
    
    # Run the interface
    recommender = InhalerRecommender()
    recommender.run_interactive_interface()
