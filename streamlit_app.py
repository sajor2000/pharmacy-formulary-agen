#!/usr/bin/env python3
"""
Pharmacy Formulary AI Assistant
-------------------------------
A chat-like interface for healthcare providers to query medication coverage
across different insurance formularies with a focus on respiratory medications.
"""

import os
import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
from inhaler_recommender import InhalerRecommender

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Pharmacy Formulary AI Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize recommender
@st.cache_resource
def get_recommender():
    return InhalerRecommender()

recommender = get_recommender()

# Custom CSS for chat-like interface
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    border: 1px solid rgba(112, 128, 144, 0.1);
}
.chat-message.user {
    background-color: #f0f2f6;
}
.chat-message.assistant {
    background-color: #e3f2fd;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding-left: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    
    st.title("Pharmacy Formulary AI")
    st.markdown("---")
    
    st.markdown("### How to Use This Assistant")
    
    st.markdown("""
    This AI assistant helps healthcare providers find the best medication options based on insurance formulary coverage, with a focus on respiratory medications.
    
    **The assistant will always prioritize lowest-tier medications** to reduce patient costs while ensuring appropriate treatment.
    
    You can interact with the assistant in two ways:
    
    **1. Chat Interface:**
    
    Simply type your question about medication coverage, such as:
    - *"What tier is Advair on Blue Cross Blue Shield?"*
    - *"What's the lowest tier rescue inhaler for UnitedHealthcare?"*
    - *"Does Cigna require prior authorization for Symbicort?"*
    - *"Compare Breo and Trelegy coverage on Express Scripts."*
    
    **2. Structured Search:**
    
    Use the form to specify:
    - Insurance provider
    - Medication class needed
    - Patient details (optional)
    - Device preferences (optional)
    
    **Understanding Medication Tiers:**
    
    - **Tier 1**: Lowest cost, usually generics (BEST VALUE)
    - **Tier 2**: Medium cost, preferred brands
    - **Tier 3**: Higher cost, non-preferred brands
    - **Tier 4+**: Highest cost, specialty medications
    
    The assistant will always recommend the lowest tier option when available.
    """)
    
    st.markdown("---")
    
    # Add tabs in sidebar for switching between chat and structured search
    tab_selection = st.radio("Select Interface:", ["Chat", "Structured Search"])

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    
    # Start with a welcome message from the assistant
    st.session_state.messages = [
        
        {"role": "assistant", "content": "Hello! I'm your Pharmacy Formulary AI Assistant. I can help you find the best medication options based on insurance coverage, with a focus on respiratory medications. I'll always prioritize lowest-tier medications to reduce patient costs. How can I help you today?"}
    
    ]

# Main interface based on selected tab
if tab_selection == "Chat":
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask about medication coverage..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Use the processor to get embedding and query Pinecone
                from document_processor import DocumentProcessor
                processor = DocumentProcessor()
                
                # Get embedding for the query
                query_embedding = processor.get_embedding(prompt)
                
                # Query Pinecone
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                index = pc.Index("form")
                
                results = index.query(
                    namespace="formulary",
                    vector=query_embedding,
                    top_k=5,
                    include_values=False,
                    include_metadata=True
                )
                
                # Format context from results
                context = ""
                for match in results.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata = match.metadata
                        context += f"Source: {metadata.get('source', 'Unknown')}\n"
                        if 'content' in metadata:
                            context += f"Content: {metadata.get('content')}\n\n"
                
                # Generate response with GPT-4o
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a pharmacy formulary specialist who helps healthcare providers find medication information based on insurance coverage. You ALWAYS prioritize medications with the lowest tier (Tier 1 if available) in your recommendations, unless there are compelling clinical reasons to choose a higher tier option. Be concise but thorough in your answers, focusing on practical information that helps clinicians make cost-effective prescribing decisions."},
                        {"role": "user", "content": f"Question: {prompt}\n\nRelevant formulary information:\n{context}"}
                    ],
                    stream=True
                )
                
                # Display streaming response
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                message_placeholder.markdown(error_message)
                full_response = error_message
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

else:  # Structured Search
    st.header("Get Inhaler Recommendation")
    
    # Create columns for form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Required Information")
        
        # Insurance provider selection
        providers = list(recommender.insurance_formularies.keys())
        insurance_provider = st.selectbox(
            "Insurance Provider",
            options=providers,
            index=None,
            placeholder="Select insurance provider"
        )
        
        # Medication class selection
        med_classes = recommender.medication_classes
        med_class_options = list(med_classes.values())
        medication_class = st.selectbox(
            "Medication Class Needed",
            options=med_class_options,
            index=None,
            placeholder="Select medication class"
        )
        
        # Show examples for selected medication class
        if medication_class and medication_class in recommender.medication_examples:
            examples = recommender.medication_examples[medication_class]
            st.caption(f"Examples: {', '.join(examples)}")
    
    with col2:
        st.subheader("Optional Clinical Details")
        
        # Patient age
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=None,
            placeholder="Leave blank if not relevant"
        )
        
        # Patient conditions
        patient_conditions = st.text_input(
            "Patient Conditions",
            placeholder="e.g., COPD, asthma, allergies (comma separated)"
        )
        
        # Device preference
        device_options = ["No preference", "MDI (Metered Dose Inhaler)", "DPI (Dry Powder Inhaler)", 
                          "Respimat (Soft Mist Inhaler)", "Nebulizer"]
        device_preference = st.selectbox(
            "Device Preference",
            options=device_options,
            index=0
        )
        device_preference = None if device_preference == "No preference" else device_preference.split(" ")[0]
        
        # Brand preference
        brand_options = ["Generic Preferred", "Brand Preferred", "No Preference"]
        brand_map = {
            "Generic Preferred": "generic preferred",
            "Brand Preferred": "brand preferred",
            "No Preference": "no preference"
        }
        brand_selection = st.selectbox(
            "Brand Preference",
            options=brand_options,
            index=0
        )
        brand_preference = brand_map[brand_selection]
    
    # Tier structure information
    with st.expander("Formulary Tier Information"):
        st.markdown("""
        **Tier 1**: Typically includes the lowest-cost generic drugs, often with the lowest copays.
        
        **Tier 2**: May include more expensive generic drugs, some preferred brand-name drugs, and non-preferred brand-name drugs.
        
        **Tier 3**: Often contains preferred brand-name drugs, with some generic drugs for which there are lower-cost or over-the-counter alternatives.
        
        **Tier 4**: May include non-preferred brand-name drugs and some higher-cost generic drugs, as well as preferred specialty drugs.
        
        **Tier 5**: Typically reserved for specialty drugs, which are expensive medications used to treat specific conditions.
        
        **Tier 6**: Some plans may have a Tier 6 for select medications with lower copays, often including generics and some brand-name drugs for chronic conditions.
        """)
    
    # Submit button
    submit_button = st.button("Get Recommendation", type="primary", use_container_width=True)
    
    if submit_button:
        if not insurance_provider or not medication_class:
            st.error("Please select both insurance provider and medication class")
        else:
            with st.spinner("Searching formulary database..."):
                # Process patient conditions
                conditions_list = None
                if patient_conditions:
                    conditions_list = [c.strip() for c in patient_conditions.split(",") if c.strip()]
                
                # Get recommendation
                recommendation = recommender.get_inhaler_recommendation(
                    insurance_provider=insurance_provider,
                    medication_class=medication_class,
                    patient_age=patient_age,
                    patient_conditions=conditions_list,
                    device_preference=device_preference,
                    brand_preference=brand_preference
                )
                
                # Display recommendation
                st.subheader("Recommendation")
                st.markdown(recommendation)

with tab2:
    st.header("Ask a Direct Question")
    
    # Example questions
    st.markdown("""
    **Example questions you can ask:**
    - What tier is Advair on Blue Cross Blue Shield?
    - Does Cigna require prior authorization for Symbicort?
    - What are the lowest tier options for asthma on Express Scripts?
    - Compare Breo and Trelegy coverage on UnitedHealthcare
    """)
    
    # Question input
    query = st.text_area(
        "Your Question",
        placeholder="e.g., What inhalers are covered by UnitedHealthcare for COPD?",
        height=100
    )
    
    # Submit button
    ask_button = st.button("Submit Question", type="primary", use_container_width=True)
    
    if ask_button:
        if not query:
            st.error("Please enter a question")
        else:
            with st.spinner("Searching formulary database..."):
                try:
                    # Use the processor to get embedding and query Pinecone
                    from document_processor import DocumentProcessor
                    processor = DocumentProcessor()
                    
                    # Get embedding for the query
                    query_embedding = processor.get_embedding(query)
                    
                    # Query Pinecone
                    from pinecone import Pinecone
                    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                    index = pc.Index("form")
                    
                    results = index.query(
                        namespace="formulary",
                        vector=query_embedding,
                        top_k=5,
                        include_values=False,
                        include_metadata=True
                    )
                    
                    # Format context from results
                    context = ""
                    for match in results.matches:
                        if hasattr(match, 'metadata') and match.metadata:
                            metadata = match.metadata
                            context += f"Source: {metadata.get('source', 'Unknown')}\n"
                            if 'content' in metadata:
                                context += f"Content: {metadata.get('content')}\n\n"
                    
                    # Generate response with GPT-4o
                    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a pharmacy formulary specialist who helps healthcare providers find medication information based on insurance coverage. You ALWAYS prioritize medications with the lowest tier (Tier 1 if available) in your recommendations, unless there are compelling clinical reasons to choose a higher tier option."},
                            {"role": "user", "content": f"Question: {query}\n\nRelevant formulary information:\n{context}"}
                        ]
                    )
                    
                    # Display response
                    st.subheader("Answer")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Add footer
st.markdown("---")
st.caption("Pharmacy Formulary Assistant | Powered by AI")

# Add deployment instructions in a hidden section for admins
with st.expander("Deployment Information (For Administrators)"):
    st.markdown("""
    ### Deployment Instructions
    
    This application can be easily deployed to Streamlit Cloud for access from anywhere:
    
    1. Create a GitHub repository with this code
    2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
    3. Connect your GitHub repository
    4. Set the following secrets in Streamlit Cloud:
       - OPENAI_API_KEY
       - PINECONE_API_KEY
    
    Alternatively, you can deploy to Heroku, AWS, or other cloud platforms.
    """)
