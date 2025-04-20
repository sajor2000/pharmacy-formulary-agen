#!/usr/bin/env python3
"""
Pharmacy Formulary Web Interface
--------------------------------
A simple web interface for healthcare providers to interact with the
formulary recommendation system without any coding knowledge.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from inhaler_recommender import InhalerRecommender

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
recommender = InhalerRecommender()

@app.route('/')
def index():
    """Render the main page"""
    # Get list of insurance providers
    providers = list(recommender.insurance_formularies.keys())
    
    # Get medication classes
    med_classes = recommender.medication_classes
    
    # Get medication examples
    med_examples = recommender.medication_examples
    
    return render_template('index.html', 
                          providers=providers,
                          med_classes=med_classes,
                          med_examples=med_examples)

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    """Get a recommendation based on form data"""
    try:
        # Get form data
        data = request.json
        
        insurance_provider = data.get('insurance_provider')
        medication_class = data.get('medication_class')
        patient_age = data.get('patient_age')
        patient_conditions = data.get('patient_conditions', '').split(',')
        patient_conditions = [c.strip() for c in patient_conditions if c.strip()]
        device_preference = data.get('device_preference')
        brand_preference = data.get('brand_preference', 'generic preferred')
        
        # Convert patient age to int if provided
        if patient_age and patient_age.isdigit():
            patient_age = int(patient_age)
        else:
            patient_age = None
            
        # Get recommendation
        recommendation = recommender.get_inhaler_recommendation(
            insurance_provider=insurance_provider,
            medication_class=medication_class,
            patient_age=patient_age,
            patient_conditions=patient_conditions if patient_conditions else None,
            device_preference=device_preference if device_preference else None,
            brand_preference=brand_preference
        )
        
        return jsonify({'recommendation': recommendation})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/direct_query', methods=['POST'])
def direct_query():
    """Handle direct natural language queries"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'})
        
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
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a pharmacy formulary specialist who helps healthcare providers find medication information based on insurance coverage. You ALWAYS prioritize medications with the lowest tier (Tier 1 if available) in your recommendations, unless there are compelling clinical reasons to choose a higher tier option."},
                {"role": "user", "content": f"Question: {query}\n\nRelevant formulary information:\n{context}"}
            ]
        )
        
        return jsonify({'response': response.choices[0].message.content})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pharmacy Formulary Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 900px;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        h1 {
            color: #0d6efd;
            margin-bottom: 20px;
        }
        .card {
            margin-bottom: 20px;
            border: none;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .card-header {
            background-color: #e7f1ff;
            font-weight: bold;
            color: #0d6efd;
        }
        .btn-primary {
            background-color: #0d6efd;
            border: none;
        }
        .btn-primary:hover {
            background-color: #0b5ed7;
        }
        .form-label {
            font-weight: 500;
        }
        #recommendation {
            white-space: pre-line;
            line-height: 1.5;
        }
        .tab-content {
            padding: 20px;
            background-color: white;
            border-left: 1px solid #dee2e6;
            border-right: 1px solid #dee2e6;
            border-bottom: 1px solid #dee2e6;
            border-radius: 0 0 5px 5px;
        }
        .nav-tabs .nav-link {
            font-weight: 500;
        }
        .nav-tabs .nav-link.active {
            background-color: white;
            border-bottom: 1px solid white;
        }
        .example-text {
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center">Pharmacy Formulary Assistant</h1>
        <p class="text-center mb-4">Find the best medication options based on insurance coverage and patient needs</p>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="structured-tab" data-bs-toggle="tab" data-bs-target="#structured" type="button" role="tab" aria-controls="structured" aria-selected="true">Structured Search</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="direct-tab" data-bs-toggle="tab" data-bs-target="#direct" type="button" role="tab" aria-controls="direct" aria-selected="false">Ask a Question</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Structured Search Tab -->
            <div class="tab-pane fade show active" id="structured" role="tabpanel" aria-labelledby="structured-tab">
                <form id="recommendation-form">
                    <div class="card mb-3">
                        <div class="card-header">Required Information</div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="insurance" class="form-label">Insurance Provider</label>
                                <select class="form-select" id="insurance" required>
                                    <option value="" selected disabled>Select insurance provider</option>
                                    {% for provider in providers %}
                                    <option value="{{ provider }}">{{ provider }}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="medication-class" class="form-label">Medication Class Needed</label>
                                <select class="form-select" id="medication-class" required>
                                    <option value="" selected disabled>Select medication class</option>
                                    {% for key, value in med_classes.items() %}
                                    <option value="{{ value }}">{{ value }}</option>
                                    {% endfor %}
                                </select>
                                <div id="medication-examples" class="example-text"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card mb-3">
                        <div class="card-header">Optional Clinical Details</div>
                        <div class="card-body">
                            <div class="mb-3">
                                <label for="patient-age" class="form-label">Patient Age</label>
                                <input type="number" class="form-control" id="patient-age" placeholder="Leave blank if not relevant">
                            </div>
                            
                            <div class="mb-3">
                                <label for="patient-conditions" class="form-label">Patient Conditions</label>
                                <input type="text" class="form-control" id="patient-conditions" placeholder="e.g., COPD, asthma, allergies (comma separated)">
                            </div>
                            
                            <div class="mb-3">
                                <label for="device-preference" class="form-label">Device Preference</label>
                                <select class="form-select" id="device-preference">
                                    <option value="" selected>No preference</option>
                                    <option value="MDI">MDI (Metered Dose Inhaler)</option>
                                    <option value="DPI">DPI (Dry Powder Inhaler)</option>
                                    <option value="Respimat">Respimat (Soft Mist Inhaler)</option>
                                    <option value="Nebulizer">Nebulizer</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label for="brand-preference" class="form-label">Brand Preference</label>
                                <select class="form-select" id="brand-preference">
                                    <option value="generic preferred" selected>Generic Preferred</option>
                                    <option value="brand preferred">Brand Preferred</option>
                                    <option value="no preference">No Preference</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button type="submit" class="btn btn-primary btn-lg">Get Recommendation</button>
                    </div>
                </form>
            </div>
            
            <!-- Direct Question Tab -->
            <div class="tab-pane fade" id="direct" role="tabpanel" aria-labelledby="direct-tab">
                <div class="card mb-3">
                    <div class="card-header">Ask a Question</div>
                    <div class="card-body">
                        <form id="direct-query-form">
                            <div class="mb-3">
                                <label for="query" class="form-label">Your Question</label>
                                <textarea class="form-control" id="query" rows="3" placeholder="e.g., What inhalers are covered by UnitedHealthcare for COPD?" required></textarea>
                                <div class="example-text mt-2">
                                    Example questions:
                                    <ul>
                                        <li>What tier is Advair on Blue Cross Blue Shield?</li>
                                        <li>Does Cigna require prior authorization for Symbicort?</li>
                                        <li>What are the lowest tier options for asthma on Express Scripts?</li>
                                    </ul>
                                </div>
                            </div>
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary btn-lg">Submit Question</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Loading Spinner -->
        <div class="loading" id="loading">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Searching formulary database...</p>
        </div>
        
        <!-- Recommendation Results -->
        <div class="card mt-4" id="result-card" style="display: none;">
            <div class="card-header">Recommendation</div>
            <div class="card-body">
                <div id="recommendation"></div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Medication examples
        const medicationExamples = {{ med_examples|tojson }};
        
        // Show examples when medication class is selected
        document.getElementById('medication-class').addEventListener('change', function() {
            const selectedClass = this.value;
            const examplesDiv = document.getElementById('medication-examples');
            
            if (medicationExamples[selectedClass]) {
                examplesDiv.innerHTML = `Examples: ${medicationExamples[selectedClass].join(', ')}`;
            } else {
                examplesDiv.innerHTML = '';
            }
        });
        
        // Handle structured recommendation form submission
        document.getElementById('recommendation-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading spinner
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';
            
            // Get form data
            const data = {
                insurance_provider: document.getElementById('insurance').value,
                medication_class: document.getElementById('medication-class').value,
                patient_age: document.getElementById('patient-age').value,
                patient_conditions: document.getElementById('patient-conditions').value,
                device_preference: document.getElementById('device-preference').value,
                brand_preference: document.getElementById('brand-preference').value
            };
            
            // Send request
            fetch('/get_recommendation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                // Show results
                document.getElementById('result-card').style.display = 'block';
                
                if (data.error) {
                    document.getElementById('recommendation').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    document.getElementById('recommendation').innerHTML = data.recommendation;
                }
            })
            .catch(error => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                // Show error
                document.getElementById('result-card').style.display = 'block';
                document.getElementById('recommendation').innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
        
        // Handle direct query form submission
        document.getElementById('direct-query-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading spinner
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result-card').style.display = 'none';
            
            // Get query
            const query = document.getElementById('query').value;
            
            // Send request
            fetch('/direct_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                // Show results
                document.getElementById('result-card').style.display = 'block';
                
                if (data.error) {
                    document.getElementById('recommendation').innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    document.getElementById('recommendation').innerHTML = data.response;
                }
            })
            .catch(error => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                // Show error
                document.getElementById('result-card').style.display = 'block';
                document.getElementById('recommendation').innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
        });
    </script>
</body>
</html>
        """)
    
    # Install Flask if not already installed
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        os.system("pip install flask")
    
    # Run the app
    app.run(debug=True, port=5000)
