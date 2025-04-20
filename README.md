# Pharmacy Formulary AI Assistant

This AI assistant helps healthcare providers match respiratory medications to insurance formulary preferences. It prioritizes lowest-tier medications to reduce patient costs while ensuring appropriate treatment options.

## Features

- **Formulary Document Processing**: Extracts text and tables from PDF formulary documents
- **Tier-Based Recommendations**: Always prioritizes lowest tier medications (Tier 1 when available)
- **Multiple Interfaces**:
  - Command-line query interface
  - Web application (Flask)
  - Streamlit cloud deployment
- **Comprehensive Medication Coverage**:
  - Rescue inhalers (SABA)
  - Maintenance inhalers (ICS, LABA, LAMA)
  - Combination therapies (ICS-LABA, LAMA-LABA, Triple therapy)

## Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your PDF formulary documents to the `data` directory

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

5. Run the assistant locally:
```bash
python inhaler_recommender.py  # Interactive CLI
# OR
python app.py  # Web interface
# OR
streamlit run streamlit_app.py  # Streamlit interface
```

## Streamlit Cloud Deployment

This application is designed for easy deployment to Streamlit Cloud:

1. Fork or clone this repository to your GitHub account
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app by connecting to your GitHub repository
4. Set the main file path to `streamlit_app.py`
5. Add the following secrets in the Streamlit dashboard:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`

## Contact

- **Developer**: Juan C. Rojas
- **Email**: juancroj@gmail.com
- **GitHub**: [github.com/sajor2000](https://github.com/sajor2000)

- Processes PDF formulary documents
- Maintains knowledge of inhaler classes and formulations
- Matches medications based on insurance preferences
- Provides structured recommendations with coverage details
