# Streamlit Deployment Guide

This guide will help you deploy your Pharmacy Formulary AI Assistant to Streamlit Cloud so nurses can access it from anywhere.

## Prerequisites

1. Your code is pushed to GitHub (follow SETUP_GITHUB.md)
2. You have a Streamlit account (streamlit.io)
3. You have your API keys ready:
   - OPENAI_API_KEY
   - PINECONE_API_KEY

## Deployment Steps

### 1. Sign in to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your account credentials

### 2. Create a New App

1. Click "New app" button
2. Connect to your GitHub repository:
   - Repository: `sajor2000/pharmacy-formulary-agen`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

### 3. Configure Environment Variables

1. Click "Advanced settings" before deploying
2. Under "Secrets", add the following in TOML format:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
PINECONE_API_KEY = "your_pinecone_api_key_here"
```

3. Click "Save"

### 4. Deploy Your App

1. Click "Deploy" 
2. Wait for the deployment to complete (this may take a few minutes)
3. Your app will be available at: `https://sajor2000-pharmacy-formulary-agen.streamlit.app`

### 5. Share with Nurses

1. Copy your Streamlit app URL
2. Share with healthcare providers who need access
3. No login required - they can access from any device with internet

## Updating Your App

Whenever you push changes to your GitHub repository, Streamlit Cloud will automatically redeploy your app with the latest changes.

## Monitoring Usage

1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. View metrics like visitors, run time, and errors

## Troubleshooting

If your app fails to deploy:

1. Check the deployment logs in Streamlit Cloud
2. Verify your environment variables are set correctly
3. Make sure all dependencies are in requirements.txt
4. Test locally with `streamlit run streamlit_app.py` before deploying
