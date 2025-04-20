# GitHub Repository Setup Guide

Follow these steps to create your GitHub repository and push your code:

## 1. Create a New Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in with your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Set repository name: `pharmacy-formulary-agent`
4. Add a description: "AI-powered pharmacy formulary assistant for respiratory medications"
5. Choose "Public" visibility (or Private if you prefer)
6. Click "Create repository"

## 2. Initialize Git and Push Your Code

Open a terminal and run these commands:

```bash
# Navigate to your project directory
cd "/Users/JCR/Desktop/Windsurf IDE/ai form agent"

# Initialize git repository
git init

# Add all files (except those in .gitignore)
git add .

# Commit your changes
git commit -m "Initial commit of pharmacy formulary agent"

# Add your GitHub repository as remote
git remote add origin https://github.com/sajor2000/pharmacy-formulary-agent.git

# Push to GitHub
git push -u origin main
```

You may be prompted to enter your GitHub credentials.

## 3. Verify Your Repository

After pushing, visit https://github.com/sajor2000/pharmacy-formulary-agent to verify your code is now on GitHub.
