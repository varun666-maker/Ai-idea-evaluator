# AI Idea Evaluator

FastAPI application that evaluates business ideas using Google Gemini AI.

## Features
- SWOT Analysis
- Competitor Analysis (using Google Search)
- Market Analysis (TAM, SAM, SOM)
- Market Size & Feasibility Scoring
- Overall Viability Score with Verdict

## Deploy to Render

### Step 1: Prepare Your Repository
1. Push your code to GitHub/GitLab/Bitbucket
2. Make sure these files are included:
   - `main.py`
   - `requirements.txt`
   - `Procfile` or `render.yaml`
   - `.gitignore` (to exclude `.env` file)

### Step 2: Create Render Account
1. Go to https://render.com
2. Sign up or log in

### Step 3: Create New Web Service
1. Click "New +" → "Web Service"
2. Connect your Git repository
3. Select your repository

### Step 4: Configure Service
- **Name**: `ai-idea-evaluator` (or your choice)
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Step 5: Add Environment Variable
1. Scroll to "Environment Variables"
2. Click "Add Environment Variable"
3. Add:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your Gemini API key (AIzaSyCC7LrXTqxyRtkhLZfbYZA-Sbmku5QJaT0)

### Step 6: Deploy
1. Click "Create Web Service"
2. Wait 5-10 minutes for deployment
3. Your API will be live at: `https://your-service-name.onrender.com`

## Test Your Deployment

### Health Check
```bash
curl https://your-service-name.onrender.com/health
```

### Evaluate an Idea
```bash
curl -X POST https://your-service-name.onrender.com/evaluate-idea \
  -H "Content-Type: application/json" \
  -d '{
    "idea": "AI-powered fitness app",
    "location": "USA"
  }'
```

## API Endpoints

- `GET /health` - Health check
- `POST /evaluate-idea` - Evaluate business idea
  - Body: `{"idea": "string", "location": "string"}`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000
```

## Important Notes

- Free tier on Render may spin down after inactivity (takes 30-60s to wake up)
- First request after inactivity will be slow
- Upgrade to paid plan for always-on service
- Keep your API key secure (use environment variables).
