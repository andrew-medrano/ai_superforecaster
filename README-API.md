# AI Superforecaster API

This is a FastAPI-based API for the AI Superforecaster system. It allows integrating the forecasting engine with a React frontend or other clients.

## Setup

1. Install API dependencies:
   ```
   pip install -r api-requirements.txt
   ```

2. Run the API server:
   ```
   python api_server.py
   ```

3. API will be available at http://localhost:8000

## API Endpoints

### Start a Forecast
```
POST /forecast
Content-Type: application/json

{
  "question": "Will Bitcoin exceed $100,000 by the end of 2025?"
}
```

Response:
```json
{
  "session_id": "f7e5e3d1-9c8b-4a3e-8d2f-1a2b3c4d5e6f",
  "status": "running"
}
```

### Check Forecast Status
```
GET /forecast/{session_id}
```

Response:
```json
{
  "session_id": "f7e5e3d1-9c8b-4a3e-8d2f-1a2b3c4d5e6f",
  "status": "completed",
  "result": {
    "question": "Will Bitcoin exceed $100,000 by the end of 2025?",
    "final_estimate": 0.65,
    "final_low": 0.45,
    "final_high": 0.85,
    "rationale": "Based on the analysis..."
  },
  "buffers": {
    "user": "...",
    "background": "...",
    "parameters": "...",
    "report": "..."
  }
}
```

### Get Buffer Contents
```
GET /forecast/{session_id}/buffers
```

Response:
```json
{
  "content": {
    "user": "...",
    "background": "...",
    "parameters": "...",
    "report": "..."
  }
}
```

## Deployment

### Vercel
1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel` in project root

### Railway
1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway up` 