# AI Superforecaster API Documentation

The AI Superforecaster provides a FastAPI-based server for programmatic access to forecasting capabilities.

## Starting the API Server

```bash
python api_server.py
```

By default, the server runs on `http://localhost:8000`.

## API Endpoints

### Create a Forecast

```
POST /forecast
```

Start a new forecast session.

**Request Body:**
```json
{
  "question": "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
}
```

**Response:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "running"
}
```

### Get Forecast Status and Results

```
GET /forecast/{session_id}
```

Retrieve the status and results of a forecast session.

**Response:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "result": {
    "forecast": 0.75,
    "low": 0.60,
    "high": 0.85,
    "rationale": "Based on analysis of market trends and expert opinions..."
  },
  "buffers": {
    "user": "Content of user buffer...",
    "background": "Content of background buffer...",
    "parameters": "Content of parameters buffer...",
    "report": "Content of report buffer..."
  },
  "error": null
}
```

The `status` field can be one of:
- `running` - Forecast is still in progress
- `completed` - Forecast has completed successfully
- `error` - An error occurred during forecasting
- `not_found` - The specified session ID does not exist

### Get Buffer Contents

```
GET /forecast/{session_id}/buffers
```

Retrieve just the buffer contents for a forecast session.

**Response:**
```json
{
  "content": {
    "user": "Content of user buffer...",
    "background": "Content of background buffer...",
    "parameters": "Content of parameters buffer...",
    "report": "Content of report buffer..."
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

- `200 OK` - The request was successful
- `422 Unprocessable Entity` - Invalid request parameters
- `500 Internal Server Error` - Server-side error

For `422` errors, the response will include validation details.

## Example Usage

### Starting a Forecast

```bash
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"}'
```

### Checking Status

```bash
curl -X GET "http://localhost:8000/forecast/123e4567-e89b-12d3-a456-426614174000"
```

### Getting Buffer Contents

```bash
curl -X GET "http://localhost:8000/forecast/123e4567-e89b-12d3-a456-426614174000/buffers"
``` 