import asyncio
import uvicorn
import uuid
import time
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any

# Import your existing forecasting machinery
from src.forecasting_engine import run_full_pipeline
from src.utils.buffers import BufferManager
from src.utils.buffer_config import get_buffer_names
from src.ui.cli import init_buffers  # Import the init_buffers function

# Create FastAPI instance
app = FastAPI(title="AI Superforecaster API")

# Add CORS middleware to allow React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions store
sessions = {}

# API Models
class ForecastRequest(BaseModel):
    question: str = Field(description="The forecasting question")
    
    # Validate that the question is not empty
    @validator('question')
    def question_must_not_be_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Question cannot be empty")
        return v

class SessionResponse(BaseModel):
    session_id: str = Field(description="Unique session identifier")
    status: str = Field(description="Current status of the forecast session")

class BufferContents(BaseModel):
    content: Dict[str, str] = Field(description="Contents of each buffer section")

class ForecastResponse(BaseModel):
    session_id: str = Field(description="Unique session identifier")
    status: str = Field(description="Status of the forecast: running, completed, error")
    result: Optional[Dict[str, Any]] = Field(description="Final forecast results when completed")
    buffers: Optional[Dict[str, str]] = Field(description="Current buffer contents")
    error: Optional[str] = Field(description="Error message if status is 'error'")

# Simple input provider for non-interactive API use
class ApiInputProvider:
    def get_input(self, prompt):
        # For API use, we don't have interactive prompts
        # Always return empty to use default assumptions
        return ""

# Endpoint to start a new forecast
@app.post("/forecast", response_model=SessionResponse)
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """Start a new forecast session"""
    session_id = str(uuid.uuid4())
    
    # Create buffer manager for this session
    buffer_manager = BufferManager(echo_user=False)
    # Initialize buffers - this is crucial for the forecasting engine
    init_buffers(buffer_manager)
    
    # Store session info
    sessions[session_id] = {
        "status": "running",
        "buffer_manager": buffer_manager,
        "result": None,
        "error": None,
    }
    
    # Run forecast in background to avoid blocking
    background_tasks.add_task(
        run_forecast_background,
        session_id,
        request.question,
        buffer_manager
    )
    
    return {"session_id": session_id, "status": "running"}

# Background task to run the forecast
async def run_forecast_background(session_id: str, question: str, buffer_manager: BufferManager):
    """Run the forecast in the background"""
    start_time = time.time()
    try:
        # Record the start of forecast processing
        buffer_manager.write("user", f"Starting forecast processing at {time.strftime('%H:%M:%S')}")
        
        # Non-interactive input provider
        input_provider = ApiInputProvider()
        
        # Run the forecasting pipeline using existing code
        result = await run_full_pipeline(question, buffer_manager, input_provider)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        buffer_manager.write("user", f"Forecast processing completed in {elapsed_time:.2f} seconds")
        
        # Update session with results
        if session_id in sessions:
            sessions[session_id]["status"] = "completed"
            if result:
                # Convert Pydantic model to dict for JSON serialization
                sessions[session_id]["result"] = result.model_dump()
            else:
                sessions[session_id]["status"] = "error"
                sessions[session_id]["error"] = "Forecast failed to produce a result"
    
    except Exception as e:
        # Calculate elapsed time even for errors
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        
        # Handle errors
        if session_id in sessions:
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = error_msg
            
            # Log the error to the buffer
            buffer_manager.write("user", f"Error after {elapsed_time:.2f} seconds: {error_msg}")

# Endpoint to check the status of a forecast
@app.get("/forecast/{session_id}", response_model=ForecastResponse)
async def get_forecast(session_id: str):
    """Get the current status and results of a forecast session"""
    if session_id not in sessions:
        return {
            "session_id": session_id,
            "status": "not_found",
            "result": None,
            "buffers": None,
            "error": None
        }
    
    session = sessions[session_id]
    buffer_manager = session["buffer_manager"]
    
    # Get current buffer contents
    buffers = {}
    for section in buffer_manager.sections:
        buffers[section] = buffer_manager.dump(section)
    
    return {
        "session_id": session_id,
        "status": session["status"],
        "result": session["result"],
        "buffers": buffers,
        "error": session["error"]
    }

# Endpoint to get just the buffer contents
@app.get("/forecast/{session_id}/buffers", response_model=BufferContents)
async def get_buffer_contents(session_id: str):
    """Get just the current buffer contents for a forecast session"""
    if session_id not in sessions:
        return {"content": {}}
    
    session = sessions[session_id]
    buffer_manager = session["buffer_manager"]
    
    # Get current buffer contents
    buffers = {}
    for section in buffer_manager.sections:
        buffers[section] = buffer_manager.dump(section)
    
    return {"content": buffers}

# Start server when run directly
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 