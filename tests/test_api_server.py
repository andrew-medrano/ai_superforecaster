import pytest
import time
from fastapi.testclient import TestClient
import uuid
from api_server import app, sessions
import asyncio

client = TestClient(app)

def test_forecast_full():
    """Single comprehensive test for the forecast API."""
    # Create a forecast
    create_response = client.post(
        "/forecast",
        json={"question": "Will AI improve in 2024?"}
    )
    assert create_response.status_code == 200
    assert "session_id" in create_response.json()
    assert create_response.json()["status"] == "running"
    
    # Verify session_id is a valid UUID
    session_id = create_response.json()["session_id"]
    try:
        uuid.UUID(session_id)
    except ValueError:
        pytest.fail("session_id is not a valid UUID")
    
    # Check status of forecast
    get_response = client.get(f"/forecast/{session_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["session_id"] == session_id
    assert data["status"] in ["running", "completed", "error"]
    
    # Check buffer contents
    buffer_response = client.get(f"/forecast/{session_id}/buffers")
    assert buffer_response.status_code == 200
    assert "content" in buffer_response.json()

def test_api_failures():
    """Test API error handling for various failure scenarios."""
    # Test with invalid session ID
    fake_id = str(uuid.uuid4())
    response = client.get(f"/forecast/{fake_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_found"
    
    # Test with invalid buffer request
    buffer_response = client.get(f"/forecast/{uuid.uuid4()}/buffers")
    assert buffer_response.status_code == 200
    assert buffer_response.json()["content"] == {}
    
    # Test input validation failures
    # Empty question
    empty_response = client.post(
        "/forecast",
        json={"question": ""}
    )
    assert empty_response.status_code == 422  # FastAPI validation error
    
    # Missing question field
    missing_response = client.post(
        "/forecast",
        json={}
    )
    assert missing_response.status_code == 422  # FastAPI validation error
    
    # Invalid JSON
    invalid_response = client.post(
        "/forecast",
        data="not a json",
        headers={"Content-Type": "application/json"}
    )
    assert invalid_response.status_code == 422  # FastAPI validation error 