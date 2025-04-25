"""
Buffer Configuration for AI Superforecaster

Centralizes the configuration of buffer types and their properties.
This makes it easier to add, remove, or modify buffers without
changing code in multiple places throughout the application.
"""

DEFAULT_BUFFERS = {
    "user": {
        "description": "Interactive I/O and status messages",
        "required": True
    },
    "background": {
        "description": "Reference classes and parameter research",
        "required": True
    },
    "parameters": {
        "description": "Calculation steps and evidence strength",
        "required": True
    },
    "report": {
        "description": "Final forecast and red team analysis",
        "required": True
    }
}

def get_buffer_names():
    """Get list of all buffer names"""
    return list(DEFAULT_BUFFERS.keys())

def get_buffer_description(name):
    """Get description for a specific buffer"""
    return DEFAULT_BUFFERS.get(name, {}).get("description", "Custom buffer")

def is_buffer_required(name):
    """Check if a buffer is required"""
    return DEFAULT_BUFFERS.get(name, {}).get("required", False) 