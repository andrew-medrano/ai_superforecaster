"""
Buffer Management System for AI Superforecaster

This module provides classes for managing text buffers across different parts
of the forecasting process. The BufferManager allows different components
to write to separate buffers, which can be displayed in various ways:
- Printed to console (echo_user=True)
- Observed by UI components via callbacks

This is the core of the multi-buffer architecture that allows the system
to maintain separation of concerns without changing the core logic.
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Union
from src.utils.buffer_config import get_buffer_names, DEFAULT_BUFFERS

class TextBuffer:
    """A single text buffer that accumulates timestamped entries."""
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def write(self, content: str, content_type: str = "text") -> None:
        """Add a timestamped entry to the buffer."""
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.entries.append({
            "content": content,
            "timestamp": ts,
            "type": content_type
        })

    def dump(self) -> str:
        """Get the entire buffer contents as a string."""
        return "\n".join([f"[{entry['timestamp']}] {entry['content']}" 
                         for entry in self.entries])

class BufferManager:
    """
    Manages multiple named text buffers for different parts of the forecasting process.
    
    Key features:
    - Maintains separate buffers for user, background, parameters, report
    - Can echo user buffer to console in real-time
    - Notifies observers when buffer content changes
    - Supports different content types for future extensibility
    
    This enables multiple UI approaches (CLI, multi-window, web) with the same core logic.
    """
    def __init__(self, buffer_names: Optional[List[str]] = None, echo_user: bool = True) -> None:
        """
        Initialize a new BufferManager.
        
        Args:
            buffer_names: Names of buffers to initialize (default: all from config)
            echo_user: Whether to print user buffer contents to console (default: True)
        """
        self._bufs: Dict[str, TextBuffer] = defaultdict(TextBuffer)
        self.echo_user = echo_user
        self.observers: List[Callable[[str, str, str, str], None]] = []
        
        # Initialize requested buffers or defaults
        if buffer_names is None:
            buffer_names = get_buffer_names()
            
        # Create empty buffers for each name
        for name in buffer_names:
            self._bufs[name] = TextBuffer()

    def register_observer(self, callback: Callable[[str, str, str], None]) -> None:
        """
        Register a function to be called when buffer content changes.
        
        The callback receives three arguments:
        - section: The buffer section that was updated (e.g., "user")
        - message: The message that was added to the buffer
        - timestamp: The timestamp of the message
        
        Note: For backward compatibility, this version only passes the text content,
        not the content type.
        """
        # Wrap the callback to handle the new signature with content_type
        def wrapped_callback(section, message, timestamp, content_type=None):
            callback(section, message, timestamp)
            
        self.observers.append(wrapped_callback)

    def write(self, section: str, content: Union[str, Any], content_type: str = "text") -> None:
        """
        Write content to a named buffer section.
        
        Args:
            section: The buffer section to write to (e.g., "user", "background")
            content: Content to write (string for text, other objects for different types)
            content_type: Type of content ("text", "plot", "interactive", etc.)
        """
        # Update in-memory buffer
        buf = self._bufs[section]
        
        # For backward compatibility, join string parts if content is a list/tuple of strings
        if isinstance(content, (list, tuple)) and all(isinstance(x, str) for x in content):
            message = " ".join(content)
            content = message
        
        # Convert non-text content to a string representation for current displays
        display_content = content
        if content_type != "text":
            display_content = f"[{content_type.upper()} content - requires GUI to view]"
        
        # Echo to console if needed
        if section == "user" and self.echo_user:
            print(display_content if isinstance(display_content, str) else str(display_content))
        
        # Get timestamp for the entry
        ts = datetime.utcnow().strftime("%H:%M:%S")
        
        # Add entry to buffer
        buf.write(content, content_type)
        
        # Notify observers
        for observer in self.observers:
            # Pass the string representation for display
            display_text = display_content if isinstance(display_content, str) else str(display_content)
            observer(section, display_text, ts, content_type)

    def dump(self, section: str) -> str:
        """Get the entire contents of a buffer section."""
        return self._bufs[section].dump()

    @property
    def sections(self):
        """Get all active buffer section names."""
        return self._bufs.keys()
        
    def save_run(self, prefix=""):
        """
        Deprecated: This method is kept for backward compatibility but doesn't save files.
        In the future, this could be reimplemented to save data to a database or other storage.
        """
        pass 