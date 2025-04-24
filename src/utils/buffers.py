"""
Buffer Management System for AI Superforecaster

This module provides classes for managing text buffers across different parts
of the forecasting process. The BufferManager allows different components
to write to separate buffers, which can be displayed in various ways:
- Printed to console (echo_user=True)
- Written to files in real-time (real_time_files=True)
- Saved as complete runs with timestamps

This is the core of the multi-buffer architecture that allows the system
to maintain separation of concerns without changing the core logic.
"""
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import os

class TextBuffer:
    """A single text buffer that accumulates timestamped messages."""
    def __init__(self) -> None:
        self.lines: List[str] = []

    def write(self, *parts: str) -> None:
        """Add a timestamped message to the buffer."""
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.lines.append(f"[{ts}] " + " ".join(parts))

    def dump(self) -> str:
        """Get the entire buffer contents as a string."""
        return "\n".join(self.lines)

class BufferManager:
    """
    Manages multiple named text buffers for different parts of the forecasting process.
    
    Key features:
    - Maintains separate buffers for user, background, logodds, report
    - Can echo user buffer to console in real-time
    - Can write to live files for GUI viewing
    - Saves complete runs with timestamps
    
    This enables multiple UI approaches (CLI, multi-window, web) with the same core logic.
    """
    def __init__(self, echo_user: bool = True, real_time_files: bool = True) -> None:
        """
        Initialize a new BufferManager.
        
        Args:
            echo_user: Whether to print user buffer contents to console (default: True)
            real_time_files: Whether to write buffer contents to files in real-time (default: True)
        """
        self._bufs: Dict[str, TextBuffer] = defaultdict(TextBuffer)
        self.echo_user = echo_user
        self.real_time_files = real_time_files
        
        # Create runs directory if needed
        if self.real_time_files:
            os.makedirs("runs", exist_ok=True)

    def write(self, section: str, *parts: str) -> None:
        """
        Write a message to a named buffer section.
        
        Args:
            section: The buffer section to write to (e.g., "user", "background")
            *parts: Text parts to join and write
        """
        buf = self._bufs[section]
        buf.write(*parts)
        if section == "user" and self.echo_user:
            print(" ".join(parts))
            
        # Write to real-time files if enabled
        if self.real_time_files:
            message = " ".join(parts)
            with open(f"runs/latest_{section}.txt", "a") as f:
                ts = datetime.utcnow().strftime("%H:%M:%S")
                f.write(f"[{ts}] {message}\n")

    def dump(self, section: str) -> str:
        """Get the entire contents of a buffer section."""
        return self._bufs[section].dump()

    @property
    def sections(self):
        """Get all active buffer section names."""
        return self._bufs.keys()
        
    def save_run(self, prefix=""):
        """
        Save all buffers to timestamped files.
        
        Args:
            prefix: Optional prefix for the filename (e.g., a slug of the question)
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if prefix:
            timestamp = f"{prefix}_{timestamp}"
            
        os.makedirs("runs", exist_ok=True)
        
        for section in self.sections:
            with open(f"runs/{timestamp}_{section}.txt", "w") as f:
                f.write(self.dump(section)) 