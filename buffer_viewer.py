#!/usr/bin/env python3
"""
AI Superforecaster Buffer Viewer

A GUI application that displays the different buffers of the AI Superforecaster
in real-time. This viewer works by monitoring text files in the 'runs' directory
that are updated during forecast generation.

The viewer displays four panels in a 2x2 grid:
- USER: Interactive I/O and status messages
- BACKGROUND: Reference classes and parameter research
- LOGODDS: Calculation details for log-odds arithmetic
- REPORT: Final forecast and red team analysis

This file-based approach avoids threading issues on macOS and
allows the buffer viewer to run completely independently from
the main forecasting process.

Usage:
  python buffer_viewer.py

Note: This is typically launched automatically by run_with_buffers.py
rather than being run directly.
"""
import tkinter as tk
from tkinter import scrolledtext
import os
import glob
import time

class BufferViewer:
    """
    GUI application that displays multiple forecast buffers in real-time.
    Reads buffer content from text files and updates the display periodically.
    """
    def __init__(self, root):
        self.root = root
        root.title("AI Superforecaster - Live Buffers")
        
        # Create main container
        self.container = tk.Frame(root)
        self.container.pack(fill='both', expand=True)
        
        # Dictionary to store text widgets for each buffer
        self.buffer_views = {}
        
        # Initial buffer sections and their grid positions (row, column)
        self.sections = {
            "user": (0, 0),
            "background": (0, 1),
            "logodds": (1, 0),
            "report": (1, 1)
        }
        
        # Configure grid weights to make all cells expand equally
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=1)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_rowconfigure(1, weight=1)
        
        # Create views for each section
        for section, pos in self.sections.items():
            self.create_buffer_view(section, pos)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Watching for buffer files in runs/...")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Track last update times
        self.last_update = {section: 0 for section in self.sections}
        
        # Create runs directory if it doesn't exist
        os.makedirs("runs", exist_ok=True)
        
        # Start checking for updates periodically
        self.check_for_updates()
    
    def create_buffer_view(self, section, position):
        """Create a scrolled text widget for a buffer section at given grid position"""
        row, col = position
        frame = tk.LabelFrame(self.container, text=section.upper())
        frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        text_widget.pack(fill='both', expand=True)
        
        # Store the text widget reference
        self.buffer_views[section] = text_widget
    
    def update_buffer_content(self, section, content):
        """Update a buffer view with new content"""
        if section in self.buffer_views:
            text_widget = self.buffer_views[section]
            text_widget.configure(state='normal')
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, content)
            text_widget.see(tk.END)  # Auto-scroll to bottom
            text_widget.configure(state='disabled')
    
    def check_for_updates(self):
        """Check for updates to buffer files"""
        # Look for real-time files first (latest_{section}.txt)
        for section in self.sections:
            file_path = os.path.join("runs", f"latest_{section}.txt")
            if os.path.exists(file_path):
                try:
                    modified_time = os.path.getmtime(file_path)
                    if modified_time > self.last_update[section]:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            self.update_buffer_content(section, content)
                            self.last_update[section] = modified_time
                            self.status_var.set(f"Updated {section} from latest file")
                except Exception as e:
                    self.status_var.set(f"Error reading {file_path}: {str(e)}")
        
        # Schedule the next check
        self.root.after(500, self.check_for_updates)  # Check every 500ms

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = BufferViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 