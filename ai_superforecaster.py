#!/usr/bin/env python3
"""
AI Superforecaster
-----------------
Main entry point for the AI Superforecaster with multi-buffer visualization.

This script launches both:
1. The forecast engine (main.py)
2. A real-time buffer visualization GUI

The multi-buffer display shows all aspects of the forecasting process:
- USER: Input/output and status messages
- BACKGROUND: Reference classes and research
- LOGODDS: Calculation steps and evidence strength
- REPORT: Final forecast and red team analysis

Usage:
  ./ai_superforecaster.py
  ./ai_superforecaster.py "What is the probability that X will happen by Y?"
"""
import os
import subprocess
import sys
import time
import glob
import tkinter as tk
from tkinter import scrolledtext
import threading
import datetime

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
        updated = False
        
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
                            updated = True
                except Exception as e:
                    self.status_var.set(f"Error reading {file_path}: {str(e)}")
                    # In case of error, reset the buffer with a message
                    self.update_buffer_content(section, f"Error reading buffer file: {str(e)}\nWaiting for content...")
        
        if updated:
            self.status_var.set(f"Updated buffers at {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Schedule the next check
        self.root.after(500, self.check_for_updates)  # Check every 500ms

def run_forecast_process(question=None):
    """Run the main.py script, either interactively or with a provided question"""
    # Note: We already cleared the latest files in main(), don't need to do it again here
    
    # Run main.py with or without a provided question
    if question:
        print(f"Question: {question}")
        main_process = subprocess.Popen(["python3", "main.py"], stdin=subprocess.PIPE, text=True)
        main_process.communicate(input=question)
    else:
        print("Running AI Superforecaster. Use the buffer viewer window to see all outputs.")
        main_process = subprocess.Popen(["python3", "main.py"])
        main_process.wait()
    
    print("Forecast complete. The buffer viewer window will remain open.")
    print("Close the viewer window when you're done reviewing the forecast.")

def main():
    # Get command-line arguments
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    # Clear any existing latest files BEFORE creating the GUI
    for file in glob.glob("runs/latest_*.txt"):
        try:
            os.remove(file)
        except:
            pass
    
    # Create empty latest files for each buffer section to avoid any
    # old content being displayed when the viewer first starts
    sections = ["user", "background", "logodds", "report"]
    os.makedirs("runs", exist_ok=True)
    for section in sections:
        with open(f"runs/latest_{section}.txt", "w") as f:
            f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting new forecast...\n")
    
    # Create and start the GUI
    root = tk.Tk()
    root.geometry("1200x800")
    viewer = BufferViewer(root)
    
    # Run forecast in a separate thread so it doesn't block the GUI
    forecast_thread = threading.Thread(
        target=run_forecast_process,
        args=(question,),
        daemon=True
    )
    forecast_thread.start()
    
    # Start the Tkinter event loop (this will block until window is closed)
    root.mainloop()
    
    # When the window is closed, the program will exit naturally

if __name__ == "__main__":
    main() 