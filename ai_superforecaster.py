#!/usr/bin/env python3
"""
AI Superforecaster
-----------------
Main entry point for the AI Superforecaster with multi-buffer visualization.

This script launches both:
1. The forecast engine (using src/forecasting_engine.py)
2. A real-time buffer visualization GUI

The multi-buffer display shows all aspects of the forecasting process:
- USER: Input/output and status messages
- BACKGROUND: Reference classes and research
- PARAMETERS: Calculation steps and evidence strength
- REPORT: Final forecast and red team analysis

Usage:
  ./ai_superforecaster.py
  ./ai_superforecaster.py "What is the probability that X will happen by Y?"
  ./ai_superforecaster.py --view-only  # Just show the buffer viewer without running forecast
"""
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import scrolledtext, simpledialog
import threading
import datetime
import argparse
import asyncio
import queue

from src.utils.buffers import BufferManager
from src.forecasting_engine import run_full_pipeline
from src.ui.cli import init_buffers
from src.utils.buffer_config import get_buffer_names, DEFAULT_BUFFERS

class BufferViewer:
    """
    GUI application that displays multiple forecast buffers in real-time.
    Receives buffer updates via observer pattern.
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
            "parameters": (1, 0),
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
        
        # Create control panel (buttons)
        self.control_panel = tk.Frame(root)
        self.control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # New forecast button
        self.new_forecast_button = tk.Button(
            self.control_panel, 
            text="Run New Forecast", 
            command=self.run_new_forecast,
            state=tk.NORMAL
        )
        self.new_forecast_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = tk.Button(
            self.control_panel, 
            text="Clear Buffers", 
            command=self.clear_all_buffers
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to receive buffer updates")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Track last update times for status bar updates
        self.last_update = {section: 0 for section in self.sections}
        
        # Queue for input requests
        self.input_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Start checking for input requests
        self.check_for_input_requests()
        
        # Store forecast thread
        self.forecast_thread = None
    
    def create_buffer_view(self, section, position):
        """Create a scrolled text widget for a buffer section at given grid position"""
        row, col = position
        frame = tk.LabelFrame(self.container, text=section.upper())
        frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
        text_widget.pack(fill='both', expand=True)
        
        # Store the text widget reference
        self.buffer_views[section] = text_widget
    
    def update_buffer_line(self, section, message, timestamp):
        """
        Add a single line to a buffer section.
        This is called by the BufferManager as an observer.
        """
        if section in self.buffer_views:
            text_widget = self.buffer_views[section]
            text_widget.configure(state='normal')
            text_widget.insert(tk.END, f"[{timestamp}] {message}\n")
            text_widget.see(tk.END)  # Auto-scroll to bottom
            text_widget.configure(state='disabled')
            
            # Update last update time for status bar
            self.last_update[section] = time.time()
            self.status_var.set(f"Updated buffer '{section}' at {datetime.datetime.now().strftime('%H:%M:%S')}")

    def clear_all_buffers(self):
        """Clear all buffer views"""
        for section in self.buffer_views:
            text_widget = self.buffer_views[section]
            text_widget.configure(state='normal')
            text_widget.delete(1.0, tk.END)
            text_widget.configure(state='disabled')
        
        self.status_var.set("Cleared all buffers")
    
    def run_new_forecast(self):
        """Display dialog for entering a new forecast question"""
        question = simpledialog.askstring("AI Superforecaster", 
                                      "What would you like to forecast?",
                                      parent=self.root)
        if question:
            # Clear buffers first
            self.clear_all_buffers()
            
            # Start forecast thread
            self.forecast_thread = threading.Thread(
                target=run_forecast_process,
                args=(question, self),
                daemon=True
            )
            self.forecast_thread.start()
        else:
            self.status_var.set("No question provided")
    
    def request_user_input(self, prompt):
        """
        Request input from the user via dialog box.
        This function is called from the forecasting thread.
        """
        # Put the request in the queue and wait for the response
        self.input_queue.put(prompt)
        return self.response_queue.get()
    
    def check_for_input_requests(self):
        """Check if there are any input requests and handle them"""
        try:
            # Non-blocking check for input requests
            if not self.input_queue.empty():
                prompt = self.input_queue.get_nowait()
                self.status_var.set("Input required...")
                
                # Show an input dialog
                response = simpledialog.askstring("Input Required", 
                                                 prompt,
                                                 parent=self.root)
                
                # Default to empty string if user cancels
                if response is None:
                    response = ""
                    
                # Put the response in the queue
                self.response_queue.put(response)
                self.status_var.set("Input provided, continuing...")
        except queue.Empty:
            pass
        
        # Schedule the next check
        self.root.after(100, self.check_for_input_requests)

class GuiInputProvider:
    """Provides input functionality for the forecasting engine using the GUI"""
    def __init__(self, viewer):
        self.viewer = viewer
    
    def get_input(self, prompt):
        """Get input from the user via the GUI"""
        return self.viewer.request_user_input(prompt)

async def run_forecast_async(question, viewer):
    """Run the forecasting engine asynchronously and connect to the viewer"""
    # Create buffer manager
    buffers = BufferManager(echo_user=True)
    init_buffers(buffers)
    
    # Add starting message to each buffer
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    for section in get_buffer_names():
        viewer.update_buffer_line(section, "Starting new forecast...", ts)
    
    # Register the viewer as an observer
    buffers.register_observer(viewer.update_buffer_line)
    
    # Create input provider for the GUI
    input_provider = GuiInputProvider(viewer)
    
    # Run the forecast
    await run_full_pipeline(question, buffers, input_provider)
    
    # Final status update
    viewer.status_var.set(f"Forecast completed at {datetime.datetime.now().strftime('%H:%M:%S')} - Ready for a new forecast")
    
    # Add completion message with instructions
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    viewer.update_buffer_line("user", "Forecast completed. Click 'Run New Forecast' to start another one.", ts)

def run_forecast_process(question, viewer):
    """Run the forecast process in a new asyncio event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_forecast_async(question, viewer))
    finally:
        loop.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Superforecaster with buffer visualization")
    parser.add_argument("question", nargs="?", help="Forecasting question (optional)")
    parser.add_argument("--view-only", action="store_true", help="Only show buffer viewer without running forecast")
    args = parser.parse_args()
    
    # Create and start the GUI
    root = tk.Tk()
    root.geometry("1200x800")
    viewer = BufferViewer(root)
    
    # Run forecast in a separate thread if not in view-only mode
    if not args.view_only and args.question:
        # Clear buffers first
        viewer.clear_all_buffers()
        
        # Start forecast thread
        viewer.forecast_thread = threading.Thread(
            target=run_forecast_process,
            args=(args.question, viewer),
            daemon=True
        )
        viewer.forecast_thread.start()
    elif not args.view_only:
        # Interactive mode - ask for the question first
        question = simpledialog.askstring("AI Superforecaster", 
                                        "What would you like to forecast?",
                                        parent=root)
        if question:
            # Clear buffers first
            viewer.clear_all_buffers()
            
            # Start forecast thread
            viewer.forecast_thread = threading.Thread(
                target=run_forecast_process,
                args=(question, viewer),
                daemon=True
            )
            viewer.forecast_thread.start()
        else:
            viewer.status_var.set("No question provided. Click 'Run New Forecast' to start.")
    else:
        viewer.status_var.set("Viewer started in view-only mode (no forecasting)")
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main() 