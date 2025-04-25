#!/usr/bin/env python3
"""
AI Superforecaster
-----------------
Main entry point for the AI Superforecaster with both CLI and GUI interfaces.

This script can run in several modes:
1. GUI mode (default): Shows multi-buffer visualization in a graphical interface
2. CLI mode: Provides a command-line interface for the forecasting engine
3. View-only mode: Just shows the buffer viewer without running a forecast

The multi-buffer display shows all aspects of the forecasting process:
- USER: Input/output and status messages
- BACKGROUND: Reference classes and research
- PARAMETERS: Calculation steps and evidence strength
- REPORT: Final forecast and red team analysis

Usage:
  python main.py                      # Start with GUI interface
  python main.py --cli                # Start with command-line interface
  python main.py "What is the probability that X will happen by Y?"  # Run forecast immediately
  python main.py --view-only          # Just show the buffer viewer without running forecast

Commands (during CLI execution):
  /rerun - Start a new forecast
  /view <buffer> - View buffer contents (user, background, parameters, report)
  /gui - Launch the buffer viewer in a separate window
  /quit - Exit the application
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
from src.forecasting_engine import run_full_pipeline, ConsoleInputProvider
from src.ui.cli import init_buffers, display_welcome
from src.utils.buffer_config import get_buffer_names, get_buffer_description, DEFAULT_BUFFERS

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
        
        # Color configuration
        self.colors = {
            "user": {"fg": "#0078d7", "header_fg": "#005999", "success_fg": "#009900"},  # Blue, darker blue, green
            "background": {"fg": "#8252c7", "header_fg": "#5a3a8a"},  # Purple, darker purple
            "parameters": {"fg": "#d75f00", "header_fg": "#a04700", 
                          "positive_fg": "#007700", "negative_fg": "#cc0000"},  # Orange, darker orange, green, red
            "report": {"fg": "#007744", "header_fg": "#005533",
                      "redteam_fg": "#bb0000"}  # Green, darker green, red
        }
        
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
        frame = tk.LabelFrame(self.container, text=section.upper(), font=("TkDefaultFont", 10, "bold"), fg="black")
        frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
        
        # Set background color for frame based on section
        section_bg_colors = {
            "user": "#f0f8ff",      # Light blue background
            "background": "#f5f0ff", # Light purple background
            "parameters": "#fff5f0", # Light orange background
            "report": "#f0fff5"      # Light green background
        }
        
        bg_color = section_bg_colors.get(section, "#ffffff")
        frame.configure(bg=bg_color)
        
        # Create text widget with matching background
        text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, bg=bg_color)
        text_widget.pack(fill='both', expand=True)
        
        # Configure tags for this section
        colors = self.colors.get(section, {"fg": "black", "header_fg": "black"})
        
        # Basic color for normal text
        text_widget.tag_configure("normal", foreground=colors["fg"])
        
        # Headers (=== TEXT ===)
        text_widget.tag_configure("header", foreground=colors["header_fg"], 
                                 font=("TkDefaultFont", 10, "bold"),
                                 spacing1=5, spacing3=5)  # Add spacing around headers
        
        # Success messages (with ✓)
        if "success_fg" in colors:
            text_widget.tag_configure("success", foreground=colors["success_fg"], 
                                     font=("TkDefaultFont", 10, "bold"))
        
        # Parameters-specific tags
        if section == "parameters":
            text_widget.tag_configure("positive", foreground=colors["positive_fg"], 
                                     font=("TkDefaultFont", 9, ""))
            text_widget.tag_configure("negative", foreground=colors["negative_fg"],
                                     font=("TkDefaultFont", 9, ""))
        
        # Report-specific tags
        if section == "report":
            text_widget.tag_configure("redteam", foreground=colors["redteam_fg"],
                                     font=("TkDefaultFont", 9, "bold"))
        
        # Timestamp tag (light gray)
        text_widget.tag_configure("timestamp", foreground="#999999")
            
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
            
            # Insert the timestamp with light gray color
            text_widget.insert(tk.END, f"[{timestamp}] ", "timestamp")
            
            # Determine which tag to use based on the content and section
            tag = "normal"
            
            # Special pattern matching for different types of content
            if message.strip().startswith("===") and message.strip().endswith("==="):
                # Headers (e.g. === FINAL FORECAST ===)
                tag = "header"
            elif "✓" in message:
                # Success messages with checkmark
                tag = "success"
            elif section == "user" and message.startswith("Question:"):
                # User questions 
                tag = "header"
            elif section == "parameters":
                # Parameter-specific formatting
                if "+=" in message or "+0." in message or "+" in message and "log-odds" in message.lower():
                    # Positive parameter impact
                    tag = "positive"
                elif "-=" in message or "-0." in message or ("log-odds" in message.lower() and not "+" in message):
                    # Negative parameter impact
                    tag = "negative"
                elif "final log-odds" in message.lower() or "probability" in message.lower():
                    # Final probability
                    tag = "header"
                elif "base rate" in message.lower():
                    # Base rate
                    tag = "normal"
                elif "conservative shift" in message.lower() or "moderate shift" in message.lower():
                    # Good shifts
                    tag = "success" 
                elif "large shift" in message.lower() or "extreme shift" in message.lower():
                    # Concerning shifts
                    tag = "negative"
            elif section == "report":
                # Report-specific formatting
                if "probability:" in message.lower():
                    # Final probability
                    tag = "header"
                elif "strongest objection" in message.lower() or "alternate estimate" in message.lower() or "red team" in message.lower():
                    # Red team content
                    tag = "redteam"
            elif section == "background":
                # Background-specific formatting
                if "reference class" in message.lower() and "recommended" in message.lower():
                    # Recommended reference class
                    tag = "success"
                elif "base rate" in message.lower():
                    # Base rates are important
                    tag = "header"
                    
            # Insert the message with the appropriate tag
            text_widget.insert(tk.END, message + "\n", tag)
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
            
        # Check again after a delay (100ms)
        self.root.after(100, self.check_for_input_requests)

class GuiInputProvider:
    """Input provider for GUI mode that requests input via the GUI"""
    def __init__(self, viewer):
        self.viewer = viewer
        
    def get_input(self, prompt):
        """Get input from the user via GUI dialog"""
        return self.viewer.request_user_input(prompt)

async def run_forecast_async(question, viewer):
    """Run the forecast asynchronously with GUI updates"""
    try:
        # Create a buffer manager that will send updates to the GUI
        buffer_manager = BufferManager(echo_user=False)
        
        # Register the buffer viewer as observer
        buffer_manager.register_observer(
            lambda section, message, timestamp, content_type=None: 
            viewer.update_buffer_line(section, message, timestamp)
        )
        
        # Initialize the buffers with standard headers
        init_buffers(buffer_manager)
        
        # Create GUI input provider
        input_provider = GuiInputProvider(viewer)
        
        # Welcome message
        buffer_manager.write("user", "Starting AI Superforecaster...")
        
        # Run the forecast pipeline
        result = await run_full_pipeline(question, buffer_manager, input_provider)
        
        if result:
            buffer_manager.write("user", "Forecast completed successfully.")
        else:
            buffer_manager.write("user", "Forecast failed to produce a result.")
            
    except Exception as e:
        print(f"Error in forecast: {str(e)}")
        if viewer:
            viewer.status_var.set(f"Error: {str(e)}")

def run_forecast_process(question, viewer):
    """Run the forecast in a separate thread"""
    asyncio.run(run_forecast_async(question, viewer))

async def handle_cli_mode(buffer_manager, input_provider):
    """Handle command-line interactive mode"""
    display_welcome()
    print("Type your question or a command (/help, /quit):")

    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            # Check for commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                
                if command == "/quit":
                    print("Exiting...")
                    break
                elif command == "/help":
                    print_cli_help()
                    continue
                elif command == "/rerun":
                    print("Starting a new forecast:")
                    continue
                elif command == "/view":
                    # Handle viewing buffer contents
                    if len(parts) < 2:
                        print(f"Available buffers: {', '.join(get_buffer_names())}")
                        continue
                    
                    buffer_name = parts[1]
                    if buffer_name in buffer_manager.sections:
                        print(f"\n--- {buffer_name.upper()} BUFFER ---")
                        print(buffer_manager.dump(buffer_name))
                        print("----------------------")
                    else:
                        print(f"Buffer '{buffer_name}' not found.")
                    continue
                elif command == "/gui":
                    # Launch the buffer viewer GUI in a separate process
                    print("Launching buffer viewer...")
                    subprocess.Popen([sys.executable, __file__, "--view-only"])
                    continue
                else:
                    print("Unknown command.")
                    print(f"Available commands: /help, /rerun, /view, /quit, /gui")
                    continue
            
            # Skip empty input
            if not user_input:
                continue
                
            # Process the question
            await run_full_pipeline(user_input, buffer_manager, input_provider)
            
            print("\nEnter a new question or command (/help, /rerun, /view, /quit, /gui):")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

def print_cli_help():
    """Print help information for CLI mode"""
    print("AI Superforecaster CLI")
    print("\nThis tool creates probabilistic forecasts using reference class forecasting and")
    print("parameter estimation. The forecasting process uses the following buffers:\n")
    
    for buffer_name in get_buffer_names():
        print(f"- {buffer_name}: {get_buffer_description(buffer_name)}")
    
    print("\nCommands (during interactive session):")
    print("  /rerun - Start a new forecast")
    print(f"  /view <buffer> - View buffer contents ({', '.join(get_buffer_names())})")
    print("  /help - Show this help message")
    print("  /quit - Exit the application")
    print("  /gui - Launch the buffer viewer GUI")

async def main_async():
    """Async main function that handles all execution modes"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Superforecaster")
    parser.add_argument("question", nargs="?", help="Forecasting question (optional)")
    parser.add_argument("--cli", action="store_true", help="Run in command-line interface mode")
    parser.add_argument("--view-only", action="store_true", help="Launch only the buffer viewer")
    parser.add_argument("--non-interactive", "-n", action="store_true", 
                      help="Run in non-interactive mode with input from stdin")
    parser.add_argument("--output", "-o", choices=get_buffer_names(), default="user", help="Which buffer to output")
    parser.add_argument("--help-buffers", action="store_true", help="Show detailed information about buffers")
    args = parser.parse_args()
    
    # If help for buffers is requested, show that and exit
    if args.help_buffers:
        print_cli_help()
        return
    
    # Command-line interface mode
    if args.cli:
        # Initialize buffer manager for CLI mode
        buffer_manager = BufferManager(echo_user=True)
        init_buffers(buffer_manager)
        
        # Create console input provider
        input_provider = ConsoleInputProvider()
        
        # Non-interactive mode from command line argument
        if args.question:
            await run_full_pipeline(args.question, buffer_manager, input_provider)
            return
            
        # Non-interactive mode from stdin
        if args.non_interactive:
            print("Reading from stdin...")
            question = sys.stdin.read().strip()
            if question:
                await run_full_pipeline(question, buffer_manager, input_provider)
            else:
                print("Error: No input provided on stdin.")
            return
        
        # Interactive mode
        await handle_cli_mode(buffer_manager, input_provider)
        return
    
    # GUI mode
    # Start the Tkinter application
    root = tk.Tk()
    root.geometry("1200x800")
    viewer = BufferViewer(root)
    
    # If a question was provided, start forecast immediately
    if args.question:
        # Schedule after a short delay to let the GUI initialize
        root.after(500, lambda: run_forecast_process(args.question, viewer))
    
    # Run the Tkinter event loop
    root.mainloop()

def main():
    """Main entry point for the application"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 