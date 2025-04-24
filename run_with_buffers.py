#!/usr/bin/env python3
"""
AI Superforecaster with Multi-Buffer GUI

This script launches the AI Superforecaster CLI with a separate
buffer viewer GUI window that shows all forecasting steps in real-time.

Usage:
  python run_with_buffers.py
  python run_with_buffers.py "What is the probability that X will happen by Y?"

The buffer viewer displays four panels:
- USER: Input/output and status messages
- BACKGROUND: Reference classes and parameter research
- LOGODDS: Calculation steps and evidence strength
- REPORT: Final forecast and red team analysis
"""
import os
import subprocess
import sys
import time
import glob

def main():
    # Clear any existing latest files
    for file in glob.glob("runs/latest_*.txt"):
        try:
            os.remove(file)
        except:
            pass

    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)

    # Start the buffer viewer in a separate process
    print("Starting buffer viewer...")
    try:
        viewer_process = subprocess.Popen(["python3", "buffer_viewer.py"])
        time.sleep(1)  # Give the viewer time to start
    except Exception as e:
        print(f"Warning: Could not start buffer viewer: {str(e)}")
        viewer_process = None

    # Get user input
    if len(sys.argv) > 1:
        # Use command line argument as question
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}")
        main_command = ["python3", "main.py"]
        main_process = subprocess.Popen(main_command, stdin=subprocess.PIPE, text=True)
        main_process.communicate(input=question)
    else:
        # Run main.py in interactive mode
        print("Running AI Superforecaster. Use the buffer viewer window to see all outputs.")
        main_process = subprocess.Popen(["python3", "main.py"])
        main_process.wait()

    # Clean up
    if viewer_process:
        print("Forecast complete. The buffer viewer window will remain open.")
        print("Close the viewer window when you're done reviewing the forecast.")

if __name__ == "__main__":
    main() 