#!/usr/bin/env python3
"""
AI Superforecaster CLI

This is the main entry point for the AI Superforecaster command-line interface.
It provides a command-line interface for the forecasting engine.

This application uses a buffer system that separates content into different views:
- user: Interactive I/O and status messages
- background: Reference classes and parameter research
- parameters: Calculation details for log-odds
- report: Final forecast and red team analysis

Usage:
  python main.py                # Interactive mode
  echo "question" | python main.py  # Non-interactive mode
  python main.py --help         # Show help message

Commands (during interactive session):
  /rerun - Start a new forecast
  /view <buffer> - View buffer contents (user, background, parameters, report)
  /quit - Exit the application
  /gui - Launch the buffer viewer GUI

For a better experience with visual buffer display, use:
  python ai_superforecaster.py
"""
import asyncio
import os
import sys
import subprocess
import argparse

from src.forecasting_engine import run_full_pipeline, ConsoleInputProvider
from src.ui.cli import display_welcome, init_buffers
from src.utils.buffers import BufferManager
from src.utils.buffer_config import get_buffer_names, get_buffer_description

async def main():
    """Main entry point for the AI Superforecaster CLI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Superforecaster CLI")
    parser.add_argument("question", nargs="?", help="Forecasting question (optional)")
    parser.add_argument("--non-interactive", "-n", action="store_true", 
                      help="Run in non-interactive mode with input from stdin")
    parser.add_argument("--output", "-o", choices=get_buffer_names(), default="user", help="Which buffer to output")
    parser.add_argument("--help-buffers", action="store_true", help="Show detailed information about buffers")
    args = parser.parse_args()
    
    # If help for buffers is requested, show that and exit
    if args.help_buffers:
        print_help()
        return
    
    # Initialize buffer manager
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
    await handle_interactive_mode(buffer_manager, input_provider)

async def handle_interactive_mode(buffer_manager, input_provider):
    """Handle interactive mode with the user."""
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
                    print_help()
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
                    subprocess.Popen(["python3", "ai_superforecaster.py", "--view-only"])
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

def print_help():
    """Print help information."""
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

if __name__ == "__main__":
    asyncio.run(main())