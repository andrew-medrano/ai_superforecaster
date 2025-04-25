# AI Superforecaster Architecture

## System Components

The system is organized into several modular components:

- `src/models/`: Pydantic data models for forecast inputs and outputs
- `src/agents/`: LLM agent definitions organized by function (research, parameters, synthesis, questions)
- `src/utils/`: Utility functions including log-odds conversion and tools
- `src/ui/`: Display functions for CLI interface
- `src/forecasting_engine.py`: Core forecasting pipeline for consistent behavior across interfaces
- `main.py`: Main entry point with both GUI and CLI interfaces
- `api_server.py`: API server for web/remote access

## Workflow

1. Question validation and clarification
2. Background information gathering
3. Reference class identification and base rate extraction
4. Parameter design with interaction modeling
5. Evidence-based parameter research with log-odds estimation
6. Forecast synthesis using log-odds arithmetic
7. Red team challenge

## Agent Organization

The agents are organized into logical groups for better maintainability:

- `research_agents.py`: Background information and reference class research
- `parameter_agents.py`: Parameter design and parameter research
- `synthesis_agents.py`: Forecast synthesis and red team analysis
- `question_agents.py`: Question validation, clarification, and orchestration

## Communication System

The system uses an observer pattern for communication:
- The BufferManager maintains content in memory
- UI components register as observers to receive real-time updates
- No file I/O is used for communication, eliminating polling and improving performance
- Provides a clean foundation for future web interface development

## Benefits of Log-Odds Approach

- Eliminates probability creep (values exceeding 0-100%)
- Fixes asymmetric influence of evidence across the probability range
- Makes parameter contributions more transparent
- Properly calibrates confidence intervals 