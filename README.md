# AI Superforecaster

An LLM-based system for generating probabilistic forecasts using reference class forecasting, parameter estimation, and red team analysis with log-odds arithmetic for improved calibration.

## Architecture

The system is organized into several modular components:

- `src/models/`: Pydantic data models for forecast inputs and outputs
- `src/agents/`: LLM agent definitions organized by function (research, parameters, synthesis, questions)
- `src/utils/`: Utility functions including log-odds conversion and tools
- `src/ui/`: Display functions for CLI interface
- `src/forecasting_engine.py`: Core forecasting pipeline for consistent behavior across interfaces
- `main.py`: CLI interface with interactive command handling
- `ai_superforecaster.py`: Main entry point with integrated buffer visualization

## Workflow

1. Question validation and clarification
2. Background information gathering
3. Reference class identification and base rate extraction
4. Parameter design with interaction modeling
5. Evidence-based parameter research with log-odds estimation
6. Forecast synthesis using log-odds arithmetic
7. Red team challenge

## Key Benefits

- Log-odds arithmetic prevents common forecasting errors:
  - Eliminates probability creep (values exceeding 0-100%)
  - Fixes asymmetric influence of evidence across the probability range
  - Makes parameter contributions more transparent
  - Properly calibrates confidence intervals

## Usage

### Question Format Requirements

The system accepts questions in the following format:
```
What is the probability that [specific event] happens by [specific timeframe]?
```

Examples of valid questions:
- "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
- "What is the probability that SpaceX will launch humans to Mars before 2030?"
- "What is the probability that renewable energy will provide >50% of global electricity by 2035?"

The system requires:
- A clearly measurable event with objective criteria
- A specific timeframe or deadline
- Probability framing (not asking for values, rankings, or preferences)

### Recommended: Multi-Buffer GUI Experience

The easiest way to use the system is with the integrated GUI that shows all forecasting buffers in real-time:

```
python ai_superforecaster.py
```

Or run with a question directly:

```
python ai_superforecaster.py "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

You can also launch just the buffer viewer without starting a new forecast:

```
python ai_superforecaster.py --view-only
```

This shows separate panels for:
- **USER** - Input/output and status messages
- **BACKGROUND** - Reference classes and research
- **PARAMETERS** - Calculation steps and evidence strength
- **REPORT** - Final forecast and red team analysis

### Standard CLI (Advanced Users)

For those who prefer a text-only experience, you can use the CLI interface:

```
python main.py
```

Or run with a question directly:

```
python main.py "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

Commands available during execution:
- `/rerun` - Start a new forecast
- `/view <buffer>` - Display contents of a specific buffer (user, background, parameters, report)
- `/gui` - Launch the buffer viewer in a separate window
- `/quit` - Exit the application

## Communication System

The system uses an observer pattern for communication:
- The BufferManager maintains content in memory
- UI components register as observers to receive real-time updates
- No file I/O is used for communication, eliminating polling and improving performance
- Provides a clean foundation for future web interface development

## Agent Organization

The agents are organized into logical groups for better maintainability:

- `research_agents.py`: Background information and reference class research
- `parameter_agents.py`: Parameter design and parameter research
- `synthesis_agents.py`: Forecast synthesis and red team analysis
- `question_agents.py`: Question validation, clarification, and orchestration

## Features

- Reference class forecasting with multiple candidate classes
- Parameter estimation with calibrated log-odds contributions
- Red team analysis for challenging forecasts
- Real-time buffer visualization for tracking forecasting process
- Background research with web search capabilities
- Proper probability calibration using superforecaster techniques
- Multi-buffer agentic loop for better reasoning visibility

## Future Web Interface

The codebase includes placeholders for a future React web app interface:
- `src/utils/api_utils.py` contains scaffolding for the future API
- The observer pattern will transition smoothly to WebSocket-based communication
- Clear separation of UI and business logic enables multiple client implementations

## TODO:

- base classes have too much variance. should interpolate between them. Need to reduce variance in general. 
- Add tests
