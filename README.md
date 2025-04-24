# AI Superforecaster

An LLM-based system for generating probabilistic forecasts using reference class forecasting, parameter estimation, and red team analysis with log-odds arithmetic for improved calibration.

## Architecture

The system is organized into several modular components:

- `src/models/`: Pydantic data models for forecast inputs and outputs
- `src/agents/`: LLM agent definitions with prompts for specialized forecasting tasks
- `src/utils/`: Utility functions including log-odds conversion and tools
- `src/ui/`: Display functions for CLI interface
- `main.py`: CLI entry point for running forecasts
- `app.py`: Streamlit web interface for enhanced user experience
- `buffer_viewer.py`: GUI for visualizing forecast buffers in real-time
- `run_with_buffers.py`: Launcher for CLI + buffer viewer experience

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

### Standard CLI

The basic CLI interface with text output:

```
python main.py
```

Commands during execution:
- `/rerun` - Start a new forecast
- `/view` - Display contents of a specific buffer
- `/quit` - Exit the application
- `/gui` - Launch the buffer viewer GUI

### Multi-Buffer GUI Experience (Recommended)

To use the CLI with real-time buffer visualization:

```
python run_with_buffers.py
```

Or run with a question directly:

```
python run_with_buffers.py "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

This shows separate panels for:
- **USER** - Input/output and status messages
- **BACKGROUND** - Reference classes and research
- **LOGODDS** - Calculation steps and evidence strength
- **REPORT** - Final forecast and red team analysis

### Streamlit Web UI

To use the enhanced Streamlit interface:

```
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit UI provides a more interactive and visually appealing experience with:
- Real-time progress tracking
- Interactive visualizations of confidence intervals
- Organized display of reference classes and parameters
- Visual gauge for probability estimates
- Detailed log-odds calculation breakdowns
- Parameter impact visualization

## Buffer Management

The system uses a modular buffer system that:
- Captures all forecasting steps in separate buffers
- Writes content to files in real-time for external viewing
- Saves complete forecast runs with timestamps
- Enables multi-view interfaces without changing the core logic

## Features

- Reference class forecasting with multiple candidate classes
- Parameter estimation with calibrated log-odds contributions
- Red team analysis for challenging forecasts
- Interactive web interface with visualizations
- Background research with web search capabilities
- Proper probability calibration using superforecaster techniques
- Multi-buffer agentic loop for better reasoning visibility

## TODO:

- Add tests
- Support alternate questions and outcome formats
- Integration of expert opinions and additional data sources
- Advanced reasoning techniques (MCTS, linear models, autoregressive reasoning)
- Support for collaborative forecasting