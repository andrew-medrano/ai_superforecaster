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

### Command Line Interface

```
python main.py
```

### Streamlit UI (Recommended)

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

## Features

- Reference class forecasting with multiple candidate classes
- Parameter estimation with calibrated log-odds contributions
- Red team analysis for challenging forecasts
- Interactive web interface with visualizations
- Background research with web search capabilities
- Proper probability calibration using superforecaster techniques

## TODO:

- Add tests
- Further improve reference class diversity
- Support alternate questions and outcome formats
- Integration of expert opinions and additional data sources
- Advanced reasoning techniques (MCTS, linear models, autoregressive reasoning)
- Support for collaborative forecasting