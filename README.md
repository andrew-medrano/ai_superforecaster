# AI Superforecaster

An LLM-based system for generating probabilistic forecasts using reference class forecasting, parameter estimation, and red team analysis.

## Architecture

The system is organized into several modular components:

- `models.py`: Pydantic data models for forecast inputs and outputs
- `agent_definitions.py`: LLM agent definitions with prompts for specialized forecasting tasks
- `tools.py`: External tools (currently WebSearchTool) for data gathering
- `cli.py`: Command-line interface and display functions
- `main.py`: Core program flow and orchestration
- `app.py`: Streamlit web interface for enhanced user experience

## Workflow

1. Question validation and clarification
2. Background information gathering
3. Reference class identification and base rate extraction
4. Parameter design with interaction modeling
5. Evidence-based parameter research
6. Forecast synthesis
7. Red team challenge

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
- Tabbed interface for exploring different aspects of the forecast

## Features

- Reference class forecasting with multiple candidate classes
- Parameter estimation with confidence intervals
- Red team analysis for challenging forecasts
- Interactive web interface with visualizations
- Background research with web search capabilities

## TODO:

- alternate quetsions and alternate outcomes
- More robust parameter estimates and background research
- Integration of expert opinions and additional data sources
- Advanced reasoning techniques (MCTS, linear models, autoregressive reasoning)
- Support for collaborative forecasting