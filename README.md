# AI Superforecaster

An LLM-based system for generating probabilistic forecasts using reference class forecasting, parameter estimation, and red team analysis with log-odds arithmetic for improved calibration.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai_superforecaster.git
cd ai_superforecaster

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Forecaster

**With GUI (Default)**:
```bash
python main.py
```

**With CLI Interface**:
```bash
python main.py --cli
```

**API Server**:
```bash
python api_server.py
```

### Example Usage

Run with a specific question:
```bash
# GUI mode with a question
python main.py "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"

# CLI mode with a question
python main.py --cli "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

## Question Format

The system works best with clear, measurable probability questions:
```
What is the probability that [specific event] happens by [specific timeframe]?
```

Examples:
- "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
- "What is the probability that SpaceX will launch humans to Mars before 2030?"

## Key Features

- Reference class forecasting with multiple candidate classes
- Parameter estimation with calibrated log-odds contributions
- Red team analysis for challenging forecasts
- Real-time buffer visualization for tracking forecasting process

## GUI Interface

The GUI shows separate panels for:
- **USER** - Input/output and status messages
- **BACKGROUND** - Reference classes and research
- **PARAMETERS** - Calculation steps and evidence strength
- **REPORT** - Final forecast and red team analysis

## CLI Commands

Commands available during CLI execution:
- `/rerun` - Start a new forecast
- `/view <buffer>` - Display contents of a specific buffer (user, background, parameters, report)
- `/gui` - Launch the buffer viewer in a separate window
- `/quit` - Exit the application

## Documentation

For more detailed information, see the [documentation](docs/).
