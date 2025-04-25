# AI Superforecaster Usage Guide

## Question Format Requirements

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

## GUI Mode (Default)

The easiest way to use the system is with the integrated GUI that shows all forecasting buffers in real-time:

```bash
python main.py
```

You can also run with a question directly:

```bash
python main.py "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

Launch just the buffer viewer without starting a new forecast:

```bash
python main.py --view-only
```

## CLI Mode

For those who prefer a text-only experience, you can use the CLI interface:

```bash
python main.py --cli
```

Or run with a question directly:

```bash
python main.py --cli "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"
```

### CLI Commands

The following commands are available during CLI execution:
- `/rerun` - Start a new forecast
- `/view <buffer>` - Display contents of a specific buffer (user, background, parameters, report)
- `/gui` - Launch the buffer viewer in a separate window
- `/quit` - Exit the application

## API Server

For programmatic or web access, you can use the API server:

```bash
python api_server.py
```

The API server runs on port 8000 by default and provides the following endpoints:

- `POST /forecast` - Start a new forecast
- `GET /forecast/{session_id}` - Get forecast status and results
- `GET /forecast/{session_id}/buffers` - Get buffer contents

Example API request:
```bash
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the probability that Bitcoin will reach $100,000 before the end of 2025?"}'
``` 