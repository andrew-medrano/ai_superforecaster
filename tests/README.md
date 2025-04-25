# AI Superforecaster Tests

This directory contains tests for the AI Superforecaster system.

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

### Running All Tests

```bash
# From the project root directory
python -m pytest tests/
```

### Running Specific Tests

```bash
# Run just the question validation tests
python -m pytest tests/test_question_validation.py

# Run a specific test function
python -m pytest tests/test_question_validation.py::test_valid_questions

# Run tests with more verbose output
python -m pytest tests/test_question_validation.py -v
```

### Running Tests Directly

You can also run the test file directly, which executes all tests in sequence:

```bash
python tests/test_question_validation.py
``` 