#!/usr/bin/env python3
"""
Test Question Validation

Minimal test for essential question validation functionality.
"""
import asyncio
import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Runner
from src.agents import question_validator_agent, question_clarifier_agent
from src.models import ForecastabilityCheck, QuestionClarification

@pytest.mark.asyncio
async def test_minimal_validation():
    """Single test that validates the most essential functionality."""
    # Test valid forecasting question
    valid_question = "What is the probability that Bitcoin will reach $100,000 by the end of 2025?"
    
    print(f"Testing valid question: {valid_question}")
    valid_result = await Runner.run(question_validator_agent, valid_question)
    valid_check = valid_result.final_output_as(ForecastabilityCheck)
    assert valid_check.is_forecastable, f"Valid question failed validation: {valid_check.reasoning}"
    print(f"✓ Valid question passed")
    
    # Test invalid question
    invalid_question = "What is the best programming language?"
    
    print(f"Testing invalid question: {invalid_question}")
    invalid_result = await Runner.run(question_validator_agent, invalid_question)
    invalid_check = invalid_result.final_output_as(ForecastabilityCheck)
    assert not invalid_check.is_forecastable, "Invalid question incorrectly marked as valid"
    print(f"✓ Invalid question correctly rejected")
    
    # Test question clarification
    ambiguous_question = "Will AI replace programmers?"
    
    print(f"Testing ambiguous question: {ambiguous_question}")
    clarification_result = await Runner.run(question_clarifier_agent, ambiguous_question)
    clarification = clarification_result.final_output_as(QuestionClarification)
    clarified = clarification.clarified_question
    
    print(f"Clarified to: {clarified}")
    validation_result = await Runner.run(question_validator_agent, clarified)
    validation = validation_result.final_output_as(ForecastabilityCheck)
    assert validation.is_forecastable, f"Clarified question failed validation: {validation.reasoning}"
    print(f"✓ Clarified question passed validation")
    
    return True

if __name__ == "__main__":
    print("Starting minimal question validation test...\n")
    asyncio.run(test_minimal_validation()) 