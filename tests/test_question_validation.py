#!/usr/bin/env python3
"""
Test Question Validation

Simplified and parallel testing of question validation.
Uses asyncio.gather to run tests in batch for better performance.
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import Runner
from src.agents import question_validator_agent, question_clarifier_agent
from src.models import ForecastabilityCheck, QuestionClarification

# Simplified test sets
VALID_QUESTIONS = [
    "What is the probability that Bitcoin will reach $100,000 by the end of 2025?",
    "What is the probability that SpaceX will launch humans to Mars before 2030?",
    "What's the likelihood that Apple will release AR glasses by 2026?",
    "What is the chance that China's GDP will surpass the US GDP by 2030?",
]

NEEDS_CLARIFICATION = [
    "Will AI replace programmers?",
    "Is Bitcoin a good investment?",
    "When will we reach AGI?",
    "Will clean energy take over?",
]

INVALID_QUESTIONS = [
    "What is the best programming language?",
    "Who really killed JFK?",
    "Tell me a joke",
    "Hello, how are you?",
    "Write code to sort a list in Python",
]

async def test_validation(question, expected_valid=True):
    """Test if a question passes validation as expected."""
    result = await Runner.run(question_validator_agent, question)
    check = result.final_output_as(ForecastabilityCheck)
    
    if check.is_forecastable == expected_valid:
        print(f"✓ {'Valid' if expected_valid else 'Invalid'}: {question}")
        return True
    else:
        print(f"✗ Failed: {question}")
        print(f"  Expected: {'valid' if expected_valid else 'invalid'}")
        print(f"  Reason: {check.reasoning}")
        return False

async def test_clarification(question):
    """Test if a question can be clarified correctly."""
    # First get clarification
    clarification_result = await Runner.run(question_clarifier_agent, question)
    clarification = clarification_result.final_output_as(QuestionClarification)
    clarified = clarification.clarified_question
    
    # Then validate the clarified question
    validation_result = await Runner.run(question_validator_agent, clarified)
    validation = validation_result.final_output_as(ForecastabilityCheck)
    
    if validation.is_forecastable:
        print(f"✓ Clarified: {question} → {clarified}")
        return True
    else:
        print(f"✗ Failed clarification: {question} → {clarified}")
        print(f"  Reason: {validation.reasoning}")
        return False

async def run_parallel_tests():
    """Run all tests in parallel using asyncio.gather."""
    print("Running validation tests for valid questions...")
    valid_results = await asyncio.gather(
        *[test_validation(q, True) for q in VALID_QUESTIONS]
    )
    print(f"\nValid questions passed: {sum(valid_results)}/{len(valid_results)}\n")
    
    print("Running validation tests for invalid questions...")
    invalid_results = await asyncio.gather(
        *[test_validation(q, False) for q in INVALID_QUESTIONS]
    )
    print(f"\nInvalid questions passed: {sum(invalid_results)}/{len(invalid_results)}\n")
    
    print("Running clarification tests...")
    clarification_results = await asyncio.gather(
        *[test_clarification(q) for q in NEEDS_CLARIFICATION]
    )
    print(f"\nClarification passed: {sum(clarification_results)}/{len(clarification_results)}\n")
    
    # Summary
    total_tests = len(valid_results) + len(invalid_results) + len(clarification_results)
    total_passed = sum(valid_results) + sum(invalid_results) + sum(clarification_results)
    print(f"Overall: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests

if __name__ == "__main__":
    print("Starting parallel validation tests...\n")
    asyncio.run(run_parallel_tests()) 