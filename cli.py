from models import (BackgroundInfoOutput, ReferenceClassOutput, 
                 ParameterMeta, ParameterSample, FinalForecast, RedTeamOutput)
from typing import List

def display_welcome():
    """Display welcome message."""
    print("Welcome to the AI Superforecaster")
    print("What would you like to forecast? Please describe the question or scenario.")

def display_processing_message():
    """Display processing message."""
    print("\n=== Processing your question ===")

def display_clarification_request(follow_up_questions: List[str]):
    """Display request for clarification with follow-up questions."""
    print("\nTo better understand your question, I need some clarification:")
    for i, question in enumerate(follow_up_questions, 1):
        print(f"{i}. {question}")
    print("\nPlease provide this additional information:")

def display_forecasting_question(final_question: str):
    """Display the finalized forecasting question."""
    print(f"\nForecasting question: {final_question}")

def display_reference_search_message():
    """Display message about searching for reference classes."""
    print("\n=== Finding relevant reference class and gathering background info ===")

def display_background_info(background_info: BackgroundInfoOutput):
    """Display background information about the current world context."""
    print("\n=== Current World Context ===")
    print(f"Current date: {background_info.current_date}")
    print(f"Summary: {background_info.summary}")
    print("\nRecent Major Events:")
    for event in background_info.major_recent_events[:3]:  # Show top 3 events
        print(f"- {event}")
    print("\nKey Ongoing Trends:")
    for trend in background_info.key_trends[:3]:  # Show top 3 trends
        print(f"- {trend}")

def display_reference_classes(reference_output: ReferenceClassOutput):
    """Display information about reference classes."""
    print("\n=== REFERENCE CLASSES ===")
    for i, ref_class in enumerate(reference_output.reference_classes):
        is_recommended = i == reference_output.recommended_class_index
        print(f"\nReference Class {i+1}{' (RECOMMENDED)' if is_recommended else ''}:")
        print(f"Description: {ref_class.reference_class_description}")
        print(f"Base rate: {ref_class.base_rate} [{ref_class.low} - {ref_class.high}]")
        print(f"Sample size: {ref_class.sample_size} historical examples")
        print(f"Sources: {', '.join(ref_class.bibliography)}")
        print(f"Reasoning: {ref_class.reasoning}")
    
    print(f"\nRecommendation reasoning: {reference_output.selection_reasoning}")

def display_parameter_design_message():
    """Display message about designing parameters."""
    print("\n=== Designing key parameters ===")

def display_parameter_research_message():
    """Display message about researching parameters."""
    print("\n=== Researching parameters (this may take a moment) ===")

def display_parameters_to_research(parameters: List[ParameterMeta]):
    """Display the parameters that will be researched."""
    print("Parameters to research:")
    for i, param in enumerate(parameters, 1):
        print(f"{i}. {param.name}: {param.description}")
        print(f"   Scale: {param.scale_description}")

def display_parameter_estimates(samples: List[ParameterSample]):
    """Display the parameter estimates."""
    print("\n=== Parameter estimates ===")
    for sample in samples:
        print(f"{sample.name}: {sample.value} [{sample.low} - {sample.high}]")
        print(f"  Sources: {', '.join(sample.sources)}")

def display_synthesis_message():
    """Display message about creating the final forecast."""
    print("\n=== Creating final forecast ===")

def display_final_forecast(forecast: FinalForecast):
    """Display the final forecast."""
    print("\n=== FINAL FORECAST ===")
    print(f"Question: {forecast.question}\n")
    print(f"Probability: {forecast.final_estimate*100:.1f}% [{forecast.final_low*100:.1f}% - {forecast.final_high*100:.1f}%]")
    print(f"Starting base rate: {forecast.base_rate*100:.1f}%")
    print(f"Key drivers: {', '.join(forecast.key_parameters)}")
    print(f"Rationale: {forecast.rationale}\n")

def display_red_team_message():
    """Display message about running red team challenge."""
    print("\n=== Running red team challenge ===")

def display_red_team_challenge(red_team: RedTeamOutput):
    """Display the red team challenge."""
    print("=== RED TEAM CHALLENGE ===")
    print(f"Strongest objection: {red_team.strongest_objection}")
    print(f"Alternative estimate: {red_team.alternate_estimate*100:.1f}% [{red_team.alternate_low*100:.1f}% - {red_team.alternate_high*100:.1f}%]")
    print(f"Key disagreements: {', '.join(red_team.key_disagreements)}")
    print(f"Rationale: {red_team.rationale}")

def display_forecasting_error(reasoning: str):
    """Display error message when a question cannot be forecasted."""
    print("\n=== CANNOT PROCESS THIS QUESTION ===")
    print(f"Reason: {reasoning}")
    print("\nPlease try asking a question about a future event or trend that can be forecasted.") 