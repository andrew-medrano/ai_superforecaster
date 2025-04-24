from src.models import (BackgroundInfoOutput, ReferenceClassOutput, 
                 ParameterMeta, ParameterSample, FinalForecast, RedTeamOutput)
from typing import List
from src.utils.forecast_math import logit, inv_logit

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
        if sample.delta_log_odds is not None:
            sign = "+" if sample.delta_log_odds > 0 else ""
            if abs(sample.delta_log_odds) < 0.2:
                strength = "very weak"
            elif abs(sample.delta_log_odds) < 0.4:
                strength = "weak"
            elif abs(sample.delta_log_odds) < 0.7:
                strength = "moderate"
            elif abs(sample.delta_log_odds) <= 1.0:
                strength = "strong"
            else:
                strength = "very strong"
            print(f"  Log-odds: {sign}{sample.delta_log_odds:.3f} ({strength} {'positive' if sample.delta_log_odds > 0 else 'negative'} evidence)")
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

def display_log_odds_calculation(base_rate, parameter_contributions, final_log_odds, final_prob, 
                                adjustment_factor=None, conservatism_applied=False):
    """Display the log-odds calculation details."""
    L_base = logit(base_rate)
    
    print("\n=== LOG-ODDS CALCULATION ===")
    print(f"Base rate: {base_rate*100:.1f}% → log-odds: {L_base:.3f}")
    
    # Show each parameter's contribution, sorted by magnitude
    print("\nParameter contributions:")
    total_shift = 0
    running_log_odds = L_base
    running_prob = base_rate
    
    for name, delta in sorted(parameter_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if delta > 0 else ""
        print(f"  {name}: {sign}{delta:.3f}")
        
        # Calculate the probability impact
        running_log_odds += delta
        new_prob = inv_logit(running_log_odds)
        prob_delta = (new_prob - running_prob) * 100
        print(f"    Probability shift: {running_prob*100:.1f}% → {new_prob*100:.1f}% ({'+' if prob_delta > 0 else ''}{prob_delta:.1f}%)")
        running_prob = new_prob
        total_shift += abs(delta)
    
    # Show any calibration adjustments
    if adjustment_factor and adjustment_factor < 1.0:
        print(f"\nCalibration adjustment: ×{adjustment_factor:.2f} (scaling down large shifts)")
    
    if conservatism_applied:
        print("\nSuperforecaster conservatism applied to extreme probability")
    
    print(f"\nFinal log-odds: {final_log_odds:.3f} → Probability: {final_prob*100:.1f}%")
    
    # Interpret the total shift
    print(f"\nTotal log-odds impact: {total_shift:.2f}")
    if total_shift < 1.0:
        print("✓ Conservative shift - typical of careful superforecasters")
    elif total_shift < 2.0:
        print("✓ Moderate shift - within normal superforecaster range")
    elif total_shift < 3.0:
        print("! Large shift - requires strong evidence to justify")
    else:
        print("!! Extreme shift - superforecasters rarely make shifts this large")
    
    print("----------------------------------------") 