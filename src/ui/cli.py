from src.models import (BackgroundInfoOutput, ReferenceClassOutput, 
                 ParameterMeta, ParameterSample, FinalForecast, RedTeamOutput)
from typing import List
from src.utils.forecast_math import logit, inv_logit
from src.utils.buffers import BufferManager

buffers: BufferManager | None = None

def init_buffers(bm: BufferManager):
    global buffers
    buffers = bm

def display_welcome():
    """Display welcome message."""
    buffers.write("user", "Welcome to the AI Superforecaster")
    buffers.write("user", "This system handles questions in the format: 'What is the probability that [specific event] happens by [specific timeframe]?'")
    buffers.write("user", "Examples:")
    buffers.write("user", "- What is the probability that Bitcoin will exceed $100,000 by the end of 2025?")
    buffers.write("user", "- What is the probability that SpaceX will launch humans to Mars before 2030?")
    buffers.write("user", "Please provide your forecasting question following this format.")

def display_processing_message():
    """Display processing message."""
    buffers.write("user", "\n=== Processing your question ===")

def display_clarification_request(follow_up_questions: List[str]):
    """Display request for clarification with follow-up questions."""
    buffers.write("user", "\nTo better understand your question, I need some clarification:")
    for i, question in enumerate(follow_up_questions, 1):
        buffers.write("user", f"{i}. {question}")
    buffers.write("user", "\nPlease provide this additional information:")

def display_forecasting_question(final_question: str):
    """Display the finalized forecasting question."""
    buffers.write("user", f"\nForecasting question: {final_question}")

def display_reference_search_message():
    """Display message about searching for reference classes."""
    buffers.write("user", "\n=== Finding relevant reference class and gathering background info ===")

def display_background_info(background_info: BackgroundInfoOutput):
    """Display background information about the current world context."""
    buffers.write("background", "=== Current World Context ===")
    buffers.write("background", f"Current date: {background_info.current_date}")
    buffers.write("background", f"Summary: {background_info.summary}")
    buffers.write("background", "\nRecent Major Events:")
    for event in background_info.major_recent_events[:3]:  # Show top 3 events
        buffers.write("background", f"- {event}")
    buffers.write("background", "\nKey Ongoing Trends:")
    for trend in background_info.key_trends[:3]:  # Show top 3 trends
        buffers.write("background", f"- {trend}")
    buffers.write("user", "✓ Background information collected.")

def display_reference_classes(reference_output: ReferenceClassOutput):
    """Display information about reference classes."""
    buffers.write("background", "=== REFERENCE CLASSES ===")
    for i, ref_class in enumerate(reference_output.reference_classes):
        is_recommended = i == reference_output.recommended_class_index
        buffers.write("background", f"\nReference Class {i+1}{' (RECOMMENDED)' if is_recommended else ''}:")
        buffers.write("background", f"Description: {ref_class.reference_class_description}")
        buffers.write("background", f"Base rate: {ref_class.base_rate} [{ref_class.low} - {ref_class.high}]")
        buffers.write("background", f"Sample size: {ref_class.sample_size} historical examples")
        buffers.write("background", f"Sources: {', '.join(ref_class.bibliography)}")
        buffers.write("background", f"Reasoning: {ref_class.reasoning}")
    
    buffers.write("background", f"\nRecommendation reasoning: {reference_output.selection_reasoning}")
    buffers.write("user", "✓ Reference classes identified.")

def display_parameter_design_message():
    """Display message about designing parameters."""
    buffers.write("user", "\n=== Designing key parameters ===")

def display_parameter_research_message():
    """Display message about researching parameters."""
    buffers.write("user", "\n=== Researching parameters (this may take a moment) ===")

def display_parameters_to_research(parameters: List[ParameterMeta]):
    """Display the parameters that will be researched."""
    buffers.write("background", "Parameters to research:")
    for i, param in enumerate(parameters, 1):
        buffers.write("background", f"{i}. {param.name}: {param.description}")
        buffers.write("background", f"   Scale: {param.scale_description}")

def display_parameter_estimates(samples: List[ParameterSample]):
    """Display the parameter estimates."""
    buffers.write("background", "=== Parameter estimates ===")
    for sample in samples:
        # Display parameter values in parameters section
        buffers.write("parameters", f"{sample.name}: {sample.value} [{sample.low} - {sample.high}]")
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
            buffers.write("parameters", f"  Log-odds: {sign}{sample.delta_log_odds:.3f} ({strength} {'positive' if sample.delta_log_odds > 0 else 'negative'} evidence)")
        
        # Keep parameter definition in background section
        buffers.write("background", f"{sample.name}")
        buffers.write("background", f"  Sources: {', '.join(sample.sources)}")
    
    buffers.write("user", "✓ Parameter research completed.")

def display_synthesis_message():
    """Display message about creating the final forecast."""
    buffers.write("user", "\n=== Creating final forecast ===")

def display_final_forecast(forecast: FinalForecast):
    """Display the final forecast."""
    buffers.write("report", "=== FINAL FORECAST ===")
    buffers.write("report", f"Question: {forecast.question}\n")
    buffers.write("report", f"Probability: {forecast.final_estimate*100:.1f}% [{forecast.final_low*100:.1f}% - {forecast.final_high*100:.1f}%]")
    buffers.write("report", f"Starting base rate: {forecast.base_rate*100:.1f}%")
    buffers.write("report", f"Key drivers: {', '.join(forecast.key_parameters)}")
    buffers.write("report", f"Rationale: {forecast.rationale}\n")
    buffers.write("user", f"✓ Final forecast: {forecast.final_estimate*100:.1f}% [{forecast.final_low*100:.1f}% - {forecast.final_high*100:.1f}%]")

def display_red_team_message():
    """Display message about running red team challenge."""
    buffers.write("user", "\n=== Running red team challenge ===")

def display_red_team_challenge(red_team: RedTeamOutput):
    """Display the red team challenge."""
    buffers.write("report", "=== RED TEAM CHALLENGE ===")
    buffers.write("report", f"Strongest objection: {red_team.strongest_objection}")
    buffers.write("report", f"Alternative estimate: {red_team.alternate_estimate*100:.1f}% [{red_team.alternate_low*100:.1f}% - {red_team.alternate_high*100:.1f}%]")
    buffers.write("report", f"Key disagreements: {', '.join(red_team.key_disagreements)}")
    buffers.write("report", f"Rationale: {red_team.rationale}")
    buffers.write("user", "✓ Red team analysis completed.")

def display_forecasting_error(reasoning: str):
    """Display error message when a question cannot be forecasted."""
    buffers.write("user", "\n=== CANNOT PROCESS THIS QUESTION ===")
    buffers.write("user", f"Reason: {reasoning}")
    buffers.write("user", "\nYour question must follow the format: 'What is the probability that [specific event] happens by [specific timeframe]?'")
    buffers.write("user", "\nExamples of valid questions:")
    buffers.write("user", "- What is the probability that Bitcoin will exceed $100,000 by the end of 2025?")
    buffers.write("user", "- What is the probability that SpaceX will launch humans to Mars before 2030?")
    buffers.write("user", "- What is the probability that renewable energy will provide >50% of global electricity by 2035?")
    buffers.write("user", "\nYou'll be given a chance to reformulate your question.")

def display_parameter_calculation(base_rate, parameter_contributions, final_log_odds, final_prob, 
                                adjustment_factor=None, conservatism_applied=False):
    """Display the parameter calculation details."""
    L_base = logit(base_rate)
    
    buffers.write("parameters", "=== PARAMETER CALCULATION ===")
    buffers.write("parameters", f"Base rate: {base_rate*100:.1f}% → log-odds: {L_base:.3f}")
    
    # Show each parameter's contribution, sorted by magnitude
    buffers.write("parameters", "\nParameter contributions:")
    total_shift = 0
    running_log_odds = L_base
    running_prob = base_rate
    
    for name, delta in sorted(parameter_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if delta > 0 else ""
        buffers.write("parameters", f"  {name}: {sign}{delta:.3f}")
        
        # Calculate the probability impact
        running_log_odds += delta
        new_prob = inv_logit(running_log_odds)
        prob_delta = (new_prob - running_prob) * 100
        buffers.write("parameters", f"    Probability shift: {running_prob*100:.1f}% → {new_prob*100:.1f}% ({'+' if prob_delta > 0 else ''}{prob_delta:.1f}%)")
        running_prob = new_prob
        total_shift += abs(delta)
    
    # Show any calibration adjustments
    if adjustment_factor and adjustment_factor < 1.0:
        buffers.write("parameters", f"\nCalibration adjustment: ×{adjustment_factor:.2f} (scaling down large shifts)")
    
    if conservatism_applied:
        buffers.write("parameters", "\nSuperforecaster conservatism applied to extreme probability")
    
    buffers.write("parameters", f"\nFinal log-odds: {final_log_odds:.3f} → Probability: {final_prob*100:.1f}%")
    
    # Interpret the total shift
    buffers.write("parameters", f"\nTotal log-odds impact: {total_shift:.2f}")
    if total_shift < 1.0:
        buffers.write("parameters", "✓ Conservative shift - typical of careful superforecasters")
    elif total_shift < 2.0:
        buffers.write("parameters", "✓ Moderate shift - within normal superforecaster range")
    elif total_shift < 3.0:
        buffers.write("parameters", "! Large shift - requires strong evidence to justify")
    else:
        buffers.write("parameters", "!! Extreme shift - superforecasters rarely make shifts this large")
    
    buffers.write("parameters", "----------------------------------------")
    buffers.write("user", "✓ Parameter calculation completed.") 