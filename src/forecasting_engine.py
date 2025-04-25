#!/usr/bin/env python3
"""
AI Superforecaster Engine

This module provides the core forecasting pipeline functionality.
It implements reference class forecasting, parameter research,
log-odds arithmetic, and red team analysis.
"""
import asyncio
import json
import datetime
import os

from agents import Runner, trace, InputGuardrailTripwireTriggered
from src.models import *
from src.agents import (background_info_agent, reference_class_agent, parameter_design_agent, 
                   parameter_researcher_agent, synthesis_agent, question_validator_agent,
                   question_clarifier_agent, forecast_orchestrator, red_team_agent)
from src.ui.cli import *
from src.utils.tools import WebSearchTool
from src.utils.forecast_math import logit, inv_logit
from src.utils.buffers import BufferManager
from src.utils.buffer_config import get_buffer_names

class ConsoleInputProvider:
    """Default input provider that uses console input"""
    def get_input(self, prompt):
        """Get input from the console"""
        try:
            return input(prompt)
        except EOFError:
            return ""

async def run_full_pipeline(user_question, buffers=None, input_provider=None):
    """
    Run the full forecasting pipeline on a user question.
    
    Args:
        user_question: The user's forecasting question
        buffers: BufferManager instance for output management
        input_provider: Object with get_input method for user input (console by default)
    """
    if buffers is None:
        buffers = BufferManager()
        init_buffers(buffers)
    
    if input_provider is None:
        input_provider = ConsoleInputProvider()
    
    # Main forecasting loop - will retry if validation fails
    while True:
        with trace("Forecasting workflow"):
            try:
                # Display the initial user question
                buffers.write("user", f"Question: {user_question}")
                
                # Start with the orchestrator for question validation and clarification
                display_processing_message()
                
                # First run the clarifier directly
                clarification_result = await Runner.run(
                    question_clarifier_agent,
                    user_question,
                )
                
                clarification = clarification_result.final_output_as(QuestionClarification)
                
                # If clarification needed, ask follow-up questions
                if clarification.needs_clarification and clarification.follow_up_questions:
                    display_clarification_request(clarification.follow_up_questions)
                    
                    # Use input provider instead of direct console input
                    prompt = "Please provide the requested information: "
                    additional_info = input_provider.get_input(prompt)
                    
                    if not additional_info:
                        # Handle empty response
                        buffers.write("user", "No additional input provided. Proceeding with default assumptions.")
                        additional_info = "Please continue with default assumptions."
                    else:
                        # Echo the input to the buffer
                        buffers.write("user", f"User provided: {additional_info}")
                    
                    # Run clarifier again with the additional information
                    clarification_result = await Runner.run(
                        question_clarifier_agent,
                        f"Original question: {user_question}\nAdditional information: {additional_info}",
                    )
                    clarification = clarification_result.final_output_as(QuestionClarification)
                
                # Use the clarified question for parameter estimation
                final_question = clarification.clarified_question

                # Add explicit validation of the clarified question before proceeding
                validation_result = await Runner.run(
                    question_validator_agent, 
                    final_question,
                )
                validation = validation_result.final_output_as(ForecastabilityCheck)

                if not validation.is_forecastable:
                    raise InputGuardrailTripwireTriggered(
                        f"Even after clarification, the question cannot be forecasted: {validation.reasoning}"
                    )

                # Continue with the forecast process
                display_forecasting_question(final_question)
                
                # Start the background info collection in parallel with reference class search
                display_reference_search_message()
                
                # Get current date for background info agent
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                background_info_task = asyncio.create_task(Runner.run(
                    background_info_agent,
                    f"Provide background information as of {current_date} relevant to the question: {final_question}",
                ))
                
                # Wait for background info first since it provides context for reference class selection
                background_info_result = await background_info_task
                background_info = background_info_result.final_output_as(BackgroundInfoOutput)
                
                # Display background info
                display_background_info(background_info)
                
                # Now search for reference classes with background context
                reference_class_prompt = f"""
                Find appropriate reference classes for the following question:
                {final_question}
                
                Current world context:
                {background_info.summary}
                
                Recent major events:
                {', '.join(background_info.major_recent_events[:3])}
                
                Key ongoing trends:
                {', '.join(background_info.key_trends[:3])}
                """
                
                reference_class_task = asyncio.create_task(Runner.run(
                    reference_class_agent,
                    reference_class_prompt,
                ))
                
                # Wait for reference class results
                reference_class_result = await reference_class_task
                reference_class_output = reference_class_result.final_output_as(ReferenceClassOutput)
                
                # Display all three reference classes
                display_reference_classes(reference_class_output)
                
                # Get the recommended reference class for further processing
                recommended_ref_class = reference_class_output.reference_classes[reference_class_output.recommended_class_index]
                
                # Next, determine the key parameters and include background info
                display_parameter_design_message()
                parameter_design_prompt = f"""
                Design parameters for the forecasting question: {final_question}
                
                Current world context:
                {background_info.summary}
                
                Recent major events:
                {', '.join(background_info.major_recent_events[:3])}
                
                Key ongoing trends:
                {', '.join(background_info.key_trends[:3])}
                
                Reference class: {recommended_ref_class.reference_class_description}
                Base rate: {recommended_ref_class.base_rate} [{recommended_ref_class.low} - {recommended_ref_class.high}]
                """
                
                parameter_design_task = asyncio.create_task(Runner.run(
                    parameter_design_agent,
                    parameter_design_prompt,
                ))
                
                # Process parameters while background info continues to gather
                parameter_design_result = await parameter_design_task
                parameter_design = parameter_design_result.final_output_as(ForecastParameters)
                
                # Now research each parameter in parallel
                display_parameter_research_message()
                
                # Print the parameters that will be researched
                display_parameters_to_research(parameter_design.parameters)
                
                # Create a summary of all parameters to provide context
                all_parameters_context = ""
                for i, param in enumerate(parameter_design.parameters, 1):
                    all_parameters_context += f"{i}. {param.name}: {param.description}\n"
                    all_parameters_context += f"   Scale: {param.scale_description}\n"
                    if param.interacts_with:
                        all_parameters_context += f"   Interacts with: {', '.join(param.interacts_with)}\n"
                    if param.interaction_description:
                        all_parameters_context += f"   Interaction: {param.interaction_description}\n"
                    all_parameters_context += "\n"
                
                async def research_parameter(param: ParameterMeta) -> ParameterSample:
                    """Run research for a single parameter"""
                    param_prompt = f"""
                    Research the following parameter for the question: {final_question}
                    
                    Current world context:
                    {background_info.summary}
                    
                    Recent major events:
                    {', '.join(background_info.major_recent_events[:3])}
                    
                    Key ongoing trends:
                    {', '.join(background_info.key_trends[:3])}
                    
                    Parameter: {param.name}
                    Description: {param.description}
                    Scale: {param.scale_description}
                    
                    Other parameters being researched:
                    {all_parameters_context}
                    
                    Based on your research, provide an estimate with 90% confidence interval.
                    """
                    result = await Runner.run(
                        parameter_researcher_agent,
                        param_prompt,
                    )
                    
                    # Convert the researcher's output to a ParameterSample
                    sample = result.final_output_as(ParameterSample)
                    sample.name = param.name  # Ensure the name matches
                    return sample
                
                # Run parameter research in parallel
                parameter_samples = await asyncio.gather(
                    *(research_parameter(param) for param in parameter_design.parameters)
                )
                
                # Print interim results from the parameter research
                display_parameter_estimates(parameter_samples)
                
                # Synthesize the final forecast
                display_synthesis_message()
                
                # Create the reference classes information for the synthesis prompt
                reference_classes_info = ""
                for i, ref_class in enumerate(reference_class_output.reference_classes):
                    is_recommended = i == reference_class_output.recommended_class_index
                    reference_classes_info += f"Reference Class {i+1}{' (RECOMMENDED)' if is_recommended else ''}:\n"
                    reference_classes_info += f"- Description: {ref_class.reference_class_description}\n"
                    reference_classes_info += f"- Base rate: {ref_class.base_rate} [{ref_class.low} - {ref_class.high}]\n"
                    reference_classes_info += f"- Sample size: {ref_class.sample_size}\n\n"
                
                synthesis_prompt = f"""
                Create a final forecast for: {final_question}
                
                Current context as of {background_info.current_date}:
                {background_info.summary}
                
                Multiple reference classes were identified:
                {reference_classes_info}
                Recommendation reasoning: {reference_class_output.selection_reasoning}
                
                Primary reference class used: {recommended_ref_class.reference_class_description}
                Base rate from primary reference class: {recommended_ref_class.base_rate} [{recommended_ref_class.low} - {recommended_ref_class.high}]
                
                Parameter estimates:
                {json.dumps([sample.model_dump() for sample in parameter_samples], indent=2)}
                
                Synthesize these into a final probability estimate. You may consider insights from all reference classes,
                but primarily use the recommended one as your starting point.
                
                Remember to:
                1. Convert the base rate to log-odds: L = logit(base_rate)
                2. Add each parameter's delta_log_odds to L
                3. Convert back to probability: P = inv_logit(L)
                4. Provide a 90% confidence interval and rationale
                """
                
                synthesis_task = asyncio.create_task(Runner.run(
                    synthesis_agent,
                    synthesis_prompt,
                ))
                
                # Get the final forecast
                synthesis_result = await synthesis_task
                final_forecast = synthesis_result.final_output_as(FinalForecast)
                
                # Display the final forecast
                display_final_forecast(final_forecast)
                
                # Record parameter contributions for log-odds details
                parameter_contributions = {sample.name: sample.delta_log_odds for sample in parameter_samples if sample.delta_log_odds is not None}
                
                # Calculate the final log-odds for display
                base_log_odds = logit(recommended_ref_class.base_rate)
                final_log_odds = base_log_odds + sum(parameter_contributions.values())
                
                # Display the log-odds calculation details
                display_parameter_calculation(
                    recommended_ref_class.base_rate,
                    parameter_contributions,
                    final_log_odds,
                    final_forecast.final_estimate
                )
                
                # Now run the red team challenge
                display_red_team_message()
                
                red_team_prompt = f"""
                Challenge the following forecast with the strongest possible counterarguments.
                
                Question: {final_question}
                
                Final probability estimate: {final_forecast.final_estimate} [{final_forecast.final_low} - {final_forecast.final_high}]
                
                Rationale: {final_forecast.rationale}
                
                Parameter insights:
                {json.dumps([sample.model_dump() for sample in parameter_samples], indent=2)}
                
                Primary reference class:
                {recommended_ref_class.reference_class_description} with base rate {recommended_ref_class.base_rate}
                
                Your task:
                1. Identify the strongest objection to this forecast
                2. Propose an alternative probability estimate
                3. Highlight key areas of disagreement
                4. Provide your rationale for the alternative view
                """
                
                red_team_task = asyncio.create_task(Runner.run(
                    red_team_agent,
                    red_team_prompt,
                ))
                
                # Get the red team challenge
                red_team_result = await red_team_task
                red_team_output = red_team_result.final_output_as(RedTeamOutput)
                
                # Display the red team challenge
                display_red_team_challenge(red_team_output)
                
                # Add completion message to user buffer
                buffers.write("user", "\n✓ Forecast completed!")
                buffers.write("user", "To run another forecast, use the 'Run New Forecast' button or type /rerun in CLI mode.")
                
                # Save this run
                question_slug = final_question.lower()[:30].replace(" ", "_").replace("?", "").replace(",", "")
                buffers.save_run(question_slug)
                
                return final_forecast
                
            except InputGuardrailTripwireTriggered as e:
                # Handle case where the question is not forecastable
                display_forecasting_error(e.message)
                
                # Ask if user wants to try again
                buffers.write("user", "\nWould you like to try again with a reformulated question? (yes/no): ")
                retry_response = input_provider.get_input("")
                
                if retry_response.lower() not in ["yes", "y"]:
                    # User doesn't want to retry
                    buffers.write("user", "Forecast canceled.")
                    return None
                
                # User wants to retry - ask for a new question
                buffers.write("user", "\nPlease enter a reformulated question: ")
                user_question = input_provider.get_input("")
                
                # If user provides empty input, exit
                if not user_question:
                    buffers.write("user", "No question provided. Forecast canceled.")
                    return None