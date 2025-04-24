#!/usr/bin/env python3
"""
AI Superforecaster CLI

This is the main entry point for the AI Superforecaster command-line interface.
It implements a forecasting pipeline using reference class forecasting, parameter research,
log-odds arithmetic, and red team analysis.

This application uses a buffer system that separates content into different views:
- user: Interactive I/O and status messages
- background: Reference classes and parameter research
- logodds: Calculation details for log-odds
- report: Final forecast and red team analysis

Usage:
  python main.py                # Interactive mode
  echo "question" | python main.py  # Non-interactive mode

Commands (during interactive session):
  /rerun - Start a new forecast
  /view - View buffer contents
  /quit - Exit the application
  /gui - Launch the buffer viewer GUI

For a better experience with visual buffer display, use:
  python run_with_buffers.py
"""
import asyncio
import json
import datetime
import os
import glob

from agents import Runner, trace, InputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
from src.models import *
from src.agents import (background_info_agent, reference_class_agent, parameter_design_agent, 
                   parameter_researcher_agent, synthesis_agent, question_validator_agent,
                   question_clarifier_agent, forecast_orchestrator, red_team_agent)
from src.ui.cli import *
from src.utils.tools import WebSearchTool
from src.utils.forecast_math import logit, inv_logit
from src.utils.buffers import BufferManager

async def run_full_pipeline(user_question, buffers=None):
    """Run the full forecasting pipeline on a user question."""
    if buffers is None:
        buffers = BufferManager()
        init_buffers(buffers)
    
    with trace("Forecasting workflow"):
        try:
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
                try:
                    additional_info = input("> ")
                except EOFError:
                    # Handle case when running in non-interactive mode
                    buffers.write("user", "No additional input available (non-interactive mode). Proceeding with default assumptions.")
                    additional_info = "Please continue with default assumptions."
                
                # Run clarifier again with the additional information
                clarification_result = await Runner.run(
                    question_clarifier_agent,
                    f"Original question: {user_question}\nAdditional information: {additional_info}",
                )
                clarification = clarification_result.final_output_as(QuestionClarification)
            
            # Use the clarified question for parameter estimation
            final_question = clarification.clarified_question
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
            3. Calculate final_prob = inv_logit(L)
            """
            
            synthesis_result = await Runner.run(
                synthesis_agent,
                synthesis_prompt,
            )
            
            final_forecast = synthesis_result.final_output_as(FinalForecast)
            
            # Apply log-odds calculation
            L = logit(recommended_ref_class.base_rate)
            L_base = L  # Store the base log-odds for display
            
            # Collect valid parameter adjustments and their impacts
            valid_params = [(p.name, p.delta_log_odds) for p in parameter_samples if p.delta_log_odds is not None]
            parameter_contributions = {}
            
            # Check for excessively large cumulative shifts
            total_shift = sum(abs(delta) for _, delta in valid_params)
            scaling_factor = 1.0
            conservatism_applied = False
            
            if total_shift > 4.0:
                # Scale down all contributions proportionally if the total is too extreme
                scaling_factor = 4.0 / total_shift
            
            # Apply log-odds contributions (with possible scaling)
            for name, delta in valid_params:
                adjusted_delta = delta * scaling_factor
                parameter_contributions[name] = adjusted_delta
                L += adjusted_delta
            
            # Apply superforecaster conservatism
            # Extreme projections should be tempered unless evidence is overwhelming
            if abs(L) > 3.0:
                conservatism_factor = 0.7  # Reduce extreme shifts
                L = L * conservatism_factor
                conservatism_applied = True
            
            final_prob = inv_logit(L)
            
            # Update the final forecast with log-odds calculation
            final_forecast.final_estimate = final_prob
            
            # Calculate confidence interval (more reasonable approach than fixed ±0.15)
            # Superforecasters use narrower intervals for extreme probabilities
            if final_prob > 0.9 or final_prob < 0.1:
                ci_width = 0.08  # Narrower CI for extreme probabilities
            elif final_prob > 0.8 or final_prob < 0.2:
                ci_width = 0.12  # Medium CI for fairly confident probabilities
            else:
                ci_width = 0.15  # Wider CI for moderate probabilities
            
            final_forecast.final_low = max(0.0, final_prob - ci_width)
            final_forecast.final_high = min(1.0, final_prob + ci_width)

            # Display the final results
            display_final_forecast(final_forecast)
            
            # Display the log-odds calculation
            display_log_odds_calculation(
                base_rate=recommended_ref_class.base_rate,
                parameter_contributions=parameter_contributions,
                final_log_odds=L,
                final_prob=final_prob,
                adjustment_factor=scaling_factor,
                conservatism_applied=conservatism_applied
            )
            
            # Run red team analysis
            display_red_team_message()
            
            # Create a full context for the red team including all reference classes
            reference_classes_info = ""
            for i, ref_class in enumerate(reference_class_output.reference_classes):
                is_recommended = i == reference_class_output.recommended_class_index
                reference_classes_info += f"Reference Class {i+1}{' (RECOMMENDED)' if is_recommended else ''}:\n"
                reference_classes_info += f"- Description: {ref_class.reference_class_description}\n"
                reference_classes_info += f"- Base rate: {ref_class.base_rate} [{ref_class.low} - {ref_class.high}]\n"
                reference_classes_info += f"- Sample size: {ref_class.sample_size}\n"
                reference_classes_info += f"- Reasoning: {ref_class.reasoning}\n\n"
            
            # Create parameter info for red team
            parameters_info = ""
            for sample in parameter_samples:
                parameters_info += f"{sample.name}: {sample.value} [{sample.low} - {sample.high}]\n"
                parameters_info += f"- Reasoning: {sample.reasoning}\n\n"
            
            red_team_prompt = f"""
            Challenge the following forecast:
            
            Question: {final_question}
            
            Current world context:
            {background_info.summary}
            
            Reference classes considered:
            {reference_classes_info}
            
            Parameter estimates used:
            {parameters_info}
            
            Final forecast:
            - Estimate: {final_forecast.final_estimate} [{final_forecast.final_low} - {final_forecast.final_high}]
            - Base rate used: {final_forecast.base_rate}
            - Key parameters: {', '.join(final_forecast.key_parameters)}
            - Rationale: {final_forecast.rationale}
            
            Provide a strong alternative view. Consider whether different reference classes should have been used,
            if parameters were incorrectly estimated, or if important factors were overlooked.
            """
            
            red_team_result = await Runner.run(
                red_team_agent,
                red_team_prompt,
            )
            
            red_team = red_team_result.final_output_as(RedTeamOutput)
            
            display_red_team_challenge(red_team)
            buffers.write("user", "✓ Pipeline finished.")
            buffers.write("user", "Type /rerun to forecast a new question, /view to see all buffers, or /quit to exit.")
            
            # Save run results to files
            question_slug = user_question.replace(" ", "_")[:20]
            buffers.save_run(prefix=question_slug)
            
            return final_forecast, red_team
            
        except InputGuardrailTripwireTriggered as e:
            check = e.guardrail_result.output.output_info
            display_forecasting_error(check.reasoning)
            return None, None

async def main():
    # Initialize buffer manager
    buffers = BufferManager()
    init_buffers(buffers)
    
    # Clear any previous latest files
    for file in glob.glob("runs/latest_*.txt"):
        try:
            os.remove(file)
        except:
            pass  # Ignore errors
    
    display_welcome()
    # draw_graph(forecast_orchestrator, filename="forecast_orchestrator")
    
    try:
        user_question = input("> ")
    except EOFError:
        buffers.write("user", "No input detected. Exiting.")
        return
    
    await run_full_pipeline(user_question, buffers)
    
    # Add REPL for agentic loop
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd in {"/quit", "q"}:
                break
            elif cmd in {"/rerun", "r"}:
                # Clear latest files before starting a new run
                for file in glob.glob("runs/latest_*.txt"):
                    try:
                        os.remove(file)
                    except:
                        pass  # Ignore errors
                
                buffers.write("user", "\n=== Starting new forecast ===")
                try:
                    question = input("Enter new/revised question:\n> ")
                except EOFError:
                    buffers.write("user", "No input provided for new question. Returning to main menu.")
                    continue
                await run_full_pipeline(question, buffers)
            elif cmd in {"/view", "v"}:
                try:
                    section = input("Enter buffer to view (or 'all'): ").strip().lower()
                except EOFError:
                    print("No input provided. Showing all buffers.")
                    section = "all"
                
                if section == "all":
                    for s in buffers.sections:
                        print(f"\n\n=== {s.upper()} BUFFER ===")
                        print(buffers.dump(s))
                elif section in buffers.sections:
                    print(f"\n\n=== {section.upper()} BUFFER ===")
                    print(buffers.dump(section))
                else:
                    print(f"Buffer '{section}' not found. Available buffers: {', '.join(buffers.sections)}")
            elif cmd == "/gui":
                print("Starting GUI buffer viewer in a separate window...")
                import subprocess
                try:
                    subprocess.Popen(["python", "simple_buffer_viewer.py"])
                except Exception as e:
                    print(f"Error starting GUI: {str(e)}")
            else:
                print("Commands: /rerun, /view, /quit, /gui")
        except EOFError:
            # Exit if we're in non-interactive mode
            break

if __name__ == "__main__":
    asyncio.run(main())