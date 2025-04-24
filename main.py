import asyncio
import json
import datetime

from agents import Runner, trace, InputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
from models import *
from agent_definitions import (background_info_agent, reference_class_agent, parameter_design_agent, 
                   parameter_researcher_agent, synthesis_agent, question_validator_agent,
                   question_clarifier_agent, forecast_orchestrator, red_team_agent)
from cli import *
from tools import WebSearchTool

async def main():
    display_welcome()
    draw_graph(forecast_orchestrator, filename="forecast_orchestrator")
    
    user_question = input("> ")
    
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
                additional_info = input("> ")
                
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
            """
            
            synthesis_result = await Runner.run(
                synthesis_agent,
                synthesis_prompt,
            )
            
            final_forecast = synthesis_result.final_output_as(FinalForecast)

            # Display the final results
            display_final_forecast(final_forecast)
            
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
            
        except InputGuardrailTripwireTriggered as e:
            check = e.guardrail_result.output.output_info
            display_forecasting_error(check.reasoning)
            return


if __name__ == "__main__":
    asyncio.run(main())