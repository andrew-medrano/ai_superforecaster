import asyncio
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import datetime

from agents import Agent, Runner, trace, ItemHelpers, input_guardrail, GuardrailFunctionOutput, RunContextWrapper, InputGuardrailTripwireTriggered
from agents.tool import WebSearchTool
from agents import ModelSettings
from agents.extensions.visualization import draw_graph

class ParameterMeta(BaseModel):
    """Parameter metadata defining what to estimate"""
    name: str = Field(description="Name of the parameter")
    description: str = Field(description="Description of what this parameter represents")
    scale_description: str = Field(description="Description of measurement scale (e.g., '0-10 where 0 means no willingness and 10 means complete commitment')")
    interacts_with: Optional[List[str]] = Field(
        description="Names of other parameters this parameter interacts with", 
    )
    interaction_type: Optional[Literal["none", "additive", "multiplicative", "weak_exponential", "strong_exponential"]] = Field(
        description="How this parameter interacts with other parameters",
    )
    interaction_description: Optional[str] = Field(
        description="Description of how this parameter interacts with others",
    )

class ParameterSample(BaseModel):
    """Parameter estimates with value and confidence interval"""
    name: str = Field(description="Name of the parameter (must match a ParameterMeta)")
    value: float = Field(description="Estimated value")
    low: float = Field(description="Lower bound of 90% confidence interval")
    high: float = Field(description="Upper bound of 90% confidence interval")
    reasoning: str = Field(description="Reasoning for this estimate")
    sources: List[str] = Field(description="Sources supporting this estimate", default_factory=list)

class ReferenceClass(BaseModel):
    """Output format for a single reference class"""
    base_rate: float = Field(description="Historical base rate/frequency of similar events")
    low: float = Field(description="Lower bound of 90% confidence interval for base rate")
    high: float = Field(description="Upper bound of 90% confidence interval for base rate")
    reference_class_description: str = Field(description="Description of the reference class used")
    sample_size: int = Field(description="Size of the reference class (number of historical examples)")
    bibliography: List[str] = Field(description="Citations for historical data sources")
    reasoning: str = Field(description="Reasoning for selecting this reference class")

class ReferenceClassOutput(BaseModel):
    """Output format for reference class forecasting with multiple classes"""
    reference_classes: List[ReferenceClass] = Field(description="List of reference classes with their base rates")
    recommended_class_index: int = Field(description="Index of the recommended reference class to use (0-based)")
    selection_reasoning: str = Field(description="Reasoning for recommending the primary reference class")

class ForecastParameters(BaseModel):
    """Output format for forecast parameters"""
    question: str = Field(description="Original forecasting question")
    parameters: List[ParameterMeta] = Field(description="Parameter specifications to estimate")
    additional_considerations: List[str] = Field(description="Additional factors to consider")


class FinalForecast(BaseModel):
    """Final forecast output combining base rate with parameter adjustments"""
    question: str = Field(description="Forecast question")
    final_estimate: float = Field(description="Final probability estimate")
    final_low: float = Field(description="Lower bound of 90% confidence interval")
    final_high: float = Field(description="Upper bound of 90% confidence interval")
    base_rate: float = Field(description="Original base rate used")
    key_parameters: List[str] = Field(description="Names of most influential parameters")
    rationale: str = Field(description="Summary rationale for forecast")
    parameter_samples: List[ParameterSample] = Field(description="All parameter samples used")


class ForecastabilityCheck(BaseModel):
    """Output format for checking if a question is forecastable"""
    is_forecastable: bool = Field(description="Whether the question can be reasonably forecasted")
    reasoning: str = Field(description="Reasoning for the decision")


class QuestionClarification(BaseModel):
    """Output format for question clarification"""
    original_question: str = Field(description="The original question")
    clarified_question: str = Field(description="The clarified, time-bound question")
    follow_up_questions: List[str] = Field(description="Follow-up questions if clarification is needed")
    needs_clarification: bool = Field(description="Whether the question needs clarification")


class RedTeamOutput(BaseModel):
    """Output from red team challenge to the forecast"""
    strongest_objection: str = Field(description="Strongest objection to the forecast")
    alternate_estimate: float = Field(description="Alternative probability estimate")
    alternate_low: float = Field(description="Lower bound of alt 90% confidence interval")
    alternate_high: float = Field(description="Upper bound of alt 90% confidence interval")
    key_disagreements: List[str] = Field(description="Key points of disagreement with original forecast")
    rationale: str = Field(description="Rationale for the alternative view")


class BackgroundInfoOutput(BaseModel):
    """Output with current context about the world to account for model knowledge cutoff"""
    current_date: str = Field(description="Current date in YYYY-MM-DD format")
    major_recent_events: List[str] = Field(description="List of significant recent events with descriptions")
    key_trends: List[str] = Field(description="Key ongoing trends relevant to forecasting")
    notable_changes: List[str] = Field(description="Notable changes since the model's knowledge cutoff")
    summary: str = Field(description="Brief summary of the current global context")


background_info_agent = Agent(
    name="Background Information Provider",
    instructions="""You provide up-to-date context about the current state of the world to help with forecasting.

Your task is to:
1. Search for recent major events and developments that may impact forecasts
2. Identify key global trends that are ongoing
3. Note any significant changes since the model's knowledge cutoff
4. Provide a brief summary of the current global context

Focus on:
- Major geopolitical developments
- Significant economic indicators and trends
- Technological breakthroughs or advancements
- Social and political shifts
- Environmental changes or events

Be objective and factual. Prioritize information that would be most relevant for forecasting.""",
    tools=[WebSearchTool()],
    output_type=BackgroundInfoOutput,
    model="gpt-4.1-mini",
)


reference_class_agent = Agent(
    name="Reference Class Finder",
    instructions="""You identify appropriate reference classes for a forecasting question and determine the historical base rates.

For any forecasting question:
1. Generate THREE different reference classes that could be applied to this question
2. For each reference class:
   - Identify what makes this reference class appropriate
   - Search for historical data on this reference class using the web search tool
   - Determine the base rate (frequency of occurrence) within this reference class
   - Provide a 90% confidence interval around this base rate
   - Document your sources with at least 1-2 web searches per reference class
   - Explain your reasoning for selecting this reference class
3. Recommend which of the three reference classes should be the primary one to use
4. Provide reasoning for your recommendation

Make sure the reference classes are genuinely different from each other to provide diverse perspectives on the question.

Focus on finding relevant historical analogues. Be precise about:
- Sample size (how many cases in your reference class)
- Time period covered
- Any adjustments needed for this specific case
- Quality and limitations of available data

Example:
"What is the probability that China will invade Taiwan by 2030?"
Reference Classes:
1. Military invasions of claimed territories by nuclear powers (1950-present)
2. Historical Chinese military actions to enforce territorial claims
3. Militarized disputes between countries with strong economic ties

Aim for objective, quantifiable reference classes whenever possible.""",
    tools=[WebSearchTool()],
    output_type=ReferenceClassOutput,
    model="gpt-4.1-mini",
)


parameter_design_agent = Agent(
    name="Parameter Designer",
    instructions="""You design the key parameters needed to estimate a forecasting question given a reference class and base rate.

For each forecasting question:
1. Start with first principles thinking to identify what fundamentally matters to this question
2. Identify 5 key parameters that:
   - Cover different aspects of the forecast (e.g. market, technology, human factors)
   - Are measurable and as objective as possible
   - Together give a comprehensive view of what drives the outcome
   - Would adjust the base rate up or down
3. For each parameter:
   - Define a clear, measurable name and description
   - Provide a clear scale description explaining what the values mean
   - Identify how this parameter interacts with others (additive, multiplicative, etc.)
   - Do NOT provide numeric estimates yet - focus on structure only

Focus on parameters that are both measurable and meaningful for modeling the forecast outcome.

Good example parameters:
- What is the probability that China invades Taiwan by 2030?
  * China's military capability growth (1-10 scale)
  * Taiwan's defense capabilities (1-10 scale)
  * US willingness to intervene (0-100% probability)
  * Economic interdependence (0-100 index)
  * CCP domestic political stability (1-10 scale)

The goal is to define parameters that:
1. Have clear, measurable definitions
2. Drive the outcome in an understandable way
3. Can be researched separately""",
    output_type=ForecastParameters,
    tools=[WebSearchTool()],
    model="gpt-4.1",
)


parameter_researcher_agent = Agent(
    name="Parameter Researcher",
    instructions="""You research a specific forecasting parameter to provide an evidence-based estimate.

For the given parameter:
1. Run at least 3 web searches to gather relevant data and evidence
2. Based on the data, estimate:
   - The most likely value
   - A 90% confidence interval (10th and 90th percentile)
3. Cite your sources clearly
4. Provide concise reasoning for your estimate

Focus on finding objective data wherever possible. If data is limited, use analogous situations or expert judgments from reputable sources.

Keep your reasoning concise and data-driven.""",
    tools=[WebSearchTool()],
    output_type=ParameterSample,
    model="gpt-4.1-mini",
)


synthesis_agent = Agent(
    name="Forecast Synthesizer",
    instructions="""You create the final forecast by combining the base rate with parameter estimates.

Starting with:
1. A base rate from reference class forecasting
2. Estimates for 5 key parameters with confidence intervals

Your task:
1. Start with the reference class base rate as your prior
2. Adjust this probability based on the parameter estimates:
   - For additive interactions, apply direct adjustments
   - For multiplicative interactions, apply as multipliers
   - For exponential interactions, apply appropriate transformations
3. Determine how each parameter should shift the base rate
4. Calculate a final probability estimate with 90% confidence interval
5. Identify the 2 most influential parameters
6. Write a clear, one-paragraph rationale that references these key parameters

Your final forecast should strike a balance between the outside view (base rate) and the inside view (parameter adjustments). Be explicit about how much weight you give to the base rate versus specific parameters.""",
    model="gpt-4.1",
    output_type=FinalForecast,
)


question_validator_agent = Agent(
    name="Question Validator",
    instructions="""You determine if a question can be reasonably forecasted.

Valid forecast questions:
- Ask about future events, trends, or outcomes
- Can be measured or verified objectively
- Have a defined timeframe or condition
- Deal with real-world possibilities

Invalid forecast questions:
- Basic factual queries like "What is the color of the sky?"
- Questions about personal preferences or opinions
- Questions with no objective answer
- Questions about fictional scenarios with no real-world basis
- Questions about the past or present (unless projecting forward)

Always provide clear reasoning for your decision.
""",
    output_type=ForecastabilityCheck,
)


question_clarifier_agent = Agent(
    name="Question Clarifier",
    instructions="""You help users clarify vague forecasting questions into concrete, time-bound questions.

For any input question:
1. Identify if the question is already well-defined with a clear timeframe and measurable outcome
2. If not, suggest a more specific formulation with:
   - A clear timeframe (e.g., "by 2030" or "within the next 3 years")
   - Measurable criteria for resolution
   - Objective conditions

Examples:
- "Will China invade Taiwan?" becomes "What is the probability that China will initiate a military invasion of Taiwan before December 31, 2030?"
- "Will AI take my job?" becomes "What is the probability that at least 30% of [specific job type] roles will be automated by AI by 2028?"

If you need specific information from the user to properly clarify the question, list specific follow-up questions.
""",
    output_type=QuestionClarification,
    model="gpt-4.1-mini",
)


@input_guardrail
async def forecastability_guardrail(
    context: RunContextWrapper[None], agent: Agent, input: str
) -> GuardrailFunctionOutput:
    """Checks if the user's query is a valid forecastable question."""
    result = await Runner.run(question_validator_agent, input, context=context.context)
    check = result.final_output_as(ForecastabilityCheck)
    
    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=not check.is_forecastable,
    )

red_team_agent = Agent(
    name="Red Team Challenger",
    instructions="""You challenge the forecast by providing the strongest possible counterargument.

Given a complete forecast with base rate, parameters, and final estimate:
1. Identify the weakest assumptions in the forecast
2. Provide the strongest objection to the forecast
3. Offer an alternative estimate with confidence interval
4. Explain key points of disagreement
5. Provide a concise rationale for your alternative view

Focus on:
- Overlooked reference classes that might be more appropriate
- Alternative interpretations of the same evidence
- Important missing parameters
- Ways the parameter interactions might be misunderstood
- Cognitive biases that might be affecting the forecast

Your goal is NOT to be contrarian for its own sake, but to provide a genuinely strong alternative view that could improve the forecast.""",
    output_type=RedTeamOutput,
    tools=[WebSearchTool()],
    model="gpt-4.1",
)


forecast_orchestrator = Agent(
    name="Forecast Orchestrator",
    instructions="""You are the main interface for a forecasting system. Your job is to:

1. Ensure the user's question is appropriate for forecasting
2. Clarify vague questions into specific, time-bound forecasting questions
3. Orchestrate the forecasting pipeline:
   - Gather current world context and background information
   - Find an appropriate reference class and historical base rate
   - Design key parameters that would adjust this base rate
   - Research each parameter with evidence 
   - Synthesize into a final probability
   - Challenge the forecast with a red team

Always maintain a helpful, educational tone. If a question isn't suitable for forecasting, politely explain why and suggest alternatives.
""",
    input_guardrails=[forecastability_guardrail],
    handoffs=[
        question_clarifier_agent,
        background_info_agent,
        reference_class_agent,
        parameter_design_agent,
        synthesis_agent,
        red_team_agent
    ],
    model="gpt-4.1-mini",
)

async def main():
    print("Welcome to the AI Superforecaster")
    draw_graph(forecast_orchestrator, filename="forecast_orchestrator.png")
    print("What would you like to forecast? Please describe the question or scenario.")
    
    user_question = input("> ")
    
    with trace("Forecasting workflow"):
        try:
            # Start with the orchestrator for question validation and clarification
            print("\n=== Processing your question ===")
            
            # First run the clarifier directly
            clarification_result = await Runner.run(
                question_clarifier_agent,
                user_question,
            )
            
            clarification = clarification_result.final_output_as(QuestionClarification)
            
            # If clarification needed, ask follow-up questions
            if clarification.needs_clarification and clarification.follow_up_questions:
                print("\nTo better understand your question, I need some clarification:")
                for i, question in enumerate(clarification.follow_up_questions, 1):
                    print(f"{i}. {question}")
                print("\nPlease provide this additional information:")
                additional_info = input("> ")
                
                # Run clarifier again with the additional information
                clarification_result = await Runner.run(
                    question_clarifier_agent,
                    f"Original question: {user_question}\nAdditional information: {additional_info}",
                )
                clarification = clarification_result.final_output_as(QuestionClarification)
            
            # Use the clarified question for parameter estimation
            final_question = clarification.clarified_question
            print(f"\nForecasting question: {final_question}")
            
            # Start the background info collection in parallel with reference class search
            print("\n=== Finding relevant reference class and gathering background info ===")
            
            # Get current date for background info agent
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            background_info_task = asyncio.create_task(Runner.run(
                background_info_agent,
                f"Provide background information as of {current_date} relevant to the question: {final_question}",
            ))
            
            # Wait for background info first since it provides context for reference class selection
            background_info_result = await background_info_task
            background_info = background_info_result.final_output_as(BackgroundInfoOutput)
            
            # Print background info
            print("\n=== Current World Context ===")
            print(f"Current date: {background_info.current_date}")
            print(f"Summary: {background_info.summary}")
            print("\nRecent Major Events:")
            for event in background_info.major_recent_events[:3]:  # Show top 3 events
                print(f"- {event}")
            print("\nKey Ongoing Trends:")
            for trend in background_info.key_trends[:3]:  # Show top 3 trends
                print(f"- {trend}")
            
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
            print("\n=== REFERENCE CLASSES ===")
            for i, ref_class in enumerate(reference_class_output.reference_classes):
                is_recommended = i == reference_class_output.recommended_class_index
                print(f"\nReference Class {i+1}{' (RECOMMENDED)' if is_recommended else ''}:")
                print(f"Description: {ref_class.reference_class_description}")
                print(f"Base rate: {ref_class.base_rate} [{ref_class.low} - {ref_class.high}]")
                print(f"Sample size: {ref_class.sample_size} historical examples")
                print(f"Sources: {', '.join(ref_class.bibliography)}")
                print(f"Reasoning: {ref_class.reasoning}")
            
            print(f"\nRecommendation reasoning: {reference_class_output.selection_reasoning}")
            
            # Get the recommended reference class for further processing
            recommended_ref_class = reference_class_output.reference_classes[reference_class_output.recommended_class_index]
            
            # Next, determine the key parameters and include background info
            print("\n=== Designing key parameters ===")
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
            print("\n=== Researching parameters (this may take a moment) ===")
            
            # Print the parameters that will be researched
            print("Parameters to research:")
            for i, param in enumerate(parameter_design.parameters, 1):
                print(f"{i}. {param.name}: {param.description}")
                print(f"   Scale: {param.scale_description}")
            
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
            print("\n=== Parameter estimates ===")
            for sample in parameter_samples:
                print(f"{sample.name}: {sample.value} [{sample.low} - {sample.high}]")
                print(f"  Sources: {', '.join(sample.sources)}")
            
            # Synthesize the final forecast
            print("\n=== Creating final forecast ===")
            
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

                        # Process and print the final results
            print("\n=== FINAL FORECAST ===")
            print(f"Question: {final_question}\n")
            print(f"Probability: {final_forecast.final_estimate*100:.1f}% [{final_forecast.final_low*100:.1f}% - {final_forecast.final_high*100:.1f}%]")
            print(f"Starting base rate: {final_forecast.base_rate*100:.1f}%")
            print(f"Key drivers: {', '.join(final_forecast.key_parameters)}")
            print(f"Rationale: {final_forecast.rationale}\n")
            
            # Optionally run red team analysis
            print("\n=== Running red team challenge ===")
            
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
            
            print("=== RED TEAM CHALLENGE ===")
            print(f"Strongest objection: {red_team.strongest_objection}")
            print(f"Alternative estimate: {red_team.alternate_estimate*100:.1f}% [{red_team.alternate_low*100:.1f}% - {red_team.alternate_high*100:.1f}%]")
            print(f"Key disagreements: {', '.join(red_team.key_disagreements)}")
            print(f"Rationale: {red_team.rationale}")
            
        except InputGuardrailTripwireTriggered as e:
            check = e.guardrail_result.output.output_info
            print("\n=== CANNOT PROCESS THIS QUESTION ===")
            print(f"Reason: {check.reasoning}")
            print("\nPlease try asking a question about a future event or trend that can be forecasted.")
            return


if __name__ == "__main__":
    asyncio.run(main())