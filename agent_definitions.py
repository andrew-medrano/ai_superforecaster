from agents import Agent, Runner, trace, ItemHelpers, input_guardrail, GuardrailFunctionOutput, RunContextWrapper, InputGuardrailTripwireTriggered
from agents.tool import WebSearchTool
from agents import ModelSettings
from models import *

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