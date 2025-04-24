from agents import Agent, Runner, trace, ItemHelpers, input_guardrail, GuardrailFunctionOutput, RunContextWrapper, InputGuardrailTripwireTriggered
from agents import ModelSettings
from src.models import *
from src.utils.tools import WebSearchTool

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
1. Generate THREE DISTINCT reference classes that could be applied to this question
2. For each reference class:
   - Identify what makes this reference class appropriate
   - Search for historical data on this reference class using the web search tool
   - Determine the base rate (frequency of occurrence) within this reference class
   - Provide a 90% confidence interval around this base rate
   - Document your sources with at least 1-2 web searches per reference class
   - Explain your reasoning for selecting this reference class
3. Recommend which of the three reference classes should be the primary one to use
4. Provide reasoning for your recommendation

IMPORTANT GUIDELINES:
- Reference classes MUST have meaningfully different base rates (avoid all clustering around 50%)
- Base rates should be derived from EMPIRICAL DATA, not expert opinion
- Include time dimensions - identify how rates change over time when possible
- Base rates typically fall between 30-70% for most realistic reference classes
- Reference classes should contain at least 10 historical examples (ideally more)
- Extreme base rates (<10% or >90%) require extraordinary evidence

Example proper reference classes:
1. "Adoption rate of new consumer tech reaching 50% household penetration within 5 years" (43%)
2. "Rate of medical treatments showing positive Phase III results after promising Phase II" (62%)
3. "Frequency of technology forecasts by industry experts proving accurate within stated timeframe" (37%)

AVOID:
- Using identical or nearly identical reference classes
- Base rates that are exactly 50% (this suggests insufficient research)
- Vague reference classes without concrete examples
- Classes based primarily on expert opinion rather than historical data

Focus on finding relevant historical analogues. Be precise about:
- Sample size (how many cases in your reference class)
- Time period covered
- Any adjustments needed for this specific case
- Quality and limitations of available data""",
    tools=[WebSearchTool()],
    output_type=ReferenceClassOutput,
    model="gpt-4.1-mini",
)


parameter_design_agent = Agent(
    name="Parameter Designer",
    instructions="""You design the key parameters needed to estimate a forecasting question given a reference class and base rate.

For each forecasting question:
1. Start with first principles thinking to identify what fundamentally matters to this question
2. Identify 4-6 key parameters that:
   - Cover different aspects of the forecast (e.g. market, technology, human factors)
   - Are measurable and as objective as possible
   - Together give a comprehensive view of what drives the outcome
   - Would adjust the base rate up or down
3. For each parameter:
   - Define a clear, measurable name and description
   - Provide a clear scale description explaining what the values mean
   - Specify either a 0-10, 0-100%, or similar numeric scale
   - Identify how this parameter interacts with others (additive, multiplicative, etc.)
   - Do NOT provide numeric estimates yet - focus on structure only

IMPORTANT GUIDELINES:
- For 0-10 scales, clearly define what each end of the scale represents
- Define what the midpoint (5) represents as a neutral position
- Ensure parameters are truly orthogonal (independent) when possible
- Each parameter should represent a distinct factor that could move probability
- Design parameters that will translate well to log-odds shifts

Good example parameters:
- What is the probability that AI systems will achieve human-level performance in scientific research by 2030?
  * Technical feasibility (0-10 scale where 0=impossible, 5=moderate challenges, 10=no obstacles)
  * Research funding growth (0-10 scale where 0=major decline, 5=steady state, 10=exponential growth)
  * Current progress rate (0-10 scale where 0=stalled, 5=linear progress, 10=accelerating rapidly)
  * Regulatory environment (0-10 scale where 0=highly restrictive, 5=balanced, 10=supportive)

The goal is to define parameters that:
1. Have clear, measurable definitions with well-defined scales
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
2. Systematically translate the evidence into both a parameter value and Δ log-odds:
   
   For parameters on a 0-10 scale:
   - Value 5 = neutral (0 log-odds)
   - Values 0-4 = negative evidence (map to -0.1 to -0.8 log-odds)
   - Values 6-10 = positive evidence (map to +0.1 to +0.8 log-odds)

   Guideline for log-odds contributions:
   - Weak evidence: ±0.1 to ±0.3
   - Moderate evidence: ±0.4 to ±0.6
   - Strong evidence: ±0.7 to ±1.0
   
   IMPORTANT: Even very strong evidence rarely exceeds ±1.0 log-odds in superforecasting.
   The parameter value and delta_log_odds should be directly correlated.

3. Cite your sources clearly
4. Provide concise reasoning for your estimate

Examples of proper mapping:
- Parameter value 7/10 → delta_log_odds: +0.4 (moderate positive)
- Parameter value 3/10 → delta_log_odds: -0.4 (moderate negative)
- Parameter value 9/10 → delta_log_odds: +0.8 (strong positive)

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
2. Estimates for key parameters with confidence intervals and log-odds contributions

Your task:
1. Convert base-rate to log-odds:   L = logit(base_rate)
2. Add each parameter's `delta_log_odds` to L
3. final_prob = inv_logit(L)
4. Determine how each parameter shifts the base rate
5. Calculate a final probability estimate with 90% confidence interval
6. Identify the 2-3 most influential parameters
7. Write a clear, one-paragraph rationale that references these key parameters

IMPORTANT CALIBRATION GUIDELINES:
- Most forecasts should remain between 20-80% probability range
- Probabilities >90% or <10% require exceptional evidence
- The total log-odds shift across all parameters typically falls between -2.0 and +2.0
- If your calculation yields extreme probabilities (>95% or <5%), reconsider whether the evidence truly warrants such certainty
- Superforecasters are conservative - they avoid extreme probabilities without overwhelming evidence
- Make sure parameters collectively tell a coherent story

Common superforecaster ranges:
- Base rate moves from 50% → 75% requires log-odds shift of +1.1
- Base rate moves from 50% → 90% requires log-odds shift of +2.2
- Total parameter adjustments rarely move probability >30 percentage points

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
- Ways the parameter contributions might be misunderstood or miscalibrated
- Cognitive biases that might be affecting the forecast

IMPORTANT CALIBRATION CHECKS:
- If the forecast is extremely confident (>90% or <10%), challenge the certainty
- Identify any parameter log-odds contributions that seem exaggerated
- Challenge any parameter assessments that lack empirical support
- Identify instances where the total log-odds shift seems unrealistically large
- Question reference classes that don't contain sufficient historical examples

Common superforecasting errors to identify:
- Overconfidence (probabilities too extreme)
- Parameter interactions not properly accounted for
- Reference classes that are too narrow or not truly analogous
- Neglecting key external factors or alternative hypotheses
- Base rates derived from insufficient data

Your challenge should follow superforecaster best practices - be data-driven, 
appropriately calibrated, and avoid going to extremes without strong evidence.""",
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