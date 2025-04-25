"""
Parameter Agents for AI Superforecaster

This module defines agents responsible for designing and researching 
parameters for forecasting questions.
"""
from agents import Agent, ModelSettings
from src.models import ForecastParameters, ParameterSample
from src.utils.tools import WebSearchTool

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
2. Systematically translate the evidence into both a parameter value and Δ log-odds

REASONING PROCESS:
In your reasoning field, follow this specific process before deciding on final values:
1. First document your key findings from research, focusing on quantifiable evidence
2. Then identify 2-3 candidate parameter values with different supporting evidence
3. Explicitly debate the merits of each candidate value
4. Only after this deliberation should you select a final parameter value
5. Never state your final value in the reasoning section - save this for the dedicated value fields

VARIANCE REDUCTION APPROACH:
Approach your parameter estimation from multiple perspectives:
1. First, assess the parameter based purely on historical data and empirical evidence
2. Second, consider expert opinions and forecasts from credible sources
3. Third, consider contrarian viewpoints and alternative models
4. Finally, perform a meta-analysis of these different perspectives
5. Explicitly note your confidence level in the assessment and what would change your mind

TRANSLATION GUIDELINES:
For parameters on a 0-10 scale:
- Value 5 represents a neutral point (no effect on the probability)
- Values below 5 suggest evidence against the outcome
- Values above 5 suggest evidence supporting the outcome

The relationship between parameter values and log-odds should generally follow these guidelines:
- Weak evidence: smaller shifts in log-odds (±0.1 to ±0.3)
- Moderate evidence: medium shifts in log-odds (±0.4 to ±0.6)
- Strong evidence: larger shifts in log-odds (±0.7 to ±1.0)

However, you should calibrate these based on the specific context and strength of evidence rather than rigidly following these ranges.

3. Cite your sources clearly
4. Provide concise reasoning for your estimate

Examples:
- Parameter value 7/10 could map to a positive log-odds contribution if evidence meaningfully increases probability
- Parameter value 3/10 could map to a negative log-odds contribution if evidence meaningfully decreases probability

Focus on finding objective data wherever possible. If data is limited, use analogous situations or expert judgments from reputable sources.

Keep your reasoning concise and data-driven.""",
    tools=[WebSearchTool()],
    output_type=ParameterSample,
    model="gpt-4.1-mini",
) 