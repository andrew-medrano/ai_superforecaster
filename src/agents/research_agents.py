"""
Research Agents for AI Superforecaster

This module defines agents responsible for gathering background information
and finding appropriate reference classes for forecasting questions.
"""
from agents import Agent, ModelSettings
from src.models import BackgroundInfoOutput, ReferenceClassOutput
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

CRITICAL FOCUS: First, deeply understand WHAT is being predicted and find TRULY RELEVANT historical analogues.

Initial analysis (before searching):
1. Break down the question: What EXACT outcome is being forecasted? By what EXACT timeframe?
2. Identify the key domain (technology adoption, political event, economic trend, etc.)
3. List at least 3 specific criteria that define a valid reference class for this question

For any forecasting question:
1. Generate FOUR DISTINCT reference classes that could be applied to this question
2. For each reference class:
   - Identify what makes this reference class appropriate
   - Search for historical data on this reference class using the web search tool
   - Determine the base rate (frequency of occurrence) within this reference class
   - Provide a 90% confidence interval around this base rate
   - Document your sources with at least 1-2 web searches per reference class
   - Explain your reasoning for selecting this reference class
3. SELF-REFLECTION STEP: After creating all reference classes, explicitly critique each one:
   - How closely does it match the SPECIFIC event being predicted?
   - Are there significant contextual differences from historical examples?
   - Is the data quality sufficient for confident estimates?
   - Assign a "relevance score" (1-10) to each reference class
4. DISCARD the lowest-scoring reference class
5. Recommend which of the remaining three reference classes should be the primary one to use
6. Provide reasoning for your recommendation

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
- Classes that fail to capture the SPECIFIC outcome being forecast

CRITICAL SELF-CHECK:
Before finalizing, ask yourself:
- "If I were betting my own money, would I trust these reference classes?"
- "Have I found the most relevant historical analogues possible?"
- "Would domain experts agree these are appropriate comparisons?"
- "Have I adjusted for unique aspects of the current situation?"

If you're not satisfied with the quality of reference classes after searches, CREATE A CUSTOM reference class that more precisely matches the question, even if you have to make educated estimates about the base rate.""",
    tools=[WebSearchTool()],
    output_type=ReferenceClassOutput,
    model="gpt-4.1-mini",
) 