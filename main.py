import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from agents import Agent, Runner, trace, ItemHelpers
from agents.tool import WebSearchTool
from agents.extensions.visualization import draw_graph
from agents import ModelSettings

class ParameterEstimate(BaseModel):
    """Parameter estimate with name, value and confidence interval"""
    name: str = Field(description="Name of the parameter")
    value: float = Field(description="Estimated value")
    low: float = Field(description="Lower bound of 90% confidence interval")
    high: float = Field(description="Upper bound of 90% confidence interval")
    reasoning: str = Field(description="Reasoning for this estimate")


class ForecastParameters(BaseModel):
    """Output format for forecast parameters"""
    question: str = Field(description="Original forecasting question")
    parameters: List[ParameterEstimate] = Field(description="List of parameters to estimate")
    additional_considerations: List[str] = Field(description="Additional factors to consider")


parameter_estimator_agent = Agent(
    name="Parameter Estimator",
    instructions="""You help users break down forecasting questions into key parameters that need to be estimated.

For any forecasting question:
1. Start with first principles thinking to identify what fundamentally matters
2. Search extensively for historical data, trends, and analogous situations to inform estimates
3. Identify 5 key parameters that:
   - Cover different aspects of the forecast (e.g. market, technology, human factors)
   - Are measurable and objective
   - Together give a comprehensive view of what drives the outcome
4. For each parameter:
   - Search for relevant historical base rates and data
   - Provide estimates (value, 90% confidence interval)
   - Document reasoning, citing data sources and historical examples
5. Consider extreme scenarios and ways the forecast could be wrong

Focus on parameters that are both measurable and meaningful for modeling the forecast outcome.

Make sure to use the web search tool to find relevant data. This is required. 
""",
    tools=[WebSearchTool()],
    output_type=ForecastParameters,
    model="gpt-4.1-mini",
)


async def main():
    draw_graph(parameter_estimator_agent, filename="parameter_estimator_agent.png")
    print("Welcome to the Parameter Estimator for Forecasting")
    print("What would you like to forecast? Please describe the question or scenario.")
    
    user_question = input("> ")
    
    with trace("Parameter estimation"):
        # Use run_streamed instead of run
        print("\n=== Starting parameter estimation (streaming) ===")
        result = Runner.run_streamed(
            parameter_estimator_agent,
            user_question,
        )
        
        # Process streaming events
        async for event in result.stream_events():
            # raw response events
            if event.type == "raw_response_event":
                # print("raw response event", event.data)
                pass
            # When the agent updates, print that
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
            # When items are generated, print them
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print(f"🔍 Searching for: {event.item.raw_item}")
                elif event.item.type == "tool_call_output_item":
                    print(f"📊 Found relevant data")
                elif event.item.type == "message_output_item":
                    message = ItemHelpers.text_message_output(event.item)
                    if message:
                        print(f"💭 Agent thinking: {message[:100]}...")
        
        # Get the final output
        forecast_params = result.final_output_as(ForecastParameters)
    
    # Print the final formatted results
    print("\n=== FORECASTING PARAMETERS ===")
    print(f"Question: {forecast_params.question}\n")
    
    print("Key Parameters:")
    for i, param in enumerate(forecast_params.parameters, 1):
        print(f"{i}. {param.name}")
        print(f"   Estimate: {param.value} [{param.low} - {param.high}]")
        print(f"   Reasoning: {param.reasoning}\n")
    
    print("Additional Considerations:")
    for i, consideration in enumerate(forecast_params.additional_considerations, 1):
        print(f"{i}. {consideration}")


if __name__ == "__main__":
    asyncio.run(main())