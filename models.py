from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

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