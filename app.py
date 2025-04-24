import streamlit as st
import asyncio
from asyncio import create_task
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from agents import Runner, trace, InputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
from src.models import *
from src.agents import (background_info_agent, reference_class_agent, parameter_design_agent, 
                   parameter_researcher_agent, synthesis_agent, question_validator_agent,
                   question_clarifier_agent, forecast_orchestrator, red_team_agent)
from src.utils.tools import WebSearchTool
from src.utils.forecast_math import logit, inv_logit

# App configuration
st.set_page_config(
    page_title="AI Superforecaster",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to ensure percentages stay within 0-100% range
def safe_percentage(value):
    """Ensure a value is between 0 and 1 for percentage display"""
    return max(0, min(1, value))

# Helper functions for streamlit-friendly asyncio
async def run_forecast_workflow(question):
    with trace("Forecasting workflow"):
        try:
            # Phase 1: Clarification
            with st.status("Processing your question...", expanded=True) as status:
                status.update(label="Clarifying your question...")
                
                # Run the clarifier
                clarification_result = await Runner.run(
                    question_clarifier_agent,
                    question,
                )
                
                clarification = clarification_result.final_output_as(QuestionClarification)
                
                # If clarification needed, ask follow-up questions
                if clarification.needs_clarification and clarification.follow_up_questions:
                    status.update(label="Additional information needed", state="complete")
                    st.info("To better understand your question, I need some clarification:")
                    
                    # Create a form for follow-up answers
                    with st.form("clarification_form"):
                        follow_up_answers = []
                        for i, question in enumerate(clarification.follow_up_questions, 1):
                            follow_up_answers.append(st.text_input(f"{i}. {question}"))
                        
                        submit_clarification = st.form_submit_button("Submit")
                    
                    if submit_clarification:
                        additional_info = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in 
                                                enumerate(zip(clarification.follow_up_questions, follow_up_answers))])
                        
                        # Run clarifier again with the additional information
                        status.update(label="Processing additional information...", state="running")
                        clarification_result = await Runner.run(
                            question_clarifier_agent,
                            f"Original question: {question}\nAdditional information: {additional_info}",
                        )
                        clarification = clarification_result.final_output_as(QuestionClarification)
                else:
                    status.update(label="Question clarified", state="complete")
                
                # Use the clarified question for the rest of the process
                final_question = clarification.clarified_question
                st.session_state.final_question = final_question
                
                # Display the clarified question immediately
                with status:
                    st.subheader("Clarified Question")
                    st.write(final_question)
                
            # Phase 2: Background info and reference classes
            with st.status("Gathering background information and finding reference classes...", expanded=True) as status:
                # Get current date for background info agent
                import datetime
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Run background info collection
                status.update(label="Gathering background information...")
                background_info_task = create_task(Runner.run(
                    background_info_agent,
                    f"Provide background information as of {current_date} relevant to the question: {final_question}",
                ))
                
                # Wait for background info first since it provides context for reference class selection
                background_info_result = await background_info_task
                background_info = background_info_result.final_output_as(BackgroundInfoOutput)
                
                # Store in session state
                st.session_state.background_info = background_info
                
                # Display background info as it becomes available
                with status:
                    st.subheader("Background Information")
                    st.write(f"Current date: {background_info.current_date}")
                    
                    st.markdown("**Background Summary:**")
                    st.write(background_info.summary)
                    
                    st.markdown("**Recent Major Events:**")
                    for event in background_info.major_recent_events[:5]:
                        st.write(f"• {event}")
                    
                    st.markdown("**Key Ongoing Trends:**")
                    for trend in background_info.key_trends[:5]:
                        st.write(f"• {trend}")
                
                # Create reference class prompt with background context
                status.update(label="Finding relevant reference classes...")
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
                
                # Run reference class search
                reference_class_task = create_task(Runner.run(
                    reference_class_agent,
                    reference_class_prompt,
                ))
                
                # Wait for reference class results
                reference_class_result = await reference_class_task
                reference_class_output = reference_class_result.final_output_as(ReferenceClassOutput)
                
                # Store in session state
                st.session_state.reference_class_output = reference_class_output
                
                # Get the recommended reference class for further processing
                recommended_ref_class = reference_class_output.reference_classes[reference_class_output.recommended_class_index]
                st.session_state.recommended_ref_class = recommended_ref_class
                
                # Display reference classes as they become available
                with status:
                    st.subheader("Reference Classes")
                    tabs = st.tabs([f"Reference Class {i+1}" for i in range(len(reference_class_output.reference_classes))])
                    
                    for i, (tab, ref_class) in enumerate(zip(tabs, reference_class_output.reference_classes)):
                        with tab:
                            is_recommended = i == reference_class_output.recommended_class_index
                            if is_recommended:
                                st.markdown("**🌟 RECOMMENDED CLASS 🌟**")
                            
                            st.markdown(f"**Description:** {ref_class.reference_class_description}")
                            st.markdown(f"**Base rate:** {ref_class.base_rate*100:.1f}% [{ref_class.low*100:.1f}% - {ref_class.high*100:.1f}%]")
                            
                            # Add back the confidence interval visualization
                            fig, ax = plt.subplots(figsize=(5, 0.8))
                            
                            # Draw confidence interval
                            ax.hlines(y=0, xmin=0, xmax=1, linewidth=1, color='gray')
                            ax.hlines(y=0, xmin=ref_class.low, xmax=ref_class.high, linewidth=4, color='lightblue')
                            ax.plot(ref_class.base_rate, 0, 'ro', markersize=8)
                            
                            # Set plot parameters
                            ax.set_xlim(0, 1)
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            
                            st.pyplot(fig)
                            
                            st.markdown(f"**Sample size:** {ref_class.sample_size} historical examples")
                            
                            st.markdown("**Reasoning:**")
                            st.write(ref_class.reasoning)
                            
                            st.markdown("**Sources:**")
                            for source in ref_class.bibliography:
                                st.write(f"• {source}")
                
                # After displaying all reference classes
                with status:
                    st.markdown("**Selection Reasoning:**")
                    st.write(reference_class_output.selection_reasoning)
                
                status.update(label="Information gathering complete", state="complete")
            
            # Phase 3: Parameter design and research
            with st.status("Analyzing forecast parameters...", expanded=True) as status:
                # Design parameters
                status.update(label="Designing key parameters...")
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
                
                parameter_design_task = create_task(Runner.run(
                    parameter_design_agent,
                    parameter_design_prompt,
                ))
                
                # Process parameters
                parameter_design_result = await parameter_design_task
                parameter_design = parameter_design_result.final_output_as(ForecastParameters)
                
                # Store in session state
                st.session_state.parameter_design = parameter_design
                
                # Display parameter design as it becomes available
                with status:
                    st.subheader("Parameter Design")
                    for i, param in enumerate(parameter_design.parameters, 1):
                        st.markdown(f"**{i}. {param.name}**")
                        st.markdown(f"**Description:** {param.description}")
                        st.markdown(f"**Scale:** {param.scale_description}")
                        
                        if param.interacts_with:
                            st.markdown(f"**Interacts with:** {', '.join(param.interacts_with)}")
                        
                        if param.interaction_description:
                            st.markdown(f"**Interaction:** {param.interaction_description}")
                        
                        if i < len(parameter_design.parameters):
                            st.markdown("---")  # Add a divider between parameters
                
                # Create context for parameter research
                status.update(label="Researching parameters (this may take a few minutes)...")
                
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
                
                # Create placeholder for parameter research results
                parameter_results_placeholder = status.container()
                with parameter_results_placeholder:
                    st.subheader("Parameter Research Progress")
                    parameter_progress = st.empty()
                    parameter_progress.info("Researching parameters...")
                
                # Research parameters in parallel
                async def research_parameter(param: ParameterMeta, idx: int) -> ParameterSample:
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
                    
                    # Update the progress display with completed parameter
                    with parameter_results_placeholder:
                        st.success(f"Researched parameter {idx+1}/{len(parameter_design.parameters)}: {param.name}")
                        st.markdown(f"**Estimate:** {sample.value:.2f} [{sample.low:.2f} - {sample.high:.2f}]")
                        
                        # Display log-odds contribution if available
                        if sample.delta_log_odds is not None:
                            sign = "+" if sample.delta_log_odds > 0 else ""
                            st.markdown(f"**Log-odds contribution:** {sign}{sample.delta_log_odds:.3f}")
                            
                            # Add explanation of what this means
                            if abs(sample.delta_log_odds) < 0.2:
                                strength = "very weak"
                            elif abs(sample.delta_log_odds) < 0.4:
                                strength = "weak"
                            elif abs(sample.delta_log_odds) < 0.7:
                                strength = "moderate"
                            elif abs(sample.delta_log_odds) <= 1.0:
                                strength = "strong"
                            else:
                                strength = "very strong"
                                
                            direction = "increases" if sample.delta_log_odds > 0 else "decreases"
                            prob_shift = f"50% → {inv_logit(0.5 + sample.delta_log_odds)*100:.0f}%"
                            st.markdown(f"*This parameter {direction} probability ({strength} evidence, equivalent to {prob_shift})*")
                            
                            # Check for calibration issues
                            if abs(sample.delta_log_odds) > 1.0:
                                st.warning("⚠️ This log-odds contribution is larger than typical in superforecasting. Consider if this level of impact is justified by the evidence.")
                            
                            # Validate parameter/log-odds consistency for 0-10 scale parameters
                            param_meta = next((p for p in parameter_design.parameters if p.name == sample.name), None)
                            if param_meta and "0-10" in param_meta.scale_description:
                                expected_lo_min = (sample.value - 5) * 0.1
                                expected_lo_max = (sample.value - 5) * 0.2
                                if not (min(expected_lo_min, expected_lo_max) <= sample.delta_log_odds <= max(expected_lo_min, expected_lo_max)):
                                    if sample.value == 5 and sample.delta_log_odds != 0:
                                        st.warning("⚠️ A value of 5 on a 0-10 scale should typically have 0 log-odds contribution (neutral)")
                                    elif abs(sample.delta_log_odds) > 1.5 * max(abs(expected_lo_min), abs(expected_lo_max)):
                                        st.warning(f"⚠️ The log-odds contribution seems disproportionate to the parameter value. For {sample.value}/10, expect log-odds around {expected_lo_min:.2f} to {expected_lo_max:.2f}")
                        
                            # Add visualization for parameter scale mapped to log-odds
                            fig, ax = plt.subplots(figsize=(5, 0.8))
                            
                            # Determine min and max for visualization
                            viz_min = min(0, sample.low - 0.2 * (sample.high - sample.low))
                            viz_max = max(1, sample.high + 0.2 * (sample.high - sample.low))
                            
                            # Draw confidence interval
                            ax.hlines(y=0, xmin=viz_min, xmax=viz_max, linewidth=1, color='gray')
                            ax.hlines(y=0, xmin=sample.low, xmax=sample.high, linewidth=4, color='lightblue')
                            ax.plot(sample.value, 0, 'ro', markersize=8)
                            
                            # Set plot parameters
                            ax.set_xlim(viz_min, viz_max)
                            ax.set_ylim(-0.5, 0.5)
                            ax.set_yticks([])
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.spines['top'].set_visible(False)
                            
                            st.pyplot(fig)
                            
                            st.markdown("**Reasoning:**")
                            st.write(sample.reasoning)
                    
                    return sample
                
                # Run parameter research in parallel and update progress
                param_tasks = [research_parameter(param, i) for i, param in enumerate(parameter_design.parameters)]
                parameter_samples = await asyncio.gather(*param_tasks)
                
                # Store in session state
                st.session_state.parameter_samples = parameter_samples
                
                # Update the parameter research display
                parameter_progress.success(f"All {len(parameter_samples)} parameters researched successfully!")
                
                status.update(label="Parameter analysis complete", state="complete")
            
            # Phase 4: Synthesis and red team
            with st.status("Creating final forecast...", expanded=True) as status:
                # Synthesize forecast
                status.update(label="Synthesizing final forecast...")
                
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
                
                # Collect valid parameter adjustments
                valid_params = [(p.name, p.delta_log_odds) for p in parameter_samples if p.delta_log_odds is not None]
                parameter_contributions = {}
                
                # Check for excessively large cumulative shifts
                total_shift = sum(abs(delta) for _, delta in valid_params)
                if total_shift > 4.0:
                    # Scale down all contributions proportionally if the total is too extreme
                    scaling_factor = 4.0 / total_shift
                    adjustment_warning = f"⚠️ Total log-odds shifts were scaled down by {scaling_factor:.2f} to maintain reasonable calibration"
                else:
                    scaling_factor = 1.0
                    adjustment_warning = None
                
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
                    conservatism_warning = "⚠️ Applied superforecaster conservatism to extreme probability"
                else:
                    conservatism_warning = None
                
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
                
                # Store in session state
                st.session_state.final_forecast = final_forecast
                st.session_state.log_odds_calculation = {
                    "base_rate": recommended_ref_class.base_rate,
                    "base_log_odds": L_base,
                    "parameter_contributions": parameter_contributions,
                    "final_log_odds": L,
                    "final_probability": final_prob,
                    "adjustment_warning": adjustment_warning,
                    "conservatism_warning": conservatism_warning,
                    "scaling_factor": scaling_factor
                }
                
                # Display the final forecast as it becomes available
                with status:
                    st.subheader("Final Forecast")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Probability:** {safe_percentage(final_forecast.final_estimate)*100:.1f}% [{safe_percentage(final_forecast.final_low)*100:.1f}% - {safe_percentage(final_forecast.final_high)*100:.1f}%]")
                        st.markdown(f"**Starting base rate:** {safe_percentage(final_forecast.base_rate)*100:.1f}%")
                        st.markdown(f"**Key drivers:** {', '.join(final_forecast.key_parameters)}")
                    
                    with col2:
                        # Create a simple gauge visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=safe_percentage(final_forecast.final_estimate) * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [0, 100], 'ticksuffix': "%"}},
                            number={'suffix': "%", 'valueformat': '.1f'}
                        ))
                        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a detailed calculation section using log-odds arithmetic
                    st.markdown("### Log-Odds Calculation")
                    st.markdown("This forecast uses log-odds arithmetic which properly handles probability adjustments:")
                    
                    # Show any calibration adjustments
                    if 'adjustment_warning' in st.session_state.log_odds_calculation and st.session_state.log_odds_calculation['adjustment_warning']:
                        st.warning(st.session_state.log_odds_calculation['adjustment_warning'])
                    
                    if 'conservatism_warning' in st.session_state.log_odds_calculation and st.session_state.log_odds_calculation['conservatism_warning']:
                        st.warning(st.session_state.log_odds_calculation['conservatism_warning'])
                    
                    # Create a table to show the calculation steps
                    calculation_data = []
                    
                    # Base rate row
                    calculation_data.append({
                        "Component": "Base rate (reference class)",
                        "Probability": f"{recommended_ref_class.base_rate*100:.1f}%",
                        "Log-odds": f"{L_base:.3f}",
                        "Contribution": f"{L_base:.3f}",
                    })
                    
                    # Parameter rows
                    for name, delta in sorted(parameter_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                        sign = "+" if delta > 0 else ""
                        
                        # Calculate the equivalent probability shift for this parameter
                        prior_prob = 0.5  # Using 50% as reference point
                        new_prob = inv_logit(logit(prior_prob) + delta)
                        prob_shift = f"{prior_prob*100:.0f}% → {new_prob*100:.0f}%"
                        
                        calculation_data.append({
                            "Component": name,
                            "Probability": prob_shift,
                            "Log-odds": f"{sign}{delta:.3f}",
                            "Contribution": f"{sign}{delta:.3f}",
                        })
                    
                    # Add scaling factor row if applied
                    if 'scaling_factor' in st.session_state.log_odds_calculation and st.session_state.log_odds_calculation['scaling_factor'] < 1.0:
                        calculation_data.append({
                            "Component": "Calibration adjustment",
                            "Probability": "—",
                            "Log-odds": f"×{st.session_state.log_odds_calculation['scaling_factor']:.2f}",
                            "Contribution": "Superforecaster calibration",
                        })
                    
                    # Final probability row
                    calculation_data.append({
                        "Component": "Final forecast",
                        "Probability": f"{final_prob*100:.1f}%",
                        "Log-odds": f"{L:.3f}",
                        "Contribution": "Sum of above",
                    })
                    
                    # Create and display the table
                    calc_df = pd.DataFrame(calculation_data)
                    st.table(calc_df)
                    
                    # Add explanation of the process
                    st.markdown("""
                    **Understanding log-odds:**
                    * Converting to log-odds space allows proper weighting of evidence
                    * Each parameter contributes a "chip" (delta log-odds)
                    * Positive values increase probability, negative values decrease it
                    * The final probability respects proper bounds (0-100%)
                    * Equal evidence has equal weight regardless of where in the probability scale
                    * Superforecasters apply conservatism to avoid overconfidence
                    """)
                    
                    # Display total log-odds shift and categorize
                    total_raw_shift = sum(abs(delta) for _, delta in valid_params)
                    total_adjusted_shift = sum(abs(delta) for delta in parameter_contributions.values())
                    
                    st.markdown(f"""
                    **Total log-odds impact: {total_adjusted_shift:.2f}** (raw: {total_raw_shift:.2f})
                    
                    *Interpretation:*
                    """)
                    
                    if total_adjusted_shift < 1.0:
                        st.markdown("✅ **Conservative shift** - typical of careful superforecasters")
                    elif total_adjusted_shift < 2.0:
                        st.markdown("✅ **Moderate shift** - within normal superforecaster range")
                    elif total_adjusted_shift < 3.0:
                        st.markdown("⚠️ **Large shift** - requires strong evidence to justify")
                    else:
                        st.markdown("🚨 **Extreme shift** - superforecasters rarely make shifts this large")
                    
                    # Add a visualization of how each parameter shifted the probability
                    st.markdown("### Parameter Impact Visualization")
                    
                    # Create data for cumulative parameter impact visualization
                    cumulative_data = []
                    running_log_odds = L_base
                    running_prob = recommended_ref_class.base_rate
                    
                    # Base rate starting point
                    cumulative_data.append({
                        "Stage": "Base rate",
                        "Probability": running_prob*100,
                        "Log-odds": running_log_odds
                    })
                    
                    # Add each parameter's impact
                    for name, delta in sorted(parameter_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                        running_log_odds += delta
                        new_prob = inv_logit(running_log_odds)
                        prob_change = (new_prob - running_prob) * 100
                        cumulative_data.append({
                            "Stage": name,
                            "Probability": new_prob*100,
                            "Change": f"{'+' if prob_change >= 0 else ''}{prob_change:.1f}%",
                            "Log-odds": running_log_odds
                        })
                        running_prob = new_prob
                    
                    # Add calibration adjustment if applied
                    if 'conservatism_warning' in st.session_state.log_odds_calculation and st.session_state.log_odds_calculation['conservatism_warning']:
                        # Show the effect of conservatism adjustment
                        final_adjusted_prob = final_prob
                        if running_prob != final_adjusted_prob:
                            prob_change = (final_adjusted_prob - running_prob) * 100
                            cumulative_data.append({
                                "Stage": "Conservatism",
                                "Probability": final_adjusted_prob*100,
                                "Change": f"{'+' if prob_change >= 0 else ''}{prob_change:.1f}%",
                                "Log-odds": L
                            })
                    
                    # Create waterfall chart for probability changes
                    impact_df = pd.DataFrame(cumulative_data)
                    
                    # Use plotly to create waterfall chart
                    labels = impact_df["Stage"].tolist()
                    values = impact_df["Probability"].tolist()
                    
                    waterfall_fig = go.Figure(go.Waterfall(
                        name="Probability Progression", 
                        orientation="v",
                        measure=["absolute"] + ["relative"] * (len(values)-1),
                        x=labels,
                        y=[values[0]] + [values[i] - values[i-1] for i in range(1, len(values))],
                        connector={"line":{"color":"rgb(63, 63, 63)"}},
                        textposition="outside",
                        text=[f"{values[0]:.1f}%"] + [f"{values[i]-values[i-1]:+.1f}%" for i in range(1, len(values))],
                    ))
                    
                    waterfall_fig.update_layout(
                        title=f"How Parameters Shifted the Probability",
                        showlegend=False,
                        height=400,
                        xaxis_title="Parameter",
                        yaxis_title="Probability (%)",
                        yaxis=dict(
                            range=[0, max(100, max(values) * 1.1)],
                            ticksuffix="%"
                        )
                    )
                    
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                    st.markdown("### Rationale")
                    st.write(final_forecast.rationale)
                
                # Run red team analysis
                status.update(label="Running red team challenge...")
                
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
                
                # Store in session state
                st.session_state.red_team = red_team
                
                # Display the red team challenge as it becomes available
                with status:
                    st.subheader("Red Team Challenge")
                    st.markdown(f"**Strongest objection:** {red_team.strongest_objection}")
                    st.markdown(f"**Alternative estimate:** {safe_percentage(red_team.alternate_estimate)*100:.1f}% [{safe_percentage(red_team.alternate_low)*100:.1f}% - {safe_percentage(red_team.alternate_high)*100:.1f}%]")
                    
                    # Add a comparison chart between main forecast and red team
                    fig = go.Figure()
                    
                    # Add main forecast with confidence interval
                    fig.add_trace(go.Scatter(
                        x=[safe_percentage(final_forecast.final_low)*100, safe_percentage(final_forecast.final_estimate)*100, safe_percentage(final_forecast.final_high)*100],
                        y=['Main Forecast', 'Main Forecast', 'Main Forecast'],
                        mode='markers',
                        marker=dict(color='blue', size=12),
                        name='Main Forecast'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[safe_percentage(final_forecast.final_low)*100, safe_percentage(final_forecast.final_high)*100],
                        y=['Main Forecast', 'Main Forecast'],
                        mode='lines',
                        line=dict(color='blue', width=8),
                        showlegend=False
                    ))
                    
                    # Add red team's alternative with confidence interval
                    fig.add_trace(go.Scatter(
                        x=[safe_percentage(red_team.alternate_low)*100, safe_percentage(red_team.alternate_estimate)*100, safe_percentage(red_team.alternate_high)*100],
                        y=['Red Team', 'Red Team', 'Red Team'],
                        mode='markers',
                        marker=dict(color='red', size=12),
                        name='Red Team'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[safe_percentage(red_team.alternate_low)*100, safe_percentage(red_team.alternate_high)*100],
                        y=['Red Team', 'Red Team'],
                        mode='lines',
                        line=dict(color='red', width=8),
                        showlegend=False
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title="Forecast Comparison",
                        xaxis_title="Probability (%)",
                        xaxis=dict(range=[0, 100]),
                        height=200,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Key disagreements:**")
                    for disagreement in red_team.key_disagreements:
                        st.markdown(f"• {disagreement}")
                    
                    st.markdown("**Red Team Rationale:**")
                    st.write(red_team.rationale)
                
                status.update(label="Forecast complete", state="complete")
                
                # Indicate forecast is complete
                st.session_state.forecast_complete = True
                
            return True
            
        except InputGuardrailTripwireTriggered as e:
            check = e.guardrail_result.output.output_info
            st.error(f"Cannot process this question: {check.reasoning}")
            return False

# Main app flow
def main():
    # Initialize session state
    if 'forecast_complete' not in st.session_state:
        st.session_state.forecast_complete = False
    
    # Set up the sidebar
    with st.sidebar:
        st.title("AI Superforecaster")
        
        # Try to load local image, fall back to placeholder
        logo_path = os.path.join("static", "images", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2784/2784589.png", width=150)
            
        st.markdown("---")
        st.markdown("""
        An LLM-based system for generating probabilistic forecasts using:
        - Reference class forecasting
        - Parameter estimation
        - Red team analysis
        """)
        
        # Add information about log-odds methodology
        st.markdown("---")
        with st.expander("About Log-Odds Arithmetic"):
            st.markdown("""
            **Why Log-Odds Matter**
            
            This system uses log-odds arithmetic - the same method professional superforecasters use:
            
            * **Removes probability creep**: Prevents probabilities from exceeding 0-100%
            * **Fixes asymmetric impacts**: Equal evidence has equal weight regardless of probability range
            * **Reveals factor impact**: Shows clear contribution of each parameter
            
            Example: In percentage arithmetic, +10% means different things at different points (10→20% is a bigger shift than 70→80%). In log-odds, evidence has the same weight regardless of starting probability.
            """)
            
            # Add a visual example
            st.markdown("#### Visual Example")
            example_data = pd.DataFrame({
                "Starting": ["10%", "70%"],
                "Ending": ["20%", "80%"],
                "% Change": ["+10%", "+10%"],
                "Log-Odds Change": ["+0.41", "+0.41"]
            })
            st.table(example_data)
            st.markdown("*The log-odds approach recognizes these are equivalent evidence shifts*")
            
            # Add a conversion chart
            st.markdown("#### Log-Odds to Probability Conversion")
            
            # Generate data points for the conversion chart
            log_odds_range = np.linspace(-5, 5, 100)
            probs = [inv_logit(lo) for lo in log_odds_range]
            
            # Create chart
            conversion_fig = go.Figure()
            conversion_fig.add_trace(go.Scatter(
                x=log_odds_range, 
                y=probs,
                mode='lines',
                name='Probability',
                line=dict(color='blue', width=2)
            ))
            
            # Add reference lines and annotations
            for lo, label in [(-2, "12%"), (-1, "27%"), (0, "50%"), (1, "73%"), (2, "88%")]:
                p = inv_logit(lo)
                conversion_fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="gray"),
                    x0=lo, y0=0, x1=lo, y1=p
                )
                conversion_fig.add_shape(
                    type="line", line=dict(dash="dash", width=1, color="gray"),
                    x0=-5, y0=p, x1=lo, y1=p
                )
                conversion_fig.add_annotation(
                    x=lo, y=p, text=f"({lo}, {p*100:.0f}%)", 
                    showarrow=True, arrowhead=2
                )
            
            conversion_fig.update_layout(
                title="Log-Odds to Probability Conversion",
                xaxis_title="Log-Odds",
                yaxis_title="Probability",
                yaxis=dict(tickformat=".0%", range=[0, 1]),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            st.plotly_chart(conversion_fig, use_container_width=True)
            
            # Add typical chip sizes reference with annotations about meaning
            st.markdown("""
            **Superforecaster Log-Odds Guideline:**
            
            | Evidence Strength | Log-Odds Shift | Example Probability Shift |
            | ----------------- | -------------- | ------------------------- |
            | Very weak | ±0.1 | 50% → 52% |
            | Weak | ±0.2 to ±0.3 | 50% → 57% |
            | Moderate | ±0.4 to ±0.6 | 50% → 65% |
            | Strong | ±0.7 to ±1.0 | 50% → 73% |
            | Very strong | ±1.5 | 50% → 82% |
            | Overwhelming | ±2.0 | 50% → 88% |
            
            Most parameters contribute between ±0.1 and ±0.6 log-odds.
            
            The TOTAL log-odds shift across all parameters rarely exceeds ±2.0
            (equivalent to moving from 50% to 88% or from 50% to 12%).
            
            **Parameters on a 0-10 scale typically map to log-odds as:**
            - Value 5 = neutral (0 log-odds)
            - Each point above 5 = +0.1 to +0.2 log-odds
            - Each point below 5 = -0.1 to -0.2 log-odds
            """)
        
        st.markdown("---")
        
        # Reset button
        if st.button("New Forecast"):
            # Clear session state except for auth
            for key in list(st.session_state.keys()):
                if key != 'auth':
                    del st.session_state[key]
            st.session_state.forecast_complete = False
            st.rerun()
    
    # Main content
    st.title("AI Superforecaster")
    
    # Input form - always show at the top
    if 'forecast_started' not in st.session_state:
        st.markdown("""
        Enter a forecasting question you'd like to analyze. Good questions are specific and time-bound.
        
        Examples:
        - What is the probability that China will invade Taiwan before 2030?
        - Will Ethereum's price exceed $10,000 by the end of 2025?
        - What is the likelihood of a new pandemic with >1M global deaths by 2030?
        """)
        
        with st.form("forecast_question"):
            question = st.text_area("What would you like to forecast?", height=100)
            submitted = st.form_submit_button("Generate Forecast")
        
        if submitted and question:
            st.session_state.question = question
            st.session_state.forecast_started = True
            # We don't use the display_results function anymore since results
            # are displayed incrementally during run_forecast_workflow
            forecast_result = asyncio.run(run_forecast_workflow(question))
    
    # If a forecast has been started but not completed, show a message
    elif 'forecast_started' in st.session_state and not st.session_state.forecast_complete:
        st.info("Forecast in progress. Please wait while we generate your results.")
    
    # If forecast is complete, we don't need to do anything as the results
    # are already displayed by the run_forecast_workflow function

if __name__ == "__main__":
    main() 