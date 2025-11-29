"""
Visualization functions for RCT analysis results
"""
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_coefficients(
    results: Dict,
    title: str = "Treatment Effect Estimates",
    outcome_label: Optional[str] = None
) -> go.Figure:
    """
    Create forest plot showing coefficient estimates with confidence intervals
    
    Args:
        results: Dictionary from estimation function with coef, ci_lower, ci_upper
        title: Plot title
        outcome_label: Label for outcome variable
    
    Returns:
        Plotly figure object
    """
    # Handle single result or multiple specifications
    if 'spec1' in results:
        # Multiple specifications from ITT
        specs = []
        coefs = []
        ci_lower = []
        ci_upper = []
        
        for spec_name in ['spec1', 'spec2', 'spec3']:
            if spec_name in results and 'coef' in results[spec_name]:
                spec = results[spec_name]
                specs.append(spec_name.replace('spec', 'Specification '))
                coefs.append(spec['coef'])
                ci_lower.append(spec['ci_lower'])
                ci_upper.append(spec['ci_upper'])
    else:
        # Single estimate
        specs = [outcome_label or 'Treatment Effect']
        coefs = [results.get('coef', results.get('late', results.get('marginal_effect', 0)))]
        ci_lower = [results.get('ci_lower', results.get('late_ci_lower', 0))]
        ci_upper = [results.get('ci_upper', results.get('late_ci_upper', 0))]
    
    # Calculate error bars
    error_minus = [coefs[i] - ci_lower[i] for i in range(len(coefs))]
    error_plus = [ci_upper[i] - coefs[i] for i in range(len(coefs))]
    
    # Create figure
    fig = go.Figure()
    
    # Add coefficient points with error bars
    fig.add_trace(go.Scatter(
        x=coefs,
        y=specs,
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus
        ),
        mode='markers',
        marker=dict(size=10, color='darkblue'),
        name='Estimate'
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Treatment Effect",
        yaxis_title="",
        height=max(300, len(specs) * 80),
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def plot_distributions(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    bins: int = 50,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create overlaid histograms or kernel density plots comparing treatment vs control
    
    Pattern from Emerick et al (2015) Figure 2
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of treatment column
        bins: Number of bins for histogram
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Filter data
    df = data[[outcome_var, treatment_col]].dropna()
    
    # Separate treatment and control
    treatment = df[df[treatment_col] == 1][outcome_var]
    control = df[df[treatment_col] == 0][outcome_var]
    
    # Create figure
    fig = go.Figure()
    
    # Add histograms
    fig.add_trace(go.Histogram(
        x=control,
        name='Control',
        opacity=0.6,
        nbinsx=bins,
        histnorm='probability density',
        marker_color='lightgray'
    ))
    
    fig.add_trace(go.Histogram(
        x=treatment,
        name='Treatment',
        opacity=0.6,
        nbinsx=bins,
        histnorm='probability density',
        marker_color='darkblue'
    ))
    
    # Update layout
    fig.update_layout(
        title=title or f"Distribution of {outcome_var}",
        xaxis_title=outcome_var,
        yaxis_title="Density",
        barmode='overlay',
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_heterogeneity(
    results: Dict,
    title: str = "Treatment Effects by Subgroup"
) -> go.Figure:
    """
    Create plot showing subgroup-specific treatment effects
    
    Args:
        results: Dictionary from estimate_heterogeneity with 'subgroups' key
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    if 'subgroups' not in results:
        raise ValueError("Results must contain 'subgroups' key")
    
    subgroups = results['subgroups']
    
    # Extract data
    labels = []
    effects = []
    ses = []
    
    for subgroup_name, subgroup_data in subgroups.items():
        if 'effect' in subgroup_data:
            labels.append(str(subgroup_name))
            effects.append(subgroup_data['effect'])
            ses.append(subgroup_data['se'])
    
    # Calculate confidence intervals
    ci_lower = [effects[i] - 1.96 * ses[i] for i in range(len(effects))]
    ci_upper = [effects[i] + 1.96 * ses[i] for i in range(len(effects))]
    
    # Calculate error bars
    error_minus = [effects[i] - ci_lower[i] for i in range(len(effects))]
    error_plus = [ci_upper[i] - effects[i] for i in range(len(effects))]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=effects,
        y=labels,
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus
        ),
        mode='markers',
        marker=dict(size=10, color='darkgreen'),
        name='Subgroup Effect'
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Treatment Effect",
        yaxis_title="Subgroup",
        height=max(300, len(labels) * 60),
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def plot_balance(
    balance_df: pd.DataFrame,
    title: str = "Balance Check Results"
) -> go.Figure:
    """
    Visualize balance table results
    
    Args:
        balance_df: DataFrame from generate_balance_table()
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Filter out joint F-test row for visualization
    df = balance_df[balance_df['Variable'] != 'Joint F-test'].copy()
    
    if len(df) == 0:
        raise ValueError("No variables to plot")
    
    # Calculate standardized differences (difference / pooled SD)
    df['std_diff'] = df['Difference'] / ((df['Treatment_SD'] + df['Control_SD']) / 2)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for standardized differences
    colors = ['red' if abs(x) > 0.25 else 'lightblue' for x in df['std_diff']]
    
    fig.add_trace(go.Bar(
        y=df['Variable'],
        x=df['std_diff'],
        orientation='h',
        marker_color=colors,
        text=df['Stars'],
        textposition='outside'
    ))
    
    # Add vertical lines at ±0.25 (rule of thumb for imbalance)
    fig.add_vline(x=0.25, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=-0.25, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=0, line_color="black", opacity=0.3)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Standardized Difference (Treatment - Control)",
        yaxis_title="",
        height=max(400, len(df) * 30),
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def plot_kde_comparison(
    data: pd.DataFrame,
    outcome_var: str,
    treatment_col: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create kernel density estimate comparison (smoother than histogram)
    
    Args:
        data: DataFrame containing the data
        outcome_var: Name of outcome variable
        treatment_col: Name of treatment column
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    from scipy import stats
    import numpy as np
    
    # Filter data
    df = data[[outcome_var, treatment_col]].dropna()
    
    # Separate treatment and control
    treatment = df[df[treatment_col] == 1][outcome_var]
    control = df[df[treatment_col] == 0][outcome_var]
    
    # Calculate KDE
    kde_treatment = stats.gaussian_kde(treatment)
    kde_control = stats.gaussian_kde(control)
    
    # Create range for x-axis
    x_min = df[outcome_var].min()
    x_max = df[outcome_var].max()
    x_range = np.linspace(x_min, x_max, 200)
    
    # Evaluate KDE
    y_treatment = kde_treatment(x_range)
    y_control = kde_control(x_range)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_control,
        name='Control',
        line=dict(color='gray', width=2),
        fill='tozeroy',
        opacity=0.3
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_treatment,
        name='Treatment',
        line=dict(color='darkblue', width=2, dash='dash'),
        fill='tozeroy',
        opacity=0.3
    ))
    
    # Update layout
    fig.update_layout(
        title=title or f"Distribution of {outcome_var}",
        xaxis_title=outcome_var,
        yaxis_title="Density",
        template="plotly_white",
        height=400
    )
    
    return fig
