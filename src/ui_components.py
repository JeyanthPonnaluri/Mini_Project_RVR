"""
Professional UI components for Federated Learning Research Lab.
Provides reusable, styled components for a clean research dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List


# Professional color palette
COLORS = {
    'primary': '#1f77b4',      # Deep blue
    'secondary': '#ff7f0e',    # Soft orange
    'accent': '#2ca02c',       # Green
    'warning': '#d62728',      # Red
    'neutral': '#7f7f7f',      # Grey
    'background': '#f8f9fa',   # Light grey
}


def render_header():
    """Render professional header with title and metadata."""
    st.markdown("""
        <style>
        .main-header {
            padding: 1.5rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 2rem;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin: 0;
            line-height: 1.2;
        }
        .main-subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        .header-meta {
            display: flex;
            gap: 1.5rem;
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #888;
        }
        .badge {
            background: #1f77b4;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        </style>
        
        <div class="main-header">
            <div class="main-title">🧬 Federated Learning Research Lab</div>
            <div class="main-subtitle">Advanced Heterogeneity & Sustainability Analysis for Healthcare AI</div>
            <div class="header-meta">
                <span class="badge">Version 5.0</span>
                <span>TCGA-PRAD Clinical Stage Classification</span>
                <span>📊 Multi-Modal Federated Learning</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_card(title: str, content: str = None, metrics: Dict[str, Any] = None):
    """
    Render a styled card component.
    
    Parameters:
    -----------
    title : str
        Card title
    content : str, optional
        Card content (markdown)
    metrics : Dict[str, Any], optional
        Dictionary of metrics to display
    """
    st.markdown(f"""
        <style>
        .card {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 4px solid #1f77b4;
        }}
        .card-title {{
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 1rem;
        }}
        </style>
        
        <div class="card">
            <div class="card-title">{title}</div>
    """, unsafe_allow_html=True)
    
    if content:
        st.markdown(content)
    
    if metrics:
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(key, value.get('value'), value.get('delta'))
                else:
                    st.metric(key, value)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_metrics_row(metrics: List[Dict[str, Any]], columns: int = 3):
    """
    Render a row of metrics with consistent styling.
    
    Parameters:
    -----------
    metrics : List[Dict[str, Any]]
        List of metric dictionaries with 'label', 'value', 'delta', 'help'
    columns : int
        Number of columns
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta'),
                help=metric.get('help')
            )


def render_section_header(title: str, subtitle: str = None):
    """
    Render a section header with consistent styling.
    
    Parameters:
    -----------
    title : str
        Section title
    subtitle : str, optional
        Section subtitle
    """
    st.markdown(f"""
        <style>
        .section-header {{
            margin-top: 2rem;
            margin-bottom: 1rem;
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.25rem;
        }}
        .section-subtitle {{
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 1rem;
        }}
        </style>
        
        <div class="section-header">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_divider():
    """Render a subtle section divider."""
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(to right, transparent, #e0e0e0, transparent); margin: 2rem 0;"></div>
    """, unsafe_allow_html=True)


def render_info_box(content: str, box_type: str = 'info'):
    """
    Render an information box using native Streamlit components.
    
    Parameters:
    -----------
    content : str
        Box content (plain text or markdown)
    box_type : str
        'info', 'success', 'warning', 'error'
    """
    if box_type == 'info':
        st.info(content)
    elif box_type == 'success':
        st.success(content)
    elif box_type == 'warning':
        st.warning(content)
    elif box_type == 'error':
        st.error(content)
    else:
        st.info(content)


def render_experiment_status(status: str, message: str = None):
    """
    Render experiment status indicator.
    
    Parameters:
    -----------
    status : str
        'running', 'complete', 'error', 'idle'
    message : str, optional
        Status message
    """
    status_config = {
        'running': ('🔄', '#ff7f0e', 'Running'),
        'complete': ('✅', '#2ca02c', 'Complete'),
        'error': ('❌', '#d62728', 'Error'),
        'idle': ('⏸️', '#7f7f7f', 'Idle')
    }
    
    icon, color, label = status_config.get(status, status_config['idle'])
    
    st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: {color}15;
            border-left: 4px solid {color};
            border-radius: 4px;
            margin: 1rem 0;
        ">
            <span style="font-size: 1.5rem;">{icon}</span>
            <div>
                <div style="font-weight: 600; color: {color};">{label}</div>
                {f'<div style="font-size: 0.9rem; color: #666;">{message}</div>' if message else ''}
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_comparison_table(df: pd.DataFrame, highlight_best: bool = True):
    """
    Render a styled comparison table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to display
    highlight_best : bool
        Highlight best values
    """
    if highlight_best and 'final_auc' in df.columns:
        # Style the dataframe
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #e8f5e9' if v else '' for v in is_max]
        
        styled_df = df.style.apply(highlight_max, subset=['final_auc'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def render_footer():
    """Render professional footer."""
    st.markdown("""
        <style>
        .footer {
            margin-top: 4rem;
            padding: 2rem 0 1rem 0;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #888;
            font-size: 0.85rem;
        }
        .footer-links {
            margin-top: 0.5rem;
        }
        .footer-links a {
            color: #1f77b4;
            text-decoration: none;
            margin: 0 1rem;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
        </style>
        
        <div class="footer">
            <div>Federated Learning Research Framework</div>
            <div style="margin-top: 0.25rem;">Developed by Jeyanth Ponnaluri</div>
            <div class="footer-links">
                <a href="https://github.com/JeyanthPonnaluri/Mini_Project_RVR" target="_blank">GitHub</a>
                <span style="color: #ddd;">|</span>
                <a href="#" target="_blank">Documentation</a>
                <span style="color: #ddd;">|</span>
                <a href="#" target="_blank">Citation</a>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_section(title: str, icon: str = ""):
    """
    Render a sidebar section header.
    
    Parameters:
    -----------
    title : str
        Section title
    icon : str
        Optional icon
    """
    st.sidebar.markdown(f"""
        <div style="
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
        ">
            {icon} {title}
        </div>
    """, unsafe_allow_html=True)


def render_progress_indicator(progress: float, message: str = ""):
    """
    Render a progress indicator.
    
    Parameters:
    -----------
    progress : float
        Progress value (0.0 to 1.0)
    message : str
        Progress message
    """
    st.progress(progress)
    if message:
        st.caption(message)


def render_key_findings(findings: List[str]):
    """
    Render key findings using native Streamlit components.
    
    Parameters:
    -----------
    findings : List[str]
        List of finding strings (plain text)
    """
    for i, finding in enumerate(findings, 1):
        st.markdown(f"{i}. {finding}")
        if i < len(findings):
            st.markdown("")  # Add spacing


def apply_custom_css():
    """Apply global custom CSS for professional styling."""
    st.markdown("""
        <style>
        /* Global styles */
        .main {
            background-color: #fafafa;
        }
        
        /* Remove default padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.05rem;
        }
        
        /* DataFrame styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Remove extra spacing */
        .element-container {
            margin-bottom: 0.5rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-weight: 500;
            font-size: 1rem;
        }
        
        /* Version card styling */
        .version-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .version-card:hover {
            border-color: #1f77b4;
            box-shadow: 0 4px 12px rgba(31, 119, 180, 0.15);
            transform: translateY(-2px);
        }
        
        .version-card.active {
            border-color: #1f77b4;
            background: linear-gradient(135deg, #f0f7ff 0%, #e3f2fd 100%);
            box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
        }
        
        .version-number {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .version-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin: 0.5rem 0 0.25rem 0;
        }
        
        .version-description {
            font-size: 0.9rem;
            color: #666;
            line-height: 1.5;
        }
        
        .version-tags {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
            flex-wrap: wrap;
        }
        
        .version-tag {
            background: #f0f0f0;
            color: #555;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)


def render_version_selector():
    """
    Render an interactive version selector with cards.
    Returns the selected version string.
    """
    st.markdown("### Select Research Version")
    
    versions = {
        "VERSION-1": {
            "title": "Centralized Learning",
            "description": "Traditional machine learning with sklearn's LogisticRegression for baseline performance",
            "tags": ["Baseline", "sklearn", "Centralized"],
            "icon": "🖥️"
        },
        "VERSION-2": {
            "title": "Federated Learning (FedAvg)",
            "description": "Distributed training across hospitals using Federated Averaging algorithm",
            "tags": ["FedAvg", "Distributed", "NumPy"],
            "icon": "🌐"
        },
        "VERSION-3": {
            "title": "Sustainability Analysis",
            "description": "Study scalability, free-rider behavior, and data heterogeneity effects",
            "tags": ["Scalability", "Free-Rider", "Monte Carlo"],
            "icon": "🔬"
        },
        "VERSION-4": {
            "title": "FedProx & Non-IID Study",
            "description": "Compare FedAvg vs FedProx under data heterogeneity with Dirichlet partitioning",
            "tags": ["FedProx", "Non-IID", "Convergence"],
            "icon": "⚡"
        },
        "VERSION-5": {
            "title": "Research Lab",
            "description": "Advanced analysis with contribution measurement and experiment management",
            "tags": ["Contribution", "Multi-Modal", "Publication-Ready"],
            "icon": "🧬"
        }
    }
    
    # Initialize session state
    if 'selected_version' not in st.session_state:
        st.session_state.selected_version = "VERSION-1"
    
    # Create columns for version cards
    cols = st.columns(2)
    
    for idx, (version_key, version_info) in enumerate(versions.items()):
        col_idx = idx % 2
        with cols[col_idx]:
            # Determine button type based on selection
            is_selected = st.session_state.selected_version == version_key
            button_type = "primary" if is_selected else "secondary"
            
            # Create a container for each version
            container = st.container()
            with container:
                if st.button(
                    f"{version_info['icon']} {version_key}",
                    key=f"btn_{version_key}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.selected_version = version_key
                    st.rerun()
                
                st.markdown(f"**{version_info['title']}**")
                st.caption(version_info['description'])
                
                # Display tags
                tag_text = " • ".join(version_info['tags'])
                st.markdown(f"<small style='color: #888;'>{tag_text}</small>", unsafe_allow_html=True)
                
                st.markdown("")  # Spacing
    
    return st.session_state.selected_version
