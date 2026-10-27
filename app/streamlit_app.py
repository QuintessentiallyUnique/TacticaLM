"""
Streamlit frontend for TacticaLM.
"""

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="TacticaLM | Football Analysis",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
MULTI_MATCH_VALUE = "__multi_match__"


if "question" not in st.session_state:
    st.session_state.question = ""

def fetch_matches():
    """Fetches available matches from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/matches")
        if response.status_code == 200:
            return response.json().get("matches", [])
        return []
    except requests.exceptions.RequestException:
        return []

def fetch_timeline(match_id: int):
    """Fetches momentum timeline for a match."""
    try:
        response = requests.get(f"{API_BASE_URL}/matches/{match_id}/timeline")
        if response.status_code == 200:
            return response.json().get("momentum_timeline", [])
        return []
    except requests.exceptions.RequestException:
        return []

def generate_analysis(match_id: int, team: str, question: str):
    """Calls the API to generate analysis."""
    payload = {
        "match_id": match_id,
        "team": team,
        "question": question
    }
    try:
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.error("Model is not currently loaded on the backend. Please check the server logs.")
            return None
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None


def generate_multi_match_analysis(team: str, question: str, top_k: int):
    """Calls the API to generate retrieval-augmented multi-match analysis."""
    payload = {
        "team": team or None,
        "question": question,
        "top_k": top_k,
    }
    try:
        response = requests.post(f"{API_BASE_URL}/analyze/multi-match", json=payload)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 503:
            st.error("Model is not currently loaded on the backend. Please check the server logs.")
            return None
        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {e}")
        return None


def render_timeline_chart(fig: go.Figure) -> str | None:
    """Render a timeline chart and return the clicked interval when supported."""
    try:
        from streamlit_plotly_events import plotly_events
    except ImportError:
        st.plotly_chart(fig, use_container_width=True)
        return None

    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=450,
        key="momentum_timeline",
    )
    if not selected_points:
        return None
    interval = selected_points[0].get("x")
    return str(interval) if interval else None

# --- UI Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    .analysis-container {
        background-color: #1E293B;
        color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/football2.png", width=64)
    st.title("TacticaLM Settings")
    st.markdown("Configure your analysis request.")
    
    matches = fetch_matches()
    match_options = {"Multi-match retrieval": MULTI_MATCH_VALUE}
    if matches:
        for m in matches:
            if m.get('home_team') and m.get('away_team'):
                label = f"{m['home_team']} vs {m['away_team']} (Match {m['match_id']})"
            else:
                label = f"Match {m['match_id']}"
            match_options[label] = m['match_id']
    
    selected_match_label = st.selectbox("Select Match", options=list(match_options.keys()))
    selected_match_id = match_options[selected_match_label]
    is_multi_match = selected_match_id == MULTI_MATCH_VALUE
    
    selected_match_data = (
        next((m for m in matches if m['match_id'] == selected_match_id), None)
        if matches and not is_multi_match
        else None
    )
    if is_multi_match:
        selected_team = st.text_input(
            "Team Filter",
            value="",
            help="Optional. Leave blank to retrieve across all teams.",
        )
        top_k = st.slider("Retrieved Matches", min_value=1, max_value=10, value=5)
    elif selected_match_data and selected_match_data.get('home_team') and selected_match_data.get('away_team'):
        team_options = [selected_match_data['home_team'], selected_match_data['away_team']]
        selected_team = st.selectbox("Team Focus", options=team_options, help="Select the team to focus the analysis on.")
        top_k = 1
    else:
        selected_team = st.text_input("Team Focus", value="Home Team", help="Enter the team to focus the analysis on.")
        top_k = 1
    
    st.markdown("---")
    st.markdown("### API Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
        if health.get("status") == "healthy":
            st.success("Backend: Online & Ready")
        else:
            st.warning(f"Backend: {health.get('status')}")
    except requests.exceptions.RequestException:
        st.error("Backend: Offline")
        
    st.markdown("---")
    with st.expander("How to add more matches?"):
        st.markdown(
            "To add a new match to this list, you need to download and process it using the TacticaLM data pipeline CLI:\n\n"
            "```bash\n"
            "python scripts/data_pipeline.py --match-id <STATS_BOMB_MATCH_ID>\n"
            "```\n\n"
            "This will automatically download the event data, run the feature engine, and create the summary. "
            "Refresh this page afterwards to see it in the dropdown."
        )

# --- Main Content ---
st.markdown('<div class="main-header">TacticaLM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Football Tactical Analysis</div>', unsafe_allow_html=True)

# Input section
st.markdown("### What would you like to analyze?")
question = st.text_area(
    "Enter your tactical question",
    placeholder="e.g. How did the team exploit the wide areas in the second half?",
    height=100,
    key="question",
)

generate_btn = st.button("Generate Analysis", type="primary", use_container_width=True)

# Results section
if generate_btn:
    if not selected_match_id:
        st.warning("Please select a valid match.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing tactical data..."):
            if is_multi_match:
                result = generate_multi_match_analysis(selected_team, question, top_k)
            else:
                result = generate_analysis(selected_match_id, selected_team, question)
            
            if result:
                # Create columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### Tactical Analysis")
                    st.markdown(f'<div class="analysis-container">{result["response"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Processing time: {result['processing_time']:.2f}s")

                    if result.get("retrieved_matches"):
                        st.markdown("### Retrieved Matches")
                        st.dataframe(pd.DataFrame(result["retrieved_matches"]), use_container_width=True)
                    
                    timeline_data = [] if is_multi_match else fetch_timeline(selected_match_id)
                    if timeline_data:
                        st.markdown("---")
                        st.markdown("### Momentum Timeline (Possession %)")
                        df = pd.DataFrame(timeline_data)
                        if not df.empty:
                            df['interval_label'] = df['start_minute'].astype(str) + "-" + df['end_minute'].astype(str) + "'"
                            
                            fig = go.Figure()
                            for t in df['team'].unique():
                                team_df = df[df['team'] == t]
                                fig.add_trace(go.Scatter(
                                    x=team_df['interval_label'],
                                    y=team_df['possession_pct'],
                                    mode='lines+markers',
                                    name=t,
                                    line={"width": 3},
                                    marker={"size": 8}
                                ))
                                
                            fig.update_layout(
                                xaxis_title="Match Interval",
                                yaxis_title="Possession (%)",
                                template="plotly_dark",
                                hovermode="x unified",
                                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                                legend={
                                    "orientation": "h",
                                    "yanchor": "bottom",
                                    "y": 1.02,
                                    "xanchor": "right",
                                    "x": 1
                                }
                            )
                            clicked_interval = render_timeline_chart(fig)
                            if clicked_interval:
                                st.session_state.question = (
                                    f"Analyze the tactical shift for {selected_team} "
                                    f"between {clicked_interval}."
                                )
                                st.rerun()
                    
                with col2:
                    st.markdown("### Evidence Used")
                    for i, ev in enumerate(result["evidence"]):
                        with st.expander(f"Evidence #{i+1}", expanded=(i==0)):
                            st.write(ev)
