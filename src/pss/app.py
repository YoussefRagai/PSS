import streamlit as st
import io
import os
import msgpack
import pandas as pd
from datetime import datetime
from .pitch_tab import render_pitch_tab
from .scatter_tab import render_scatter_tab
from .table_tab import render_table_tab
from .bar_tab import render_bar_tab
from .radar_tab import render_radar_tab
from .pizza_tab import render_pizza_tab
from .config import DATA_PATH, ANALYSIS_DATA_DIR, get_data_paths
from pathlib import Path
import numpy as np

# Constants
MAX_SUMMARY_ITEMS = 5
INITIAL_COLUMNS_TO_LOAD_FROM_PARQUET = None

def load_combined_seasons():
    """Load the combined seasons data"""
    try:
        return pd.read_msgpack(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading master seasons data: {str(e)}")
        st.error(f"Please ensure the file exists at: {DATA_PATH}")
        return None

def load_events_for_selected_matches(match_ids):
    """Load events for selected matches"""
    if not match_ids:
        return None
    
    all_events = []
    for match_id in match_ids:
        match_dir = Path(ANALYSIS_DATA_DIR) / match_id
        if not match_dir.exists():
            st.warning(f"Match directory not found: {match_dir}")
            continue
            
        try:
            events_file = match_dir / "events.msgpack"
            if events_file.exists():
                events = pd.read_msgpack(events_file)
                all_events.append(events)
        except Exception as e:
            st.warning(f"Error loading events for match {match_id}: {str(e)}")
            continue
    
    if not all_events:
        return None
        
    return pd.concat(all_events, ignore_index=True)

def update_group_name():
    """Update the name of the active group."""
    active_group = st.session_state.active_group_selector_sidebar
    new_name = st.session_state[f'rename_group_sidebar_{active_group}']
    st.session_state.group_names[active_group] = new_name

def update_filter_state(filter_type, value):
    """Update the filter state for the active group."""
    active_group = st.session_state.active_group_selector_sidebar
    if f'filter_{filter_type}_{active_group}' not in st.session_state:
        st.session_state[f'filter_{filter_type}_{active_group}'] = []
    st.session_state[f'filter_{filter_type}_{active_group}'] = value

def filter_matches(matches, filters, active_group):
    """Filter matches based on selected criteria."""
    filtered_matches = matches.copy()
    
    # Apply tournament filter
    if filters.get('tournaments'):
        filtered_matches = [m for m in filtered_matches if m.get('tournament_name') in filters['tournaments']]
    
    # Apply season filter
    if filters.get('seasons'):
        filtered_matches = [m for m in filtered_matches if m.get('season_name') in filters['seasons']]
    
    # Apply team filter
    if filters.get('teams'):
        filtered_matches = [m for m in filtered_matches if m.get('home_team') in filters['teams'] or m.get('away_team') in filters['teams']]
    
    # Apply stadium filter
    if filters.get('stadiums'):
        filtered_matches = [m for m in filtered_matches if m.get('stadium') in filters['stadiums']]
    
    # Apply referee filter
    if filters.get('referees'):
        filtered_matches = [m for m in filtered_matches if m.get('referee') in filters['referees']]
    
    # Apply result filter
    if filters.get('result') != "All":
        if filters['result'] == "Win":
            filtered_matches = [m for m in filtered_matches if m.get('home_score', 0) > m.get('away_score', 0)]
        elif filters['result'] == "Draw":
            filtered_matches = [m for m in filtered_matches if m.get('home_score', 0) == m.get('away_score', 0)]
        elif filters['result'] == "Loss":
            filtered_matches = [m for m in filtered_matches if m.get('home_score', 0) < m.get('away_score', 0)]
    
    return filtered_matches

def color_picker(label, default_color, key):
    """Wrapper for streamlit color picker with consistent interface."""
    return st.color_picker(label, default_color, key=key)

def download_figure(fig):
    """Save figure to bytes buffer and return filename."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return buf, 'pitch_visualization.png'

def initialize_app():
    """Initialize the Streamlit app"""
    # Initialize session state variables if they don't exist
    if 'master_seasons' not in st.session_state:
        st.session_state.master_seasons = load_combined_seasons()
        if st.session_state.master_seasons is None:
            st.error("Failed to load master seasons data. Please check the data file location.")
            return

    if 'available_matches' not in st.session_state:
        st.session_state.available_matches = []
        analysis_dir = Path(ANALYSIS_DATA_DIR)
        if analysis_dir.exists():
            st.session_state.available_matches = [d.name for d in analysis_dir.iterdir() if d.is_dir()]
        else:
            st.error(f"Analysis directory not found: {ANALYSIS_DATA_DIR}")
            return

    if 'saved_datasets' not in st.session_state:
        st.session_state.saved_datasets = {}
    if 'group_names' not in st.session_state:
        st.session_state.group_names = ["Group 1", "Group 2", "Group 3"]
    if 'group_selections' not in st.session_state:
        st.session_state.group_selections = [[], [], []]
    if 'loaded_events' not in st.session_state:
        st.session_state.loaded_events = [None, None, None]
    if 'loaded_csv' not in st.session_state:
        st.session_state.loaded_csv = [None, None, None]
    if 'show_event_data' not in st.session_state:
        st.session_state.show_event_data = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    # Sidebar for match selection and filters
    with st.sidebar:
        st.title("Match Selection")
        
        # Match filters
        st.subheader("Filters")
        
        # Tournament filter
        tournaments = sorted(st.session_state.master_seasons['tournament'].unique())
        selected_tournaments = st.multiselect(
            "Tournament",
            options=tournaments,
            default=tournaments[:1] if tournaments else []
        )
        
        # Season filter
        seasons = sorted(st.session_state.master_seasons['season'].unique())
        selected_seasons = st.multiselect(
            "Season",
            options=seasons,
            default=seasons[:1] if seasons else []
        )
        
        # Team filter
        teams = sorted(st.session_state.master_seasons['team'].unique())
        selected_teams = st.multiselect(
            "Team",
            options=teams,
            default=teams[:1] if teams else []
        )
        
        # Stadium filter
        stadiums = sorted(st.session_state.master_seasons['stadium'].unique())
        selected_stadiums = st.multiselect(
            "Stadium",
            options=stadiums,
            default=stadiums[:1] if stadiums else []
        )
        
        # Referee filter
        referees = sorted(st.session_state.master_seasons['referee'].unique())
        selected_referees = st.multiselect(
            "Referee",
            options=referees,
            default=referees[:1] if referees else []
        )
        
        # Result filter
        results = sorted(st.session_state.master_seasons['result'].unique())
        selected_results = st.multiselect(
            "Result",
            options=results,
            default=results[:1] if results else []
        )
        
        # Player status filter
        player_statuses = sorted(st.session_state.master_seasons['player_status'].unique())
        selected_player_statuses = st.multiselect(
            "Player Status",
            options=player_statuses,
            default=player_statuses[:1] if player_statuses else []
        )

    # Filter matches based on selections
    filtered_matches = st.session_state.master_seasons.copy()
    
    if selected_tournaments:
        filtered_matches = filtered_matches[filtered_matches['tournament'].isin(selected_tournaments)]
    if selected_seasons:
        filtered_matches = filtered_matches[filtered_matches['season'].isin(selected_seasons)]
    if selected_teams:
        filtered_matches = filtered_matches[filtered_matches['team'].isin(selected_teams)]
    if selected_stadiums:
        filtered_matches = filtered_matches[filtered_matches['stadium'].isin(selected_stadiums)]
    if selected_referees:
        filtered_matches = filtered_matches[filtered_matches['referee'].isin(selected_referees)]
    if selected_results:
        filtered_matches = filtered_matches[filtered_matches['result'].isin(selected_results)]
    if selected_player_statuses:
        filtered_matches = filtered_matches[filtered_matches['player_status'].isin(selected_player_statuses)]

    # Display filtered matches
    st.sidebar.subheader("Available Matches")
    
    # Pagination
    matches_per_page = 10
    total_matches = len(filtered_matches)
    total_pages = (total_matches + matches_per_page - 1) // matches_per_page
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    
    with col1:
        if st.button("Previous") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
    
    with col2:
        st.write(f"Page {st.session_state.current_page} of {total_pages}")
    
    with col3:
        if st.button("Next") and st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
    
    start_idx = (st.session_state.current_page - 1) * matches_per_page
    end_idx = min(start_idx + matches_per_page, total_matches)
    
    for idx in range(start_idx, end_idx):
        match = filtered_matches.iloc[idx]
        with st.sidebar.expander(f"{match['team']} vs {match['opponent']} ({match['date']})"):
            st.write(f"Tournament: {match['tournament']}")
            st.write(f"Season: {match['season']}")
            st.write(f"Stadium: {match['stadium']}")
            st.write(f"Referee: {match['referee']}")
            st.write(f"Result: {match['result']}")
            st.write(f"Player Status: {match['player_status']}")
            
            if st.button("Add to Group", key=f"add_{idx}"):
                match_id = match['match_id']
                if match_id not in st.session_state.saved_datasets:
                    st.session_state.saved_datasets[match_id] = {
                        'match_details': match.to_dict(),
                        'events': None,
                        'player_summary': None
                    }
                    st.success(f"Added match {match_id} to group")
                else:
                    st.warning("Match already in group")

    # Load events for selected matches
    if st.session_state.saved_datasets:
        st.sidebar.subheader("Selected Matches")
        for match_id in list(st.session_state.saved_datasets.keys()):
            if st.sidebar.button(f"Remove {match_id}", key=f"remove_{match_id}"):
                del st.session_state.saved_datasets[match_id]
                st.rerun()

        if st.sidebar.button("Load Event Data"):
            events_data = load_events_for_selected_matches(st.session_state.saved_datasets.keys())
            if events_data is not None:
                st.session_state.events_data = events_data
                st.session_state.show_event_data = True
                st.success("Event data loaded successfully!")
            else:
                st.error("Failed to load event data for selected matches")

    # Main content area
    if st.session_state.get('show_event_data', False):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Pitch Analysis", "Scatter Plot", "Table View",
            "Bar Charts", "Radar Charts", "Pizza Charts"
        ])
        
        with tab1:
            render_pitch_tab(st, st.session_state.events_data)
        with tab2:
            render_scatter_tab(st, st.session_state.events_data)
        with tab3:
            render_table_tab(st, st.session_state.events_data)
        with tab4:
            render_bar_tab(st, st.session_state.events_data)
        with tab5:
            render_radar_tab(st, st.session_state.events_data)
        with tab6:
            render_pizza_tab(st, st.session_state.events_data)
    else:
        st.info("Please select matches and load event data to view visualizations.")

st.set_page_config(
    page_title="Football Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

initialize_app() 