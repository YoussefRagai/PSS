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

# Constants
MAX_SUMMARY_ITEMS = 5
DATA_PATH = "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack"
ANALYSIS_DATA_DIR = "/Users/youssefragai/Desktop/Master/UI/analysis"
INITIAL_COLUMNS_TO_LOAD_FROM_PARQUET = None

def load_combined_seasons(data_path):
    """Load and combine seasons data from msgpack file."""
    try:
        with open(data_path, 'rb') as f:
            return msgpack.unpackb(f.read())
    except Exception as e:
        st.error(f"Error loading seasons data: {e}")
        return None

def load_events_for_selected_matches(match_codes, analysis_dir, columns_to_load=None):
    """Load events data for selected matches."""
    try:
        all_events = []
        all_csv_data = []
        
        for match_code in match_codes:
            match_dir = os.path.join(analysis_dir, f'match_{match_code}')
            if os.path.exists(match_dir):
                # Load events
                events_file = os.path.join(match_dir, f'match_{match_code}_events.parquet')
                if os.path.exists(events_file):
                    events_df = pd.read_parquet(events_file, columns=columns_to_load)
                    events_df['match_code'] = match_code
                    all_events.append(events_df)
                
                # Load CSV data
                csv_file = os.path.join(match_dir, f'match_{match_code}_player_summary.csv')
                if os.path.exists(csv_file):
                    csv_df = pd.read_csv(csv_file)
                    csv_df['match_code'] = match_code
                    all_csv_data.append(csv_df)
        
        merged_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        merged_csv = pd.concat(all_csv_data, ignore_index=True) if all_csv_data else pd.DataFrame()
        
        return merged_events, merged_csv
    except Exception as e:
        st.error(f"Error loading match data: {e}")
        return pd.DataFrame(), pd.DataFrame()

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

# Initialize session state
if 'master_seasons' not in st.session_state:
    st.session_state.master_seasons = load_combined_seasons(DATA_PATH)
if 'available_matches' not in st.session_state:
    st.session_state.available_matches = [d.split('_')[1] for d in os.listdir(ANALYSIS_DATA_DIR) if d.startswith('match_')]
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

st.set_page_config(
    page_title="Football Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Sidebar for match selection and data loading
with st.sidebar:
    st.title("Match Selection")
    
    # Active group selection
    st.subheader("1. Active Group")
    active_group_index = st.radio(
        "Active Group:",
        [0, 1, 2],
        format_func=lambda x: st.session_state.group_names[x],
        key="active_group_selector_sidebar"
    )
    
    # Group renaming
    st.text_input(
        "Rename Selected Group:",
        value=st.session_state.group_names[active_group_index],
        key=f"rename_group_sidebar_{active_group_index}",
        on_change=update_group_name
    )
    
    st.markdown("---")
    
    # Load events section
    st.subheader("2. Load Events")
    total_selected_count = sum(len(group) for group in st.session_state.group_selections)
    st.caption(f"{total_selected_count} matches selected across all groups.")
    
    # Load events buttons for each group
    for group_idx in range(3):
        if st.button(f"🚀 Load Events for {st.session_state.group_names[group_idx]}", key=f"load_events_button_group_{group_idx}"):
            group_selections = st.session_state.group_selections[group_idx]
            if not group_selections:
                st.error(f"No matches selected in {st.session_state.group_names[group_idx]}.")
                continue
                
            # Get unique match codes
            match_codes = list(set(item['match_id'] for item in group_selections))
            
            # Load events and CSV data
            merged_df_result, csv_data = load_events_for_selected_matches(
                match_codes,
                ANALYSIS_DATA_DIR,
                columns_to_load=INITIAL_COLUMNS_TO_LOAD_FROM_PARQUET
            )
            
            if not merged_df_result.empty:
                st.session_state.loaded_events[group_idx] = merged_df_result
                st.success(f"Loaded {len(merged_df_result)} events for {st.session_state.group_names[group_idx]}.")
            else:
                st.warning(f"No event data loaded for selected matches in {st.session_state.group_names[group_idx]}.")
            
            if not csv_data.empty:
                st.session_state.loaded_csv[group_idx] = csv_data
                st.success(f"Loaded {len(csv_data)} CSV files:")
                for file_name, df in csv_data.groupby('match_code'):
                    st.info(f"- Match {file_name}: {len(df)} rows")
    
    # Proceed button
    if st.button("➡️ Proceed to Event Data", key="proceed_to_events"):
        st.session_state.show_event_data = True
    
    st.markdown("---")
    
    # Filters section
    st.subheader("3. Filters for Match List")
    
    # Get unique values for filters from master_seasons
    if st.session_state.master_seasons:
        all_tournaments = sorted(list(set(item.get('tournament_name', '') for item in st.session_state.master_seasons if item.get('tournament_name'))))
        all_seasons = sorted(list(set(item.get('season_name', '') for item in st.session_state.master_seasons if item.get('season_name'))))
        all_teams = sorted(list(set(item.get('home_team', '') for item in st.session_state.master_seasons if item.get('home_team'))))
        all_teams.extend(sorted(list(set(item.get('away_team', '') for item in st.session_state.master_seasons if item.get('away_team')))))
        all_teams = sorted(list(set(all_teams)))
        all_stadiums = sorted(list(set(item.get('stadium', '') for item in st.session_state.master_seasons if item.get('stadium'))))
        all_referees = sorted(list(set(item.get('referee', '') for item in st.session_state.master_seasons if item.get('referee'))))
    else:
        all_tournaments = all_seasons = all_teams = all_stadiums = all_referees = []
    
    # Filter widgets
    selected_tournaments = st.multiselect(
        "Tournaments:",
        all_tournaments,
        key=f"filter_tournaments_{active_group_index}",
        on_change=update_filter_state,
        args=('tournaments',)
    )
    
    selected_seasons = st.multiselect(
        "Seasons:",
        all_seasons,
        key=f"filter_seasons_{active_group_index}",
        on_change=update_filter_state,
        args=('seasons',)
    )
    
    selected_teams = st.multiselect(
        "Teams:",
        all_teams,
        key=f"filter_teams_{active_group_index}",
        on_change=update_filter_state,
        args=('teams',)
    )
    
    selected_stadiums = st.multiselect(
        "Stadiums:",
        all_stadiums,
        key=f"filter_stadiums_{active_group_index}",
        on_change=update_filter_state,
        args=('stadiums',)
    )
    
    selected_referees = st.multiselect(
        "Referees:",
        all_referees,
        key=f"filter_referees_{active_group_index}",
        on_change=update_filter_state,
        args=('referees',)
    )
    
    selected_result = st.selectbox(
        "Result:",
        ["All", "Win", "Draw", "Loss"],
        key=f"filter_result_{active_group_index}",
        on_change=update_filter_state,
        args=('result',)
    )
    
    selected_player_status = st.selectbox(
        "Player Status:",
        ["All", "Starter", "Sub"],
        key=f"filter_player_status_{active_group_index}",
        on_change=update_filter_state,
        args=('player_status',)
    )

# Main content area
st.title("Football Analysis Dashboard")

# Match list display
if st.session_state.master_seasons:
    # Get current filters
    current_filters = {
        'tournaments': st.session_state.get(f'filter_tournaments_{active_group_index}', []),
        'seasons': st.session_state.get(f'filter_seasons_{active_group_index}', []),
        'teams': st.session_state.get(f'filter_teams_{active_group_index}', []),
        'stadiums': st.session_state.get(f'filter_stadiums_{active_group_index}', []),
        'referees': st.session_state.get(f'filter_referees_{active_group_index}', []),
        'result': st.session_state.get(f'filter_result_{active_group_index}', "All"),
        'player_status': st.session_state.get(f'filter_player_status_{active_group_index}', "All")
    }
    
    # Filter matches
    filtered_matches = filter_matches(st.session_state.master_seasons, current_filters, active_group_index)
    
    # Pagination
    matches_per_page = 10
    total_pages = (len(filtered_matches) + matches_per_page - 1) // matches_per_page
    start_idx = (st.session_state.current_page - 1) * matches_per_page
    end_idx = start_idx + matches_per_page
    
    # Display matches
    for match in filtered_matches[start_idx:end_idx]:
        with st.expander(f"{match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Date: {match.get('date', 'Unknown')}")
                st.write(f"Tournament: {match.get('tournament_name', 'Unknown')}")
                st.write(f"Season: {match.get('season_name', 'Unknown')}")
                st.write(f"Stadium: {match.get('stadium', 'Unknown')}")
                st.write(f"Referee: {match.get('referee', 'Unknown')}")
            with col2:
                if st.button("Add to Group", key=f"add_match_{match.get('match_id')}"):
                    if match not in st.session_state.group_selections[active_group_index]:
                        st.session_state.group_selections[active_group_index].append(match)
                        st.success("Match added to group!")
                    else:
                        st.warning("Match already in group!")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous", disabled=(st.session_state.current_page <= 1)):
            st.session_state.current_page -= 1
    with col2:
        st.write(f"Page {st.session_state.current_page} of {total_pages}")
    with col3:
        if st.button("Next ▶", disabled=(st.session_state.current_page >= total_pages)):
            st.session_state.current_page += 1
else:
    st.error("No matches available in the master data source. Please check `DATA_PATH`.")

# Create tabs for visualization
if st.session_state.show_event_data:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pitch Analysis", 
        "Scatter Plot", 
        "Table View",
        "Bar Charts",
        "Radar Charts",
        "Pizza Charts"
    ])

    with tab1:
        render_pitch_tab(st, color_picker=color_picker, download_figure=download_figure)

    with tab2:
        render_scatter_tab(st, color_picker=color_picker, download_figure=download_figure)

    with tab3:
        render_table_tab(st, color_picker=color_picker, download_figure=download_figure)

    with tab4:
        render_bar_tab(st, color_picker=color_picker, download_figure=download_figure)

    with tab5:
        render_radar_tab(st, color_picker=color_picker, download_figure=download_figure)

    with tab6:
        render_pizza_tab(st, color_picker=color_picker, download_figure=download_figure) 