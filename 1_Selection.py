# -----------------------------
# Final Version: Football Explorer UI (with Full Patch)
# -----------------------------

import streamlit as st
import msgpack
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import datetime
import json
import os # Added for path checking
import math # Added for pagination ceiling
import glob

# Set page config (must be the first Streamlit command)
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Football Match Selection", page_icon="📊")

# --- Configuration ---
# Use relative paths from the workspace root
# DATA_DIR = "DATA" # Not currently used directly for match listing/loading
MATCHES_DIR = os.path.join("UI", "analysis") # Correct path where match_{id} folders reside
ANALYSIS_BASE_PATH = MATCHES_DIR # Define ANALYSIS_BASE_PATH here as well for clarity within button logic
# Consider making this path relative or configurable later
DATA_PATH = "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack" # Path for the main season metadata

# ==================================================
# START: ALL Function Definitions
# ==================================================

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data
def load_combined_seasons(msgpack_path):
    # Add error handling for file loading (Point 5)
    if not os.path.exists(msgpack_path):
        st.error(f"Error: Data file not found at {msgpack_path}")
        st.stop()
        return None # Return None if file doesn't exist
    try:
        with open(msgpack_path, "rb") as f:
            seasons_data = msgpack.unpack(f, raw=False)
        return seasons_data
    except msgpack.UnpackException as e:
        st.error(f"Error decoding data file: {e}")
        st.stop()
        return None # Return None on decoding error
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop()
        return None # Return None on other errors

@st.cache_data
def fetch_logo(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def style_score(home_score, away_score):
    if home_score > away_score:
        return "background-color: green; padding: 5px 10px; border-radius: 10px; color: white;"
    elif home_score == away_score:
        return "background-color: gray; padding: 5px 10px; border-radius: 10px; color: white;"
    else:
        return "background-color: gray; padding: 5px 10px; border-radius: 10px; color: white;"

# -----------------------------
# Callback Functions
# -----------------------------

# --- Sidebar Callbacks ---
def update_group_name():
    active_group_index = st.session_state.active_group_index
    new_name = st.session_state[f"rename_group_{active_group_index}"]
    if new_name: # Ensure it's not empty
        st.session_state.group_names[active_group_index] = new_name

def update_filter_state(filter_key, widget_key):
    """Callback to update the active group's filter state."""
    st.session_state.group_filters[st.session_state.active_group_index][filter_key] = st.session_state[widget_key]

# --- Pagination Callback ---
def change_page(exp_key, delta):
    st.session_state[exp_key] += delta

# --- Bulk Action Logic --- 
def bulk_select(action_type, matches_list, active_group_index, n=None):
    # Ensure the group index is valid
    if not 0 <= active_group_index < len(st.session_state.selected_matches_by_group):
        st.error(f"Invalid active_group_index: {active_group_index}")
        return
        
    active_group_list = st.session_state.selected_matches_by_group[active_group_index]
    
    current_selection_set = set((item['match_id'], item['season_id']) for item in active_group_list)
    match_identifiers_to_process = []
    if action_type == 'select_last_n' and n is not None:
        match_identifiers_to_process = [
            {"match_id": m['match_info']['match_id'], "season_id": m['season_id']}
            for m in matches_list[:n]
        ]
    else:
        match_identifiers_to_process = [
            {"match_id": m['match_info']['match_id'], "season_id": m['season_id']}
            for m in matches_list
        ]
    ids_to_process_set = set((item['match_id'], item['season_id']) for item in match_identifiers_to_process)
    if action_type.startswith('select'):
        # Add matches not already selected
        added_count = 0
        for identifier in match_identifiers_to_process:
            id_tuple = (identifier['match_id'], identifier['season_id'])
            if id_tuple not in current_selection_set:
                active_group_list.append(identifier)
                added_count += 1
    elif action_type == 'deselect':
        # Explicitly build a new list, excluding items to be deselected
        new_active_group_list = []
        removed_count = 0
        for item in active_group_list:
            item_tuple = (item['match_id'], item['season_id'])
            if item_tuple not in ids_to_process_set:
                new_active_group_list.append(item)
            else:
                removed_count += 1
        # Assign the new list back to session state
        st.session_state.selected_matches_by_group[active_group_index] = new_active_group_list

# --- Individual Match Checkbox Callback ---
def handle_individual_checkbox(match_key, match_identifier, active_group_index):
    # Ensure the group index is valid
    if not 0 <= active_group_index < len(st.session_state.selected_matches_by_group):
        st.error(f"Invalid active_group_index in checkbox callback: {active_group_index}")
        return
        
    current_group_list = st.session_state.selected_matches_by_group[active_group_index]
    id_tuple = (match_identifier['match_id'], match_identifier['season_id'])
    is_currently_selected = any(item['match_id'] == id_tuple[0] and item['season_id'] == id_tuple[1] for item in current_group_list)

    if st.session_state[match_key]: # If checkbox widget is checked
        if not is_currently_selected:
            current_group_list.append(match_identifier)
    else: # If checkbox widget is unchecked
        if is_currently_selected:
            st.session_state.selected_matches_by_group[active_group_index] = [
                item for item in current_group_list 
                if not (item['match_id'] == id_tuple[0] and item['season_id'] == id_tuple[1])
            ]

# --- Bulk Action Callbacks ---
def handle_bulk_action_change(selectbox_key, exp_key, season_matches, active_group_index):
    selected_action = st.session_state[selectbox_key]
    show_last_n_key = f"show_last_n_input_{exp_key}"
    
    # Reset visibility flag first
    if show_last_n_key in st.session_state:
         st.session_state[show_last_n_key] = False

    if selected_action == "Select All Matches":
        bulk_select('select', season_matches, active_group_index)
        st.session_state[selectbox_key] = "--Select Action--" # Reset dropdown
        st.rerun() # Rerun to reflect selection and reset dropdown
    elif selected_action == "Deselect All Matches":
        bulk_select('deselect', season_matches, active_group_index)
        st.session_state[selectbox_key] = "--Select Action--" # Reset dropdown
        st.rerun() # Rerun to reflect deselection and reset dropdown
    elif selected_action == "Select Last N Matches":
        # Set flag to show the N input and confirm button
        st.session_state[show_last_n_key] = True
        # Don't reset dropdown immediately, user needs to confirm N
        # Rerun happens implicitly due to state change
    # Else (--Select Action-- selected), do nothing

def handle_select_last_n(exp_key, season_matches, active_group_index):
        num_last_matches_key = f"num_last_{exp_key}"
        show_last_n_key = f"show_last_n_input_{exp_key}"
        selectbox_key = f"bulk_action_select_{exp_key}"
        
        if num_last_matches_key not in st.session_state: 
             st.session_state[num_last_matches_key] = 5
        n_to_select = st.session_state[num_last_matches_key]
        
        # Perform the selection
        bulk_select('select_last_n', season_matches, active_group_index, n=n_to_select)
        
        # Hide the N input/button section again
        st.session_state[show_last_n_key] = False
        # Reset the main action dropdown
        st.session_state[selectbox_key] = "--Select Action--"
        st.rerun() # Rerun to hide controls and show selection

# --- Function to display group summary in sidebar ---
def display_sidebar_group_summary(group_index):
    st.sidebar.markdown("**Selected Matches in this Group:**")
    group_selections = st.session_state.selected_matches_by_group[group_index]
    
    if group_selections:
        # Limit displayed matches in sidebar for brevity if needed
        MAX_SUMMARY_ITEMS = 10 
        display_count = 0
        for item in reversed(group_selections): # Show most recent selections first
            if display_count >= MAX_SUMMARY_ITEMS:
                st.sidebar.caption(f"...and {len(group_selections) - MAX_SUMMARY_ITEMS} more")
                break
                
            match_key_tuple = (item['match_id'], item['season_id'])
            match_data = match_lookup.get(match_key_tuple) # Use global match_lookup
            
            if match_data:
                home_team = match_data['teams']['home']['team_info']['name']
                away_team = match_data['teams']['away']['team_info']['name']
                home_score = match_data['teams']['home']['score']
                away_score = match_data['teams']['away']['score']
                date_str = match_data['match_info'].get('date')
                date_display = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%y") if date_str else "N/A"
                
                # Use caption for smaller text in sidebar
                st.sidebar.caption(f"{date_display} | {home_team} {home_score}-{away_score} {away_team}")
            else:
                st.sidebar.caption(f"M:{item['match_id']} (S:{item['season_id']}) - Error finding details")
            display_count += 1
    else:
        st.sidebar.info("No matches selected in this group yet.")
# --- End sidebar summary function ---

# --- Helper Functions ---
def list_available_matches(matches_folder):
    """Lists available match IDs by finding folders starting with 'match_'"""
    available_matches = []
    if os.path.isdir(matches_folder):
        for item in os.listdir(matches_folder):
            # Check if it's a directory AND starts with 'match_' and the rest is digits
            if os.path.isdir(os.path.join(matches_folder, item)) and item.startswith('match_') and item.split('_')[-1].isdigit():
                try:
                    match_id = int(item.split('_')[-1]) # Extract the ID
                    available_matches.append(match_id)
                except (ValueError, IndexError):
                    pass # Ignore folders that don't match the exact pattern
    return sorted(available_matches)

# ==================================================
# END: ALL Function Definitions
# ==================================================

# -----------------------------
# Load Data
# -----------------------------

# Consider making this path relative or configurable
DATA_PATH = "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack" # Use relative path
seasons_data = load_combined_seasons(DATA_PATH)

# Exit if data loading failed
if seasons_data is None:
    st.stop()

# -----------------------------
# Prepare Filters
# -----------------------------

all_tournaments = sorted(list(set(s['tournament_name'] for s in seasons_data['seasons'])), reverse=True)
all_seasons = sorted(list(set(s['season_name'] for s in seasons_data['seasons'])), reverse=True)
all_matches = []

all_teams = set()
all_players = set()
all_stadiums = set()
all_referees = set()

for season in seasons_data['seasons']:
    for match in season['matches']:
        match['season_id'] = season['season_id']
        match['season_name'] = season['season_name']
        match['tournament_name'] = season['tournament_name']
        all_matches.append(match)

        home_team = match['teams']['home']['team_info']['name']
        away_team = match['teams']['away']['team_info']['name']
        all_teams.update([home_team, away_team])

        for p in match['teams']['home']['players']['starting'] + match['teams']['home']['players']['subs']:
            all_players.add(p['player_name'])
        for p in match['teams']['away']['players']['starting'] + match['teams']['away']['players']['subs']:
            all_players.add(p['player_name'])

        stadium = match['match_info'].get('stadium')
        referee = match['match_info'].get('referee')

        if stadium and isinstance(stadium, str):
            all_stadiums.add(stadium)
        if referee and isinstance(referee, str):
            all_referees.add(referee)

all_teams = sorted(list(all_teams))
all_players = sorted(list(all_players))
all_stadiums = sorted(list(all_stadiums))
all_referees = sorted(list(all_referees))

# --- Create a lookup dictionary for faster match retrieval by ID --- 
match_lookup = {
    (match['match_info']['match_id'], match['season_id']): match 
    for match in all_matches
}
# --- End lookup dictionary ---

# Initialize session state for persistent storage
# Remove old grouping state
# if "current_group" not in st.session_state:
#     st.session_state.current_group = []
# if "saved_groups" not in st.session_state:
#     st.session_state.saved_groups = []

# New state for multi-group management
if "active_group_index" not in st.session_state:
    st.session_state.active_group_index = 0 # Start with Group 1
if "group_names" not in st.session_state: # ADDED: Store custom group names
    st.session_state.group_names = [f"Group {i+1}" for i in range(3)]
if "group_filters" not in st.session_state:
    # List of dicts, one for each group's filter settings
    st.session_state.group_filters = [
        {"tournaments": [], "seasons": [], "teams": [], "players": [], "stadiums": [], "referees": [], "result": "All", "player_status": "All"} for _ in range(3)
    ]
if "selected_matches_by_group" not in st.session_state:
    # List of lists, one for each group's selected matches [{match_id: X, season_id: Y}, ...]
    st.session_state.selected_matches_by_group = [[] for _ in range(3)]

# -----------------------------
# Sidebar - Now controls Active Group and its Filters
# -----------------------------
st.sidebar.title("📊 Group Selection & Filters")

# Select Active Group - Use custom names
active_group_index = st.sidebar.radio(
    "Select Group:",
    [0, 1, 2], # 0-indexed for list access
    format_func=lambda x: st.session_state.group_names[x], # Use custom names
    key="active_group_selector"
)
st.session_state.active_group_index = active_group_index # Update state if radio changes

# Rename Active Group
st.sidebar.text_input(
    "Rename Selected Group:", 
    value=st.session_state.group_names[active_group_index], 
    key=f"rename_group_{active_group_index}",
    on_change=update_group_name,
    # help="Enter a new name and press Enter."
)

# --- Display Summary for Active Group --- 
display_sidebar_group_summary(st.session_state.active_group_index)
# --- End Summary Display ---

# --- Add Proceed Button ---
# Define base path for analysis data
ANALYSIS_BASE_PATH = "UI/analysis" 

st.sidebar.markdown("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
# Calculate total selected across all groups
total_selected_count = sum(len(st.session_state.selected_matches_by_group[i]) for i in range(3))
st.sidebar.subheader(f"Proceed with {total_selected_count} total selected matches")

if st.sidebar.button("🚀 Proceed", key="proceed_button", disabled=total_selected_count == 0):
    # --- Process selected matches to load detailed events, metrics, and player info ---
    all_selected_identifiers = []
    for i in range(3):
        all_selected_identifiers.extend(st.session_state.selected_matches_by_group[i])

    # Deduplicate identifiers (in case a match is in multiple groups)
    unique_match_ids = set() # Store only match IDs
    seen_ids = set()
    for item in all_selected_identifiers:
        id_tuple = (item['match_id'], item['season_id'])
        if id_tuple not in seen_ids:
            unique_match_ids.add(item['match_id']) # Add only the match_id
            seen_ids.add(id_tuple)

    if not unique_match_ids:
        st.sidebar.warning("No matches selected across any group.")
    else:
        with st.spinner(f"Loading event data for {len(unique_match_ids)} matches..."):
            # --- Initialize Data Containers ---
            all_event_dfs = []
            all_raw_metrics_data = []
            all_player_info_dfs = [] # This will hold the data from *_player_info_with_metrics.csv
            # REMOVED: all_agg_player_metrics = [] 

            # --- File Paths and Counters ---
            event_files_found = 0
            metrics_files_found = 0
            player_info_files_found = 0 # This counter tracks *_player_info_with_metrics.csv
            # REMOVED: agg_metrics_files_found = 0
            processed_ids_count = 0
            error_messages = [] # Keep error messages
            missing_files = [] # Keep missing files list
            
            for match_id in unique_match_ids:
                match_folder_name = f"match_{match_id}"
                match_folder_path = os.path.join(MATCHES_DIR, match_folder_name)
                files_found_for_match = False

                # Define file paths
                event_file_path = os.path.join(match_folder_path, f"match_{match_id}_events.parquet")
                metrics_file_path = os.path.join(match_folder_path, f"match_{match_id}_combined_metrics.msgpack")
                # This path correctly targets the desired CSV file
                player_info_file_path = os.path.join(match_folder_path, f"match_{match_id}_player_info_with_metrics.csv") 

                # --- Load Events (Parquet) --- (No changes needed here)
                if os.path.exists(event_file_path): 
                    try: # ... load events ... 
                        df = pd.read_parquet(event_file_path); df['source_match_id'] = match_id; all_event_dfs.append(df); files_found_for_match = True; event_files_found += 1
                    except Exception as e: error_messages.append(f"E: {event_file_path}: {e}")
                else: missing_files.append(os.path.basename(event_file_path))

                # --- Load Metrics (Msgpack) --- (No changes needed here)
                if os.path.exists(metrics_file_path): 
                    try: # ... load metrics ...
                        with open(metrics_file_path, "rb") as f: data = msgpack.unpack(f, raw=False)
                        if isinstance(data, list): # Add source_id
                            for item in data: 
                                if isinstance(item, dict): item['source_match_id'] = match_id
                        elif isinstance(data, dict): data['source_match_id'] = match_id
                        all_raw_metrics_data.append(data); files_found_for_match = True; metrics_files_found += 1
                    except Exception as e: error_messages.append(f"M: {metrics_file_path}: {e}")
                else: missing_files.append(os.path.basename(metrics_file_path))

                # --- Load Player Info / Aggregated Metrics (CSV) --- 
                # This block now correctly loads the file needed for Tabs 2, 3, 6 and 7
                if os.path.exists(player_info_file_path):
                    try:
                        df = pd.read_csv(player_info_file_path)
                        df['source_match_id'] = match_id 
                        all_player_info_dfs.append(df) # Add to this list
                        files_found_for_match = True
                        player_info_files_found += 1 # Use this counter
                    except Exception as e:
                        error_messages.append(f"P: {player_info_file_path}: {e}")
                else:
                    # Use the correct filename in the missing message
                    missing_files.append(os.path.basename(player_info_file_path)) 
                
                # REMOVED Block 4: The redundant loading attempt for the same CSV file.

                if files_found_for_match:
                    processed_ids_count += 1

            # --- Consolidate Data & Store in Session State ---
            final_dfs = {}
            # Consolidate Events (No change)
            if all_event_dfs: st.session_state['detailed_events_df'] = pd.concat(all_event_dfs, ignore_index=True); final_dfs['Events'] = len(st.session_state['detailed_events_df'])
            else: st.session_state['detailed_events_df'] = pd.DataFrame()
            
            # Store Raw Metrics (No change)
            if all_raw_metrics_data: st.session_state['raw_metrics_data'] = all_raw_metrics_data; final_dfs['Metrics Matches'] = len(all_raw_metrics_data)
            else: st.session_state['raw_metrics_data'] = []
            st.session_state['combined_metrics_df'] = pd.DataFrame() # Keep this empty for now

            # Consolidate Player Info (This now contains the aggregated metrics)
            if all_player_info_dfs:
                st.session_state['combined_player_info_df'] = pd.concat(all_player_info_dfs, ignore_index=True)
                final_dfs['Player Info/Metrics'] = len(st.session_state['combined_player_info_df'])
            else:
                st.session_state['combined_player_info_df'] = pd.DataFrame()
                st.warning(f"No '*_player_info_with_metrics.csv' files were found or loaded.") # Make warning more specific

            # REMOVED: Consolidation logic for aggregated_player_metrics_df
            # Ensure the old state variable is removed if it exists
            if 'aggregated_player_metrics_df' in st.session_state: 
                del st.session_state['aggregated_player_metrics_df']

            if missing_files:
                 st.sidebar.warning(f"Missing files: {', '.join(sorted(list(set(missing_files))))}")
            if error_messages:
                 with st.sidebar.expander("Show Loading Errors"):
                     for err in error_messages: st.error(err)

            # Check if *any* data relevant for visualizations was loaded
            data_loaded_for_viz = final_dfs.get('Events', 0) > 0 or final_dfs.get('Metrics Matches', 0) > 0 or final_dfs.get('Player Info/Metrics', 0) > 0

            if data_loaded_for_viz:
                summary_message = f"Loaded data for {processed_ids_count} matches: " + ", ".join([f"{k} ({v})" for k, v in final_dfs.items()])
                st.sidebar.success(summary_message)
                # Clear old manual upload state
                if 'df' in st.session_state: del st.session_state['df']
                if 'original_df' in st.session_state: del st.session_state['original_df']
                # Navigate
                try: st.switch_page("pages/2_⚽_Visualizations.py")
                except AttributeError: st.warning("...") # Keep navigation
            else:
                st.sidebar.error("No usable data files could be loaded for the selected matches.")

st.sidebar.divider() # Separator after button

st.sidebar.markdown(f"**Filters for {st.session_state.group_names[active_group_index]}**") # Use custom name
st.sidebar.caption("You can type in the filter boxes below to search.")

# Get the filter state for the currently active group
current_filters = st.session_state.group_filters[active_group_index]

# Use unique keys for widgets based on group index to prevent state conflicts
selected_tournaments = st.sidebar.multiselect(
    "Competition Name:", all_tournaments, 
    default=current_filters['tournaments'], 
    key=f"filter_tournaments_{active_group_index}",
    on_change=update_filter_state, args=('tournaments', f"filter_tournaments_{active_group_index}")
)
selected_seasons = st.sidebar.multiselect(
    "Season:", all_seasons, 
    default=current_filters['seasons'], 
    key=f"filter_seasons_{active_group_index}",
    on_change=update_filter_state, args=('seasons', f"filter_seasons_{active_group_index}")
)
selected_teams = st.sidebar.multiselect(
    "Teams:", all_teams, 
    default=current_filters['teams'], 
    key=f"filter_teams_{active_group_index}",
    on_change=update_filter_state, args=('teams', f"filter_teams_{active_group_index}")
)
selected_players = st.sidebar.multiselect(
    "Players:", all_players, 
    default=current_filters['players'], 
    key=f"filter_players_{active_group_index}",
    on_change=update_filter_state, args=('players', f"filter_players_{active_group_index}")
)
selected_stadiums = st.sidebar.multiselect(
    "Stadiums:", all_stadiums, 
    default=current_filters['stadiums'], 
    key=f"filter_stadiums_{active_group_index}",
    on_change=update_filter_state, args=('stadiums', f"filter_stadiums_{active_group_index}")
)
selected_referees = st.sidebar.multiselect(
    "Referees:", all_referees, 
    default=current_filters['referees'], 
    key=f"filter_referees_{active_group_index}",
    on_change=update_filter_state, args=('referees', f"filter_referees_{active_group_index}")
)
selected_result = st.sidebar.selectbox(
    "Match Result (for selected Team):", ["All", "Win", "Draw", "Loss"], 
    index=["All", "Win", "Draw", "Loss"].index(current_filters['result']), 
    key=f"filter_result_{active_group_index}",
    on_change=update_filter_state, args=('result', f"filter_result_{active_group_index}")
)
selected_player_status = st.sidebar.selectbox(
    "Player Status:", ["All", "Starter", "Sub"], 
    index=["All", "Starter", "Sub"].index(current_filters['player_status']), 
    key=f"filter_player_status_{active_group_index}",
    on_change=update_filter_state, args=('player_status', f"filter_player_status_{active_group_index}")
)

# -----------------------------
# Main Page Layout
# -----------------------------

# REMOVE st.columns for main layout
# main_content_col, summary_col = st.columns([4, 1])

# Code previously in main_content_col now runs directly
# with main_content_col:
st.title("⚽ Football Explorer")
st.caption("Select matches and group them for analysis!")

# Apply filters before grouping based on the ACTIVE group's filters
filtered_matches = []
active_filters = st.session_state.group_filters[st.session_state.active_group_index]

# Retrieve filter values directly from state for the active group
_selected_tournaments = active_filters['tournaments']
_selected_seasons = active_filters['seasons']
_selected_teams = active_filters['teams']
_selected_players = active_filters['players']
_selected_stadiums = active_filters['stadiums']
_selected_referees = active_filters['referees']
_selected_result = active_filters['result']
_selected_player_status = active_filters['player_status']

for match in all_matches:
    home = match['teams']['home']
    away = match['teams']['away']
    match_info = match['match_info']
    home_name = home['team_info']['name']
    away_name = away['team_info']['name']
    home_score = home['score']
    away_score = away['score']

    # Basic Filters - Use filter values for the active group
    tournament_pass = not _selected_tournaments or match['tournament_name'] in _selected_tournaments
    season_pass = not _selected_seasons or match['season_name'] in _selected_seasons
    team_pass = not _selected_teams or home_name in _selected_teams or away_name in _selected_teams
    stadium_pass = not _selected_stadiums or match_info.get('stadium') in _selected_stadiums
    referee_pass = not _selected_referees or match_info.get('referee') in _selected_referees

    # Player Filter - Use filter values for the active group
    player_pass = True
    if _selected_players:
        player_found = False
        if _selected_player_status == "Starter":
            if any(p['player_name'] in _selected_players for p in home['players']['starting'] + away['players']['starting']):
                player_found = True
        elif _selected_player_status == "Sub":
            if any(p['player_name'] in _selected_players for p in home['players']['subs'] + away['players']['subs']):
                player_found = True
        else: # "All"
            if any(p['player_name'] in _selected_players for p in home['players']['starting'] + home['players']['subs'] + away['players']['starting'] + away['players']['subs']):
                player_found = True
        player_pass = player_found

    # Result Filter - Use filter values for the active group
    result_pass = True
    if _selected_result != "All" and _selected_teams:
        match_result_satisfied = False
        for team_name in _selected_teams:
            if team_name == home_name:
                if _selected_result == "Win" and home_score > away_score: match_result_satisfied = True; break
                if _selected_result == "Draw" and home_score == away_score: match_result_satisfied = True; break
                if _selected_result == "Loss" and home_score < away_score: match_result_satisfied = True; break
            elif team_name == away_name:
                if _selected_result == "Win" and away_score > home_score: match_result_satisfied = True; break
                if _selected_result == "Draw" and home_score == away_score: match_result_satisfied = True; break
                if _selected_result == "Loss" and away_score < home_score: match_result_satisfied = True; break
        result_pass = match_result_satisfied

    if tournament_pass and season_pass and team_pass and stadium_pass and referee_pass and player_pass and result_pass:
        filtered_matches.append(match)

# Group the filtered matches
seasons_grouped = {}
for match in filtered_matches: # Use filtered_matches
    key = (match['tournament_name'], match['season_name'])
    if key not in seasons_grouped:
        seasons_grouped[key] = []
    seasons_grouped[key].append(match)

seasons_sorted = sorted(seasons_grouped.keys(), reverse=True)

# Display Logic
if not seasons_grouped:
    st.info("No matches found matching the selected criteria.")

ITEMS_PER_PAGE = 10 # Point 4: Items per page for pagination

for tournament_name, season_name in seasons_sorted:
    expander_key = f"expander_{tournament_name}_{season_name}" # Unique key for expander state
    expander_state_key = f"page_{tournament_name}_{season_name}"

    # Initialize page number for this expander in session state if it doesn't exist
    if expander_state_key not in st.session_state:
        st.session_state[expander_state_key] = 1

    with st.expander(f"{season_name} - {tournament_name}", expanded=False):
        matches_in_group = sorted(seasons_grouped[(tournament_name, season_name)], key=lambda x: (x['match_info']['date'], x['match_info'].get('time', '00:00:00')), reverse=True)
        total_items = len(matches_in_group)
        total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))
        current_page = st.session_state[expander_state_key]

        # --- Pagination Controls --- 
        
        cols_pagination = st.columns([2, 1, 1, 1, 2]) # Layout: spacer, prev, page#, next, spacer
        
        with cols_pagination[1]:
            st.button(
                "←", 
                key=f"prev_{expander_state_key}", 
                on_click=change_page, 
                args=(expander_state_key, -1), 
                disabled=current_page <= 1
            )
        
        with cols_pagination[2]:
            st.write(f"Page {current_page}/{total_pages}", style={"text-align": "center"})

        with cols_pagination[3]:
            st.button(
                "→", 
                key=f"next_{expander_state_key}", 
                on_click=change_page, 
                args=(expander_state_key, 1), 
                disabled=current_page >= total_pages
            )
        
        # --- NEW Bulk Selection Controls (Dropdown Menu) --- 
        st.markdown("***") # Visual separator

        action_selectbox_key = f"bulk_action_select_{expander_state_key}"
        action_options = [
            "--Select Action--", 
            "Select All Matches", 
            "Deselect All Matches", 
            "Select Last N Matches"
        ]
        
        # We need the callback defined before the widget that uses it
        # Define handle_bulk_action_change globally later
        
        st.selectbox(
            "Season Actions:",
            options=action_options,
            key=action_selectbox_key,
            index=0, # Default to placeholder
            on_change=handle_bulk_action_change, # Will define this globally
            args=(action_selectbox_key, expander_state_key, matches_in_group, st.session_state.active_group_index)
        )

        # --- Conditional Controls for Select Last N --- 
        # These will appear only when the action is selected
        show_last_n_key = f"show_last_n_input_{expander_state_key}"
        if st.session_state.get(show_last_n_key, False):
            num_last_matches_key = f"num_last_{expander_state_key}"
            if num_last_matches_key not in st.session_state:
                 st.session_state[num_last_matches_key] = 5 # Default N

            cols_last_n = st.columns([1, 2])
            with cols_last_n[0]:
                st.number_input(
                    "Number (N):", 
                    min_value=1, 
                    max_value=len(matches_in_group), 
                    key=num_last_matches_key, 
                    label_visibility="visible"
                )
            with cols_last_n[1]:
                st.button(
                    "Confirm Select Last N", 
                    key=f"confirm_last_n_{expander_state_key}",
                    on_click=handle_select_last_n, # Re-use existing logic handler
                    args=(expander_state_key, matches_in_group, st.session_state.active_group_index)
                )
                # We might need to hide this section again after button click
                # Add logic in handle_select_last_n or a separate callback if needed

        st.markdown("***") # Visual separator

        # --- Calculate paginated matches BEFORE defining buttons that use it ---
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        paginated_matches = matches_in_group[start_idx:end_idx]
        # --- End Calculation ---

        # --- Display Matches for the Current Page ---
        for match in paginated_matches: # Iterate through paginated matches
            home = match['teams']['home']
            away = match['teams']['away']
            match_info = match['match_info']
            match_identifier = {"match_id": match_info['match_id'], "season_id": match['season_id']}
            match_key = f"match_{match_info['match_id']}_{tournament_name}_{season_name}_{st.session_state.active_group_index}"

            with st.container():
                # Use columns to place checkbox on the left
                match_cols = st.columns([1, 12]) # Narrow col for checkbox, wide for content

                with match_cols[0]: # Checkbox Column
                    # --- Checkbox Logic --- 
                    is_selected_in_active_group = any(
                        item['match_id'] == match_identifier['match_id'] and item['season_id'] == match_identifier['season_id']
                        for item in st.session_state.selected_matches_by_group[st.session_state.active_group_index]
                    )
                    
                    st.checkbox(
                        "", # No Label
                        value=is_selected_in_active_group, 
                        key=match_key,
                        on_change=handle_individual_checkbox,
                        args=(match_key, match_identifier, st.session_state.active_group_index) 
                    )

                with match_cols[1]: # Content Column
                    # Place rest of the card content here
                    date_str = match_info.get('date')
                    time_str = match_info.get('time')
                    date_display = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y") if date_str else "N/A"
                    time_display = datetime.datetime.strptime(time_str, "%H:%M:%S").strftime("%I:%M %p") if time_str else ""
                    st.markdown(f"<p style='text-align: center; font-size:16px; color:#666'>{date_display} {('• '+time_display) if time_display else ''}</p>", unsafe_allow_html=True)

                    # --- Start: HTML/Markdown for score/teams WITH LOGOS ---
                    home_name = home['team_info']['name']
                    away_name = away['team_info']['name']
                    home_score = home['score']
                    away_score = away['score']
                    score_style = style_score(home_score, away_score)
                    home_logo_url = home['team_info'].get('logo')
                    away_logo_url = away['team_info'].get('logo')

                    # Build HTML parts conditionally
                    home_logo_html = f'<img src="{home_logo_url}" style="height: 25px; width: auto; margin-left: 5px;" alt="">' if home_logo_url else ""
                    away_logo_html = f'<img src="{away_logo_url}" style="height: 25px; width: auto; margin-right: 5px;" alt="">' if away_logo_url else ""

                    html_layout = f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 5px;">
                        <div style="flex: 5; display: flex; align-items: center; justify-content: flex-end;">
                            <span style="font-size:20px;">{home_name}</span>
                            {home_logo_html}
                        </div>
                        <div style="flex: 2; text-align: center;"><span style="{score_style}">{home_score} - {away_score}</span></div>
                        <div style="flex: 5; display: flex; align-items: center; justify-content: flex-start;">
                            {away_logo_html}
                            <span style="font-size:20px;">{away_name}</span>
                        </div>
                    </div>
                    """
                    st.markdown(html_layout, unsafe_allow_html=True)
                    # --- End: HTML/Markdown ---

                    if home.get("goals") or away.get("goals"):
                        # Use HTML for goal layout instead of nested columns
                        home_goals_html = "".join([f"<div style='text-align:right; font-size:15px;'>{g['scorer']} {g['minute']}\'</div>" for g in home.get("goals", [])])
                        away_goals_html = "".join([f"<div style='text-align:left; font-size:15px;'>{g['scorer']} {g['minute']}\'</div>" for g in away.get("goals", [])])
                        
                        goal_layout_html = f"""
                        <div style="display: flex; width: 100%; margin-top: 5px; justify-content: center; align-items: flex-start;">
                            <div style="flex: 5; max-width: 40%;">{home_goals_html}</div>
                            <div style="flex: 2; text-align:center; min-width: 50px; padding-top: 5px;">&#9917;&#65039;</div>
                            <div style="flex: 5; max-width: 40%;">{away_goals_html}</div>
                        </div>
                        """
                        st.markdown(goal_layout_html, unsafe_allow_html=True)
                        # Add vertical space after goals are displayed
                        st.markdown(" ") 

                    # Referee and Stadium info
                    stadium_display = match_info.get('stadium', 'N/A')
                    referee_display = match_info.get('referee', 'N/A')
                    st.markdown(f"<p style='text-align: center; color: #aaa; font-size:15px;'>🏟️ {stadium_display} | 🧑‍⚖️ {referee_display}</p>", unsafe_allow_html=True)
                    
                    st.divider()

# Ensure the script ends properly if the block was the last thing
pass

# --- Configuration ---
# Use relative paths from the workspace root
DATA_DIR = "DATA"
# MATCHES_DIR = os.path.join(DATA_DIR, "matches") # Old incorrect path
MATCHES_DIR = os.path.join("UI", "analysis") # Correct path based on user sample
# TEMP_DATA_FILE = "UI/temp_analysis_data.parquet" # No longer used for raw data
