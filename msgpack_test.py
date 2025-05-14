import streamlit as st
import msgpack
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import datetime
import json
import os
import math
import glob
import re
from scatter_tab import render_scatter_tab
from pitch_tab import render_pitch_tab
from radar_tab import render_radar_tab
from bar_tab import render_bar_tab
from pizza_tab import render_pizza_tab
from table_tab import render_table_tab
import sys

# --- Configuration & Session State Initialization ---
# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Football Match Event Explorer", page_icon="⚽")

# --- Configuration ---
DATA_PATH = "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack"
ANALYSIS_DATA_DIR = "/Users/youssefragai/Desktop/Master/UI/analysis"
INITIAL_COLUMNS_TO_LOAD_FROM_PARQUET = None

COLUMNS_FOR_DYNAMIC_FILTER_CONFIG = [
    # Group 2
    {"name": "match_display_name", "type": "multiselect", "label": "Match", "group": 2, "source_col": "match_code_source"},
    {"name": "team", "type": "multiselect", "label": "Team", "group": 2},
    {"name": "nickname", "type": "multiselect", "label": "Player Nickname", "group": 2},
    {"name": "half", "type": "multiselect", "label": "Half", "group": 2},
    {"name": "min", "type": "slider", "label": "Minute", "group": 2},
    {"name": "x", "type": "slider", "label": "X Coordinate", "group": 2},
    {"name": "y", "type": "slider", "label": "Y Coordinate", "group": 2},
    {"name": "end_x", "type": "slider_or_progressive", "label": "End X / Progressive", "group": 2},
    {"name": "end_y", "type": "slider", "label": "End Y", "group": 2},
    {"name": "end_nickname", "type": "multiselect", "label": "Receiver Nickname", "group": 2},
    # Group 1
    {"name": "category", "type": "multiselect", "label": "Category", "group": 1},
    {"name": "event", "type": "multiselect", "label": "Event Type", "group": 1},
    {"name": "result", "type": "multiselect", "label": "Result", "group": 1},
    {"name": "extra", "type": "multiselect", "label": "Extra Info", "group": 1},
]
ALL_OPTION_LABEL = "--- ALL ---"

# Add this after the imports
METRIC_FORMULAS = {
    # === Shooting ===
    "Shooting": {
        "event": ["Shoot", "Penalty", "OneONOne", "FreeKick"],
        "result": ["Goal", "OnTarget", "Save"]
    },
    # === Passing ===
    "Passing": {
        "event": ["Pass", "EffectivePass", "GK Long Pass", "GK Pass", "GKHand", "LongPass"],
        "result": ["Success"]
    },
    # === Progressive Passes ===
    "Progressive Passes": {
        "event": ["Pass", "EffectivePass", "GK Long Pass", "GK Pass", "GKHand", "LongPass"],
        "result": ["Success"],
        "end_x": {"type": "progressive", "value": 5}  # At least 5 units forward
    },
    # === Carries ===
    "Carries": {
        "event": ["Carry", "Dribble"]
    },
    # === Defensive Actions ===
    "Defensive Actions": {
        "event": ["Intercept", "Clear", "TackleClear", "Tackle", "Block", "Recover"],
        "result": ["Success"]
    },
    # === Crosses ===
    "Crosses": {
        "event": ["Cross"],
        "result": ["Success"],
        "end_x": {"type": "range", "value": (83, 100)},
        "end_y": {"type": "range", "value": (18, 82)}
    },
    # === Errors ===
    "Errors": {
        "extra": ["ErrorLeadToGoal", "ErrorLeadToOpportunity"]
    },
    # === Assists ===
    "Assists": {
        "extra": ["Assist"]
    }
}

# Add this after the imports
if "saved_datasets" not in st.session_state:
    st.session_state.saved_datasets = {}
if "saved_player_datasets" not in st.session_state:
    st.session_state.saved_player_datasets = {}
print("TOP OF SCRIPT: saved_player_datasets", st.session_state.get("saved_player_datasets"), file=sys.stderr)

# --- All Helper Functions (load_combined_seasons, fetch_logo, etc. - PASTE THEM HERE) ---
@st.cache_data
def load_combined_seasons(msgpack_path):
    if not os.path.exists(msgpack_path):
        st.error(f"Error: Data file not found at {msgpack_path}")
        st.stop()
        return None
    try:
        with open(msgpack_path, "rb") as f:
            seasons_data = msgpack.unpack(f, raw=False)
        return seasons_data
    except msgpack.UnpackException as e:
        st.error(f"Error decoding data file: {e}")
        st.stop()
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop()
        return None

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
        return "background-color: #F08080; padding: 5px 10px; border-radius: 10px; color: white;"

def update_group_name():
    active_group_index = st.session_state.active_group_index
    new_name = st.session_state[f"rename_group_{active_group_index}"]
    if new_name:
        st.session_state.group_names[active_group_index] = new_name

def update_filter_state(filter_key, widget_key): # For match list filters
    st.session_state.group_filters[st.session_state.active_group_index][filter_key] = st.session_state[widget_key]

def change_page(exp_key, delta):
    st.session_state[exp_key] += delta

def bulk_select(action_type, matches_list, active_group_index, n=None):
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
    if action_type.startswith('select'):
        for identifier in match_identifiers_to_process:
            id_tuple = (identifier['match_id'], identifier['season_id'])
            if id_tuple not in current_selection_set:
                active_group_list.append(identifier)
    elif action_type == 'deselect':
        ids_to_deselect_set = set((item['match_id'], item['season_id']) for item in match_identifiers_to_process)
        st.session_state.selected_matches_by_group[active_group_index] = [
            item for item in active_group_list
            if (item['match_id'], item['season_id']) not in ids_to_deselect_set
        ]

def handle_individual_checkbox(match_key, match_identifier, active_group_index):
    if not 0 <= active_group_index < len(st.session_state.selected_matches_by_group):
        st.error(f"Invalid active_group_index in checkbox callback: {active_group_index}")
        return
    current_group_list = st.session_state.selected_matches_by_group[active_group_index]
    id_tuple = (match_identifier['match_id'], match_identifier['season_id'])
    is_currently_selected = any(item['match_id'] == id_tuple[0] and item['season_id'] == id_tuple[1] for item in current_group_list)
    if st.session_state[match_key]:
        if not is_currently_selected:
            current_group_list.append(match_identifier)
    else:
        if is_currently_selected:
            st.session_state.selected_matches_by_group[active_group_index] = [
                item for item in current_group_list
                if not (item['match_id'] == id_tuple[0] and item['season_id'] == id_tuple[1])
            ]

def handle_bulk_action_change(selectbox_key, exp_key, season_matches, active_group_index):
    selected_action = st.session_state[selectbox_key]
    show_last_n_key = f"show_last_n_input_{exp_key}"
    if show_last_n_key in st.session_state:
         st.session_state[show_last_n_key] = False
    if selected_action == "Select All Matches":
        bulk_select('select', season_matches, active_group_index)
    elif selected_action == "Deselect All Matches":
        bulk_select('deselect', season_matches, active_group_index)
    elif selected_action == "Select Last N Matches":
        st.session_state[show_last_n_key] = True
        return
    st.session_state[selectbox_key] = "--Select Action--"
    # No st.rerun() here to allow "Select Last N" to show input

def handle_select_last_n(exp_key, season_matches, active_group_index):
    num_last_matches_key = f"num_last_{exp_key}"
    show_last_n_key = f"show_last_n_input_{exp_key}"
    selectbox_key = f"bulk_action_select_{exp_key}"
    if num_last_matches_key not in st.session_state:
         st.session_state[num_last_matches_key] = 5
    n_to_select = st.session_state[num_last_matches_key]
    bulk_select('select_last_n', season_matches, active_group_index, n=n_to_select)
    st.session_state[show_last_n_key] = False
    st.session_state[selectbox_key] = "--Select Action--"
    # st.rerun() # Rerun if state change needs immediate reflection of counts

def reset_dynamic_filters_to_all(df, config, id_lookup):
    """Initializes filters to their 'ALL' or full range state."""
    filters = {}
    if df is None or df.empty:
        # Initialize with ALL even if df is empty, widgets need a default
        for item in config:
             col_name = item["name"]
             filter_type = item["type"]
             if filter_type in ["multiselect", "multiselect_search"]:
                 filters[col_name] = [ALL_OPTION_LABEL]
             elif filter_type == "slider":
                 # Provide generic defaults if no data
                 filters[col_name] = (0.0, 100.0) if col_name in ['x','y','end_x','end_y'] else (0, 90 if col_name == 'min' else 0)
             elif filter_type == "slider_or_progressive":
                 filters[col_name] = {"type": "range", "value": (0.0, 100.0)}
             else:
                 filters[col_name] = None
        return filters

    for item in config:
        col_name = item["name"]
        filter_type = item["type"]
        source_col = item.get("source_col", col_name)

        if col_name == "match_display_name":
            filters[col_name] = [ALL_OPTION_LABEL]
            continue

        if source_col in df.columns:
            if filter_type in ["multiselect", "multiselect_search"]:
                 filters[col_name] = [ALL_OPTION_LABEL]
            elif filter_type == "slider":
                min_val, max_val = df[source_col].min(), df[source_col].max()
                default_min, default_max = (0.0, 100.0) if source_col in ['x','y','end_x','end_y'] else (0, 90 if source_col == 'min' else 0)
                filters[col_name] = (float(min_val) if pd.notna(min_val) else default_min,
                                     float(max_val) if pd.notna(max_val) else default_max)
            elif filter_type == "slider_or_progressive":
                 filters[col_name] = {"type": "range", "value": (0.0, 100.0)} # Default end_x to range
                 # Initialize progressive amount state if not present - store separately or within the dict? Let's store in dict
                 filters[f"{col_name}_progressive_amount"] = 5 # Store amount separately for number_input state persistence
        else:
             filters[col_name] = [ALL_OPTION_LABEL] if filter_type in ["multiselect", "multiselect_search"] else None
    return filters

def display_sidebar_group_summary(group_index, match_lookup_dict):
    st.sidebar.markdown("**Selected Matches in this Group:**")
    group_selections = st.session_state.selected_matches_by_group[group_index]
    if group_selections:
        MAX_SUMMARY_ITEMS = 10
        display_count = 0
        for item in reversed(group_selections): # Show most recent selections first
            if display_count >= MAX_SUMMARY_ITEMS:
                st.sidebar.caption(f"...and {len(group_selections) - MAX_SUMMARY_ITEMS} more")
                break
            match_key_tuple = (item['match_id'], item['season_id'])
            match_data = match_lookup_dict.get(match_key_tuple) # Use the passed lookup
            if match_data:
                home_team = match_data['teams']['home']['team_info']['name']
                away_team = match_data['teams']['away']['team_info']['name']
                home_score = match_data['teams']['home']['score']
                away_score = match_data['teams']['away']['score']
                date_str = match_data['match_info'].get('date')
                date_display = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%y") if date_str else "N/A"
                st.sidebar.caption(f"{date_display} | {home_team} {home_score}-{away_score} {away_team}")
            else:
                st.sidebar.caption(f"M:{item['match_id']} (S:{item['season_id']}) - Details missing")
            display_count += 1
    else:
        st.sidebar.info("No matches selected in this group yet.")

# Add new function to load CSV files
@st.cache_data
def load_csv_files_for_selected_matches(selected_match_codes, base_folder):
    all_csv_dfs = []
    if not selected_match_codes:
        return {}
    
    # Collect all CSV files and their data
    for match_code in selected_match_codes:
        match_folder_path = os.path.join(base_folder, match_code)
        if not os.path.exists(match_folder_path):
            st.warning(f"Match folder not found for {match_code} at {match_folder_path}")
            continue
            
        # Find all CSV files in the match directory
        csv_files = glob.glob(os.path.join(match_folder_path, "*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['match_code_source'] = match_code  # Add match code for reference
                all_csv_dfs.append(df)
            except Exception as e:
                st.error(f"Error loading CSV file {csv_file}: {e}")
                continue
    
    # Combine all CSVs into a single DataFrame
    if all_csv_dfs:
        try:
            combined_df = pd.concat(all_csv_dfs, ignore_index=True)
            return {"combined_player_data.csv": combined_df}
        except Exception as e:
            st.error(f"Error combining CSV files: {e}")
            return {}
    return {}

# Modify the existing load_events_for_selected_matches function
@st.cache_data
def load_events_for_selected_matches(selected_match_codes, base_folder, columns_to_load=None):
    all_dfs = []
    if not selected_match_codes:
        return pd.DataFrame(), {}  # Return empty DataFrame and empty dict for CSV files
    
    # Load parquet files
    for match_code in selected_match_codes:
        events_filepath = os.path.join(base_folder, match_code, f"{match_code}_events.parquet")
        try:
            if not os.path.exists(events_filepath):
                st.warning(f"Events file not found for {match_code} at {events_filepath}")
                continue
            df = pd.read_parquet(events_filepath, columns=columns_to_load)
            if '_id' not in df.columns and 'guid' in df.columns:
                df = df.rename(columns={'guid': '_id'})
            elif '_id' not in df.columns and 'guid' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'event_index_placeholder'})
            df['match_code_source'] = match_code
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading events for {match_code}: {e}")
            continue
    
    # Load CSV files
    csv_dfs = load_csv_files_for_selected_matches(selected_match_codes, base_folder)
    
    # Combine parquet data
    try:
        combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        # Convert relevant columns to numeric
        for col in ['half', 'min', 'x', 'y', 'end_x', 'end_y']:
             if col in combined_df.columns:
                 combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    except Exception as e:
        st.error(f"Error concatenating event DataFrames: {e}")
        combined_df = pd.DataFrame()

    return combined_df, csv_dfs

# REMOVED cache - caching this prevents cascading updates
def get_options_for_filter(df, column_name):
    """Gets unique, sorted, non-NaN values from a potentially filtered DataFrame."""
    if df is None or df.empty or column_name not in df.columns:
        return []
    try:
        unique_list = df[column_name].dropna().unique().tolist()
        try:
            if column_name == 'half':
                # Filter out non-integer values before conversion and sorting for 'half'
                # Also handle potential floats if needed
                valid_halves = [v for v in unique_list if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())]
                return sorted([int(float(h)) for h in valid_halves])  # Convert via float for robustness
            else:
                return sorted(unique_list)
        except TypeError:
            return sorted(map(str, unique_list))
    except Exception:  # Catch any other potential error during unique/sort
        return []

# Add this function after the imports
def apply_predefined_filters(df, filter_group):
    """Apply predefined filter conditions to the dataframe."""
    if filter_group == "Custom":
        return df
    
    filter_conditions = METRIC_FORMULAS.get(filter_group, {})
    filtered_df = df.copy()
    
    for col, value in filter_conditions.items():
        if isinstance(value, list):
            filtered_df = filtered_df[filtered_df[col].isin(value)]
        elif isinstance(value, dict):
            if value["type"] == "range":
                min_val, max_val = value["value"]
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
            elif value["type"] == "progressive":
                filtered_df = filtered_df[filtered_df[col] > filtered_df["x"] + value["value"]]
    
    return filtered_df

# Add these helper functions after the imports
def color_picker(label, default_color, key=None):
    return st.color_picker(label, default_color, key=key)

def download_figure(fig, filename=None):
    # Generate a default filename if none provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure_{timestamp}.png"
    
    # Convert the figure to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf, filename

# --- Load Master Seasons Data (for match selection UI) ---
seasons_data = load_combined_seasons(DATA_PATH)

# --- Prepare Data for Match Selection UI ---
all_tournaments = sorted(list(set(s['tournament_name'] for s in seasons_data['seasons'])), reverse=True)
all_seasons = sorted(list(set(s['season_name'] for s in seasons_data['seasons'])), reverse=True)
all_matches_for_selection_ui = []
all_teams = set()
all_players = set()
all_stadiums = set()
all_referees = set()
match_id_to_details_lookup = {} # NEW: Lookup by match_id only

for season in seasons_data['seasons']:
    for match in season['matches']:
        match_data_copy = match.copy()
        match_data_copy['season_id'] = season['season_id']
        match_data_copy['season_name'] = season['season_name']
        match_data_copy['tournament_name'] = season['tournament_name']
        all_matches_for_selection_ui.append(match_data_copy)

        match_id = match_data_copy.get('match_info', {}).get('match_id')
        if match_id:
             try:
                 # Ensure match_id is consistently integer for the lookup key
                 match_id_int_key = int(match_id)
                 home_t = match_data_copy.get('teams',{}).get('home',{}).get('team_info',{}).get('name', 'N/A')
                 away_t = match_data_copy.get('teams',{}).get('away',{}).get('team_info',{}).get('name', 'N/A')
                 date_str = match_data_copy.get('match_info', {}).get('date', '')
                 date_display = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d %b %Y") if date_str else ""
                 display_name = f"{home_t} vs {away_t} ({date_display})"
                 match_id_to_details_lookup[match_id_int_key] = {"display_name": display_name, "data": match_data_copy}
             except (ValueError, TypeError):
                  st.warning(f"Could not process match_id {match_id} for lookup.")


        home_team = match_data_copy['teams']['home']['team_info']['name']
        away_team = match_data_copy['teams']['away']['team_info']['name']
        all_teams.update([home_team, away_team])
        for p_list_type in ['starting', 'subs']:
            for p in match_data_copy['teams']['home']['players'].get(p_list_type, []): all_players.add(p['player_name'])
            for p in match_data_copy['teams']['away']['players'].get(p_list_type, []): all_players.add(p['player_name'])
        stadium = match_data_copy['match_info'].get('stadium'); referee = match_data_copy['match_info'].get('referee')
        if stadium and isinstance(stadium, str): all_stadiums.add(stadium)
        if referee and isinstance(referee, str): all_referees.add(referee)

all_teams = sorted(list(all_teams)); all_players = sorted(list(all_players)); all_stadiums = sorted(list(all_stadiums)); all_referees = sorted(list(all_referees))
match_lookup_for_selection_ui = {(m['match_info']['match_id'], m['season_id']): m for m in all_matches_for_selection_ui}

# --- Initialize Session State ---
if "active_group_index" not in st.session_state: st.session_state.active_group_index = 0
if "group_names" not in st.session_state: st.session_state.group_names = [f"Group {i+1}" for i in range(3)]
if "group_filters" not in st.session_state:
    st.session_state.group_filters = [{"tournaments": [], "seasons": [], "teams": [], "players": [], "stadiums": [], "referees": [], "result": "All", "player_status": "All"} for _ in range(3)]
if "selected_matches_by_group" not in st.session_state: st.session_state.selected_matches_by_group = [[] for _ in range(3)]
if "show_event_data" not in st.session_state: st.session_state.show_event_data = False

# Initialize group rename keys
for i in range(3):
    if f"rename_group_{i}" not in st.session_state:
        st.session_state[f"rename_group_{i}"] = st.session_state.group_names[i]

if "merged_events_by_group" not in st.session_state:
    st.session_state.merged_events_by_group = [pd.DataFrame() for _ in range(3)]
if "dynamic_event_filters_by_group" not in st.session_state:
    st.session_state.dynamic_event_filters_by_group = [{} for _ in range(3)]

# Initialize CSV data in session state
if "csv_data_by_group" not in st.session_state:
    st.session_state.csv_data_by_group = [{} for _ in range(3)]

# --- Sidebar: Match Selection Management ---
st.sidebar.title("Match Selection")
st.sidebar.subheader("1. Active Group")
active_group_index = st.sidebar.radio("Active Group:", [0, 1, 2], format_func=lambda x: st.session_state.group_names[x], key="active_group_selector_sidebar")
st.session_state.active_group_index = active_group_index
st.sidebar.text_input("Rename Selected Group:", value=st.session_state.group_names[active_group_index], key=f"rename_group_sidebar_{active_group_index}", on_change=update_group_name)
display_sidebar_group_summary(st.session_state.active_group_index, match_lookup_for_selection_ui)  # Pass the correct lookup
st.sidebar.markdown("---")

st.sidebar.subheader("2. Load Events")
total_selected_count = sum(len(st.session_state.selected_matches_by_group[i]) for i in range(len(st.session_state.selected_matches_by_group)))
st.sidebar.caption(f"{total_selected_count} matches selected across all groups.")

# Load events button for each group
for group_idx in range(3):
    group_matches = st.session_state.selected_matches_by_group[group_idx]
    if group_matches:
        if st.sidebar.button(f"🚀 Load Events for {st.session_state.group_names[group_idx]}", key=f"load_events_button_group_{group_idx}"):
            unique_match_ids_int = set()
            seen_id_season_tuples = set()
            for item in group_matches:
                id_tuple = (item['match_id'], item['season_id'])
                if id_tuple not in seen_id_season_tuples:
                    try:
                        match_id_int_key = int(item['match_id'])
                        unique_match_ids_int.add(match_id_int_key)
                        seen_id_season_tuples.add(id_tuple)
                    except (ValueError, TypeError):
                        st.warning(f"Skipping invalid match_id: {item['match_id']}")

            if not unique_match_ids_int:
                st.sidebar.error(f"No unique, valid matches selected in {st.session_state.group_names[group_idx]}.")
                st.session_state.merged_events_by_group[group_idx] = pd.DataFrame()
                st.session_state.dynamic_event_filters_by_group[group_idx] = {}
                st.session_state.csv_data_by_group[group_idx] = {}
            else:
                selected_match_codes_to_load = sorted([f"match_{mid}" for mid in unique_match_ids_int])
                with st.spinner(f"Loading events and CSV files for {len(selected_match_codes_to_load)} matches in {st.session_state.group_names[group_idx]}..."):
                    merged_df_result, csv_data = load_events_for_selected_matches(selected_match_codes_to_load, ANALYSIS_DATA_DIR, columns_to_load=INITIAL_COLUMNS_TO_LOAD_FROM_PARQUET)
                
                st.session_state.merged_events_by_group[group_idx] = merged_df_result
                st.session_state.dynamic_event_filters_by_group[group_idx] = reset_dynamic_filters_to_all(merged_df_result, COLUMNS_FOR_DYNAMIC_FILTER_CONFIG, match_id_to_details_lookup)
                st.session_state.csv_data_by_group[group_idx] = csv_data

                if not merged_df_result.empty:
                    st.sidebar.success(f"Loaded {len(merged_df_result)} events for {st.session_state.group_names[group_idx]}.")
                else:
                    st.sidebar.warning(f"No event data loaded for selected matches in {st.session_state.group_names[group_idx]}.")
                    st.session_state.dynamic_event_filters_by_group[group_idx] = {}
                
                # Display CSV file information
                if csv_data:
                    st.sidebar.success(f"Loaded {len(csv_data)} CSV files:")
                    for file_name, df in csv_data.items():
                        st.sidebar.info(f"- {file_name}: {len(df)} rows")

# Add Proceed button if any group has loaded events
if any(not df.empty for df in st.session_state.merged_events_by_group):
    if st.sidebar.button("➡️ Proceed to Event Data", key="proceed_to_events"):
        st.session_state.show_event_data = True

st.sidebar.markdown("---")

st.sidebar.subheader("3. Filters for Match List Below")
current_match_list_filters = st.session_state.group_filters[active_group_index]
selected_tournaments_matchlist = st.sidebar.multiselect("Competition (Match List):", all_tournaments, default=current_match_list_filters['tournaments'], key=f"filter_tournaments_matchlist_{active_group_index}", on_change=update_filter_state, args=('tournaments', f"filter_tournaments_matchlist_{active_group_index}"))
selected_seasons_matchlist = st.sidebar.multiselect("Season (Match List):", all_seasons, default=current_match_list_filters['seasons'], key=f"filter_seasons_matchlist_{active_group_index}", on_change=update_filter_state, args=('seasons', f"filter_seasons_matchlist_{active_group_index}"))
selected_teams_matchlist = st.sidebar.multiselect("Teams (Match List):", all_teams, default=current_match_list_filters['teams'], key=f"filter_teams_matchlist_{active_group_index}", on_change=update_filter_state, args=('teams', f"filter_teams_matchlist_{active_group_index}"))
selected_players_matchlist = st.sidebar.multiselect("Players (Match List):", all_players, default=current_match_list_filters['players'], key=f"filter_players_matchlist_{active_group_index}", on_change=update_filter_state, args=('players', f"filter_players_matchlist_{active_group_index}"))
selected_stadiums_matchlist = st.sidebar.multiselect("Stadiums (Match List):", all_stadiums, default=current_match_list_filters['stadiums'], key=f"filter_stadiums_matchlist_{active_group_index}", on_change=update_filter_state, args=('stadiums', f"filter_stadiums_matchlist_{active_group_index}"))
selected_referees_matchlist = st.sidebar.multiselect("Referees (Match List):", all_referees, default=current_match_list_filters['referees'], key=f"filter_referees_matchlist_{active_group_index}", on_change=update_filter_state, args=('referees', f"filter_referees_matchlist_{active_group_index}"))
selected_result_matchlist = st.sidebar.selectbox("Result (Match List):", ["All", "Win", "Draw", "Loss"], index=["All", "Win", "Draw", "Loss"].index(current_match_list_filters['result']), key=f"filter_result_matchlist_{active_group_index}", on_change=update_filter_state, args=('result', f"filter_result_matchlist_{active_group_index}"))
selected_player_status_matchlist = st.sidebar.selectbox("Player Status (Match List):", ["All", "Starter", "Sub"], index=["All", "Starter", "Sub"].index(current_match_list_filters['player_status']), key=f"filter_player_status_matchlist_{active_group_index}", on_change=update_filter_state, args=('player_status', f"filter_player_status_matchlist_{active_group_index}"))

# --- Main Page ---
st.title("⚽ Football Match Event Explorer")

# Initialize the DataFrame that will be filtered for display
final_filtered_df_display = pd.DataFrame()
if not st.session_state.merged_events_by_group[st.session_state.active_group_index].empty:
    final_filtered_df_display = st.session_state.merged_events_by_group[st.session_state.active_group_index].copy()

# --- Displaying the Final Filtered DataFrame ---
if st.session_state.show_event_data:
    # Add filter group selector at the top
    filter_groups = ["Custom"] + list(METRIC_FORMULAS.keys())
    selected_filter_group = st.selectbox(
        "Select Filter Group",
        options=filter_groups,
        key="filter_group_selector"
    )
    
    # This df is progressively filtered *within this block* to generate options
    df_options_builder = st.session_state.merged_events_by_group[st.session_state.active_group_index].copy()
    
    # Apply predefined filters if selected
    if selected_filter_group != "Custom":
        df_options_builder = apply_predefined_filters(df_options_builder, selected_filter_group)
        st.info(f"Showing {len(df_options_builder)} events matching {selected_filter_group} criteria")
    
    # Store selections made *in this render pass* to update session state later
    current_pass_selections = {}
    # Store the mapping for the current run
    match_name_to_code_map_current_run = {}

    # Show custom filters regardless of selected filter group
    st.markdown("### Event Detail Filters")
    filter_cols_group1 = st.columns(4)
    col_idx_g1 = 0
    for filter_config in COLUMNS_FOR_DYNAMIC_FILTER_CONFIG:
        if filter_config["group"] == 1:
            col_name = filter_config["name"]
            filter_type = filter_config["type"]
            label = filter_config["label"]
            widget_key = f"dynamic_event_filter_widget_{col_name}"
            source_col = filter_config.get("source_col", col_name)

            with filter_cols_group1[col_idx_g1 % len(filter_cols_group1)]:
                # Check if column exists in the progressively filtered df
                if source_col not in df_options_builder.columns:
                    current_pass_selections[col_name] = st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index].get(col_name)
                    col_idx_g1 += 1
                    continue

                # Get options from the *currently* filtered df_options_builder
                options = get_options_for_filter(df_options_builder, source_col)
                current_selection_from_state = st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index].get(col_name)
                user_selection = None

                if filter_type in ["multiselect", "multiselect_search"]:
                    options_with_all = [ALL_OPTION_LABEL] + options
                    default_val = current_selection_from_state if current_selection_from_state is not None else [ALL_OPTION_LABEL]
                    if not isinstance(default_val, list) or not all(d in options_with_all for d in default_val):
                        default_val = [ALL_OPTION_LABEL]
                    user_selection = st.multiselect(label, options_with_all, default=default_val, key=widget_key)

                # Store this widget's selection for later update to session state
                current_pass_selections[col_name] = user_selection

                # --- Apply this filter *immediately* to df_options_builder ---
                if user_selection is not None:
                    if filter_type in ["multiselect", "multiselect_search"] and ALL_OPTION_LABEL not in user_selection and user_selection:
                        df_options_builder = df_options_builder[df_options_builder[source_col].isin(user_selection)]
                    elif filter_type in ["multiselect", "multiselect_search"] and not user_selection:
                        df_options_builder = df_options_builder.iloc[0:0]
                    if df_options_builder.empty:
                        break
            col_idx_g1 += 1

    st.markdown("### Context Filters (Match, Player, Time, Location)")
    filter_cols_group2 = st.columns(4)
    col_idx_g2 = 0
    for filter_config in COLUMNS_FOR_DYNAMIC_FILTER_CONFIG:
        if filter_config["group"] == 2:
            col_name = filter_config["name"]
            filter_type = filter_config["type"]
            label = filter_config["label"]
            widget_key = f"dynamic_event_filter_widget_{col_name}"
            source_col = filter_config.get("source_col", col_name)

            with filter_cols_group2[col_idx_g2 % len(filter_cols_group2)]:
                # Check column existence in the potentially filtered df_options_builder
                if source_col not in df_options_builder.columns and col_name != "match_display_name":
                    current_pass_selections[col_name] = st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index].get(col_name)
                    col_idx_g2 += 1
                    continue

                # Get options based on df_options_builder
                options = []
                current_selection_from_state = st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index].get(col_name)
                user_selection = None

                if filter_type in ["multiselect", "multiselect_search"]:
                    if col_name == "match_display_name":
                        unique_codes = get_options_for_filter(df_options_builder, source_col)
                        temp_name_to_code = {}
                        display_names = []
                        for code in unique_codes:
                            match_id_search = re.search(r'\d+$', code)
                            if match_id_search:
                                match_id = int(match_id_search.group())
                                details = match_id_to_details_lookup.get(match_id)
                                if details:
                                    display_name = details["display_name"]
                                    display_names.append(display_name)
                                    temp_name_to_code[display_name] = code
                        options = sorted(list(set(display_names)))
                        match_name_to_code_map_current_run.update(temp_name_to_code)
                    elif col_name == 'half':
                        current_halves = get_options_for_filter(df_options_builder, source_col)
                        options = sorted(list(set(current_halves + [1, 2, 3, 4])))
                    else:
                        options = get_options_for_filter(df_options_builder, source_col)

                    options_with_all = [ALL_OPTION_LABEL] + options
                    default_val = current_selection_from_state if current_selection_from_state is not None else [ALL_OPTION_LABEL]
                    if not isinstance(default_val, list) or not all(d in options_with_all for d in default_val):
                        default_val = [ALL_OPTION_LABEL]
                    user_selection = st.multiselect(label, options_with_all, default=default_val, key=widget_key)

                elif filter_type == "slider":
                    overall_min = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].min()
                    overall_max = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].max()
                    min_s_val, max_s_val = (0.0, 100.0) if source_col in ['x','y','end_y'] else (0, 90 if source_col == 'min' else 0)
                    if pd.notna(overall_min):
                        min_s_val = float(overall_min)
                    if pd.notna(overall_max):
                        max_s_val = float(overall_max)
                    if min_s_val >= max_s_val:
                        max_s_val = min_s_val + (1.0 if isinstance(min_s_val, float) else 1)

                    default_s_val_tuple = current_selection_from_state if isinstance(current_selection_from_state, tuple) else (min_s_val, max_s_val)
                    clamped_s_default = (max(min_s_val, default_s_val_tuple[0]), min(max_s_val, default_s_val_tuple[1]))
                    user_selection = st.slider(label, min_s_val, max_s_val, value=clamped_s_default, key=widget_key)

                elif filter_type == "slider_or_progressive" and source_col == "end_x":
                    end_x_state = current_selection_from_state if isinstance(current_selection_from_state, dict) else {"type": "range", "value": (0.0, 100.0)}
                    mode_options = ("Range", "Progressive")
                    current_mode_idx = 0 if end_x_state.get("type", "range") == "range" else 1
                    selected_mode = st.radio(f"{label} Mode:", mode_options, index=current_mode_idx, key=f"{widget_key}_mode", horizontal=True)
                    new_widget_state = {"type": "range" if selected_mode == "Range" else "progressive"}
                    if new_widget_state["type"] == "range":
                        overall_min_endx = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].min()
                        overall_max_endx = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].max()
                        min_slider_endx, max_slider_endx = (0.0, 100.0)
                        if pd.notna(overall_min_endx):
                            min_slider_endx = float(overall_min_endx)
                        if pd.notna(overall_max_endx):
                            max_slider_endx = float(overall_max_endx)
                        if min_slider_endx >= max_slider_endx:
                            max_slider_endx = min_slider_endx + 1.0
                        default_range_tuple_endx = (min_slider_endx, max_slider_endx)
                        if end_x_state.get("type") == "range" and isinstance(end_x_state.get("value"), tuple):
                            default_range_tuple_endx = end_x_state["value"]
                        clamped_default_endx_range = (max(min_slider_endx, default_range_tuple_endx[0]), min(max_slider_endx, default_range_tuple_endx[1]))
                        selected_range_val = st.slider(f"{label} Range:", min_slider_endx, max_slider_endx, value=clamped_default_endx_range, key=f"{widget_key}_range_val")
                        new_widget_state["value"] = selected_range_val
                    else:  # Progressive
                        default_prog_val = 5
                        if end_x_state.get("type") == "progressive" and isinstance(end_x_state.get("value"), (int, float)):
                            default_prog_val = end_x_state["value"]
                        prog_input_val = st.number_input(f"{label} by at least:", value=default_prog_val, min_value=1, step=1, key=f"{widget_key}_prog_val_input")
                        new_widget_state["value"] = prog_input_val
                    user_selection = new_widget_state

                # Store selection and filter df_options_builder
                current_pass_selections[col_name] = user_selection
                if user_selection is not None:
                    # --- Apply immediate filter logic for cascading ---
                    filter_applied_cascade = False
                    if filter_type in ["multiselect", "multiselect_search"]:
                        selected_codes_for_cascade = []
                        is_active_multiselect = False
                        if col_name == "match_display_name":
                            if ALL_OPTION_LABEL not in user_selection and user_selection:
                                selected_codes_for_cascade = [match_name_to_code_map_current_run[name] for name in user_selection if name in match_name_to_code_map_current_run]
                                is_active_multiselect = True
                        else:  # Standard multiselect
                            if ALL_OPTION_LABEL not in user_selection and user_selection:
                                selected_codes_for_cascade = user_selection
                                is_active_multiselect = True

                        if is_active_multiselect:
                            if selected_codes_for_cascade:
                                df_options_builder = df_options_builder[df_options_builder[source_col].isin(selected_codes_for_cascade)]
                                filter_applied_cascade = True
                            elif user_selection:
                                df_options_builder = df_options_builder.iloc[0:0]
                                filter_applied_cascade = True

                    elif filter_type == "slider":
                        min_ov_casc = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].min()
                        max_ov_casc = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].max()
                        if pd.notna(min_ov_casc) and pd.notna(max_ov_casc):
                            if user_selection[0] > float(min_ov_casc) or user_selection[1] < float(max_ov_casc):
                                df_options_builder = df_options_builder[(df_options_builder[source_col].notna()) & 
                                                                      (df_options_builder[source_col] >= user_selection[0]) & 
                                                                      (df_options_builder[source_col] <= user_selection[1])]
                                filter_applied_cascade = True
                    elif filter_type == "slider_or_progressive" and source_col == "end_x":
                        mode = user_selection.get("type")
                        value = user_selection.get("value")
                        if mode == "range":
                            min_ov_endx_casc = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].min()
                            max_ov_endx_casc = st.session_state.merged_events_by_group[st.session_state.active_group_index][source_col].max()
                            min_check_casc, max_check_casc = (0.0, 100.0)
                            if pd.notna(min_ov_endx_casc):
                                min_check_casc = float(min_ov_endx_casc)
                            if pd.notna(max_ov_endx_casc):
                                max_check_casc = float(max_ov_endx_casc)
                            if value[0] > min_check_casc or value[1] < max_check_casc:
                                df_options_builder = df_options_builder[(df_options_builder[source_col].notna()) & 
                                                                      (df_options_builder[source_col] >= value[0]) & 
                                                                      (df_options_builder[source_col] <= value[1])]
                                filter_applied_cascade = True
                        elif mode == "progressive":
                            if 'x' in df_options_builder.columns:
                                df_options_builder = df_options_builder[(df_options_builder['x'].notna()) & 
                                                                      (df_options_builder[source_col].notna()) & 
                                                                      (df_options_builder[source_col] > df_options_builder['x'] + value)]
                                filter_applied_cascade = True

                    if df_options_builder.empty:
                        break
                col_idx_g2 += 1

    # --- Update session state AFTER rendering all widgets ---
    for col, val in current_pass_selections.items():
        st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index][col] = val

    st.markdown("---")
    st.header("Data by Group")

    # Create tabs for each group
    tabs = st.tabs([st.session_state.group_names[i] for i in range(3)])

    # Display data for each group in its respective tab
    for group_idx, tab in enumerate(tabs):
        with tab:
            if not st.session_state.merged_events_by_group[group_idx].empty or st.session_state.csv_data_by_group[group_idx]:
                # Create inner tabs for Event Data and Player Data
                data_tabs = st.tabs(["Event Data", "Player Data"])
                
                # Event Data Tab
                with data_tabs[0]:
                    if not st.session_state.merged_events_by_group[group_idx].empty:
                        st.markdown("### Event Data")
                        # Apply filters based on the FINAL state in dynamic_event_filters_by_group
                        final_filtered_df_display = st.session_state.merged_events_by_group[group_idx].copy()
                        
                        # Apply predefined filters if selected
                        if selected_filter_group != "Custom":
                            final_filtered_df_display = apply_predefined_filters(final_filtered_df_display, selected_filter_group)
                            st.info(f"Showing {len(final_filtered_df_display)} events matching {selected_filter_group} criteria")
                        
                        # Apply custom filters
                        active_filters_applied_count_display = 0

                        # Rebuild match name map based on potentially filtered final data
                        display_name_to_code_final = {}
                        if "match_display_name" in st.session_state.dynamic_event_filters_by_group[group_idx]:
                            codes_in_final_df = final_filtered_df_display['match_code_source'].unique()
                            for code in codes_in_final_df:
                                match_id_search = re.search(r'\d+$', code)
                                if match_id_search:
                                    match_id = int(match_id_search.group())
                                    details = match_id_to_details_lookup.get(match_id)
                                    if details:
                                        display_name_to_code_final[details["display_name"]] = code

                        selected_match_names_final = st.session_state.dynamic_event_filters_by_group[group_idx].get("match_display_name", [ALL_OPTION_LABEL])
                        selected_match_codes_final_apply = []
                        if ALL_OPTION_LABEL not in selected_match_names_final and selected_match_names_final:
                            selected_match_codes_final_apply = [display_name_to_code_final[name] for name in selected_match_names_final if name in display_name_to_code_final]

                        for filter_config_display in COLUMNS_FOR_DYNAMIC_FILTER_CONFIG:
                            col_display = filter_config_display["name"]
                            source_col_display = filter_config_display.get("source_col", col_display)
                            current_selection_for_display = st.session_state.dynamic_event_filters_by_group[group_idx].get(col_display)

                            if current_selection_for_display is None:
                                continue
                            if source_col_display not in final_filtered_df_display.columns and col_display != "match_display_name":
                                continue

                            is_filter_active_final = False
                            # --- Apply Match Name Filter ---
                            if col_display == "match_display_name":
                                if selected_match_codes_final_apply:
                                    final_filtered_df_display = final_filtered_df_display[final_filtered_df_display['match_code_source'].isin(selected_match_codes_final_apply)]
                                    is_filter_active_final = True
                                elif ALL_OPTION_LABEL not in selected_match_names_final and selected_match_names_final:
                                    final_filtered_df_display = final_filtered_df_display.iloc[0:0]
                                    is_filter_active_final = True
                            # --- Apply Other Filters ---
                            elif filter_config_display["type"] in ["multiselect", "multiselect_search"]:
                                if ALL_OPTION_LABEL not in current_selection_for_display and current_selection_for_display:
                                    final_filtered_df_display = final_filtered_df_display[final_filtered_df_display[source_col_display].isin(current_selection_for_display)]
                                    is_filter_active_final = True
                                elif not current_selection_for_display:
                                    final_filtered_df_display = final_filtered_df_display.iloc[0:0]
                                    is_filter_active_final = True
                            elif filter_config_display["type"] == "slider":
                                min_overall_disp = st.session_state.merged_events_by_group[group_idx][source_col_display].min()
                                max_overall_disp = st.session_state.merged_events_by_group[group_idx][source_col_display].max()
                                if pd.notna(min_overall_disp) and pd.notna(max_overall_disp):
                                    if current_selection_for_display[0] > float(min_overall_disp) or current_selection_for_display[1] < float(max_overall_disp):
                                        final_filtered_df_display = final_filtered_df_display[(final_filtered_df_display[source_col_display].notna()) & \
                                                                                            (final_filtered_df_display[source_col_display] >= current_selection_for_display[0]) & \
                                                                                            (final_filtered_df_display[source_col_display] <= current_selection_for_display[1])]
                                        is_filter_active_final = True
                            elif filter_config_display["type"] == "slider_or_progressive" and source_col_display == "end_x":
                                mode_disp = current_selection_for_display.get("type")
                                value_disp = current_selection_for_display.get("value")
                                if mode_disp == "range":
                                    min_overall_endx_disp = st.session_state.merged_events_by_group[group_idx][source_col_display].min()
                                    max_overall_endx_disp = st.session_state.merged_events_by_group[group_idx][source_col_display].max()
                                    full_range_min_disp, full_range_max_disp = (0.0, 100.0)
                                    if pd.notna(min_overall_endx_disp):
                                        full_range_min_disp = float(min_overall_endx_disp)
                                    if pd.notna(max_overall_endx_disp):
                                        full_range_max_disp = float(max_overall_endx_disp)
                                    if value_disp[0] > full_range_min_disp or value_disp[1] < full_range_max_disp:
                                        final_filtered_df_display = final_filtered_df_display[(final_filtered_df_display[source_col_display].notna()) & \
                                                                                            (final_filtered_df_display[source_col_display] >= value_disp[0]) & \
                                                                                            (final_filtered_df_display[source_col_display] <= value_disp[1])]
                                        is_filter_active_final = True
                                elif mode_disp == "progressive":
                                    if 'x' in final_filtered_df_display.columns:
                                        final_filtered_df_display = final_filtered_df_display[(final_filtered_df_display['x'].notna()) & \
                                                                                            (final_filtered_df_display[source_col_display].notna()) & \
                                                                                            (final_filtered_df_display[source_col_display] > final_filtered_df_display['x'] + value_disp)]
                                        is_filter_active_final = True

                            if is_filter_active_final:
                                active_filters_applied_count_display += 1
                            if final_filtered_df_display.empty:
                                break

                        if active_filters_applied_count_display > 0:
                            st.write(f"Showing {len(final_filtered_df_display)} events after applying dynamic filters (out of {len(st.session_state.merged_events_by_group[group_idx])} merged events).")
                        else:
                            st.write(f"Showing all {len(final_filtered_df_display)} merged events (no specific dynamic filters active, or \"{ALL_OPTION_LABEL}\" selected for all applicable filters).")

                        # Add save dataset functionality
                        st.markdown("### Save Filtered Dataset")
                        save_cols = st.columns([3, 1])
                        with save_cols[0]:
                            dataset_name = st.text_input(
                                "Dataset Name",
                                key=f"dataset_name_{group_idx}",
                                placeholder="Enter a name for this filtered dataset"
                            )
                        with save_cols[1]:
                            if st.button("💾 Save Dataset", key=f"save_dataset_{group_idx}"):
                                if dataset_name:
                                    # Create a unique key for this dataset
                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    dataset_key = f"{dataset_name}_{timestamp}"
                                    # Store the dataset and its metadata
                                    st.session_state.saved_datasets[dataset_key] = {
                                        "name": dataset_name,
                                        "data": final_filtered_df_display,
                                        "group": group_idx,
                                        "filter_group": selected_filter_group,
                                        "filters": st.session_state.dynamic_event_filters_by_group[group_idx].copy(),
                                        "timestamp": timestamp,
                                        "row_count": len(final_filtered_df_display)
                                    }
                                    st.success(f"Dataset '{dataset_name}' saved successfully!")
                                else:
                                    st.error("Please enter a name for the dataset")

                        # Display the filtered data
                        if 'match_code_source' in final_filtered_df_display.columns and 'timeInSec' in final_filtered_df_display.columns:
                            display_df_to_show = final_filtered_df_display.sort_values(by=['match_code_source', 'timeInSec'])
                        elif 'timeInSec' in final_filtered_df_display.columns:
                            display_df_to_show = final_filtered_df_display.sort_values(by=['timeInSec'])
                        else:
                            display_df_to_show = final_filtered_df_display

                        st.dataframe(display_df_to_show)
                        csv = display_df_to_show.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download displayed events as CSV", 
                                         data=csv,
                                         file_name=f"filtered_events_group_{group_idx + 1}.csv", 
                                         mime='text/csv')

                        # Add visualization tabs for event data
                        st.markdown("### Event Data Visualizations")
                        viz_tabs = st.tabs([
                            "Scatter Plot",
                            "Pitch Plot"
                        ])
                        
                        with viz_tabs[0]:
                            render_scatter_tab(st, display_df_to_show, color_picker, download_figure, key_prefix=f"{group_idx}_event")
                        
                        with viz_tabs[1]:
                            render_pitch_tab(st, display_df_to_show, color_picker, download_figure, key_prefix=f"{group_idx}_event")
                    else:
                        st.info(f"No event data loaded for {st.session_state.group_names[group_idx]}. Click 'Load Events' for this group in the sidebar.")

                    # Add Saved Event Datasets Repository section
                    st.markdown("---")
                    with st.expander("Saved Event Datasets Repository", expanded=False):
                        if st.session_state.saved_datasets:
                            saved_event_datasets_info = []
                            for key, dataset in st.session_state.saved_datasets.items():
                                saved_event_datasets_info.append({
                                    "Name": dataset["name"],
                                    "Group": st.session_state.group_names[dataset["group"]],
                                    "Filter Group": dataset.get("filter_group", "N/A"),
                                    "Rows": dataset["row_count"],
                                    "Saved At": datetime.datetime.strptime(dataset["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                                    "Key": key
                                })
                            saved_event_datasets_df = pd.DataFrame(saved_event_datasets_info)
                            st.dataframe(saved_event_datasets_df, hide_index=True)
                            st.markdown("### Event Dataset Management")
                            event_dataset_cols = st.columns([3, 1, 1])
                            with event_dataset_cols[0]:
                                selected_event_dataset = st.selectbox(
                                    "Select Event Dataset",
                                    options=list(st.session_state.saved_datasets.keys()),
                                    format_func=lambda x: f"{st.session_state.saved_datasets[x]['name']} ({st.session_state.saved_datasets[x]['row_count']} rows)",
                                    key=f"select_event_dataset_{group_idx}"
                                )
                            with event_dataset_cols[1]:
                                if st.button("📥 Load Event Dataset", key=f"load_saved_event_dataset_{group_idx}"):
                                    if selected_event_dataset:
                                        dataset = st.session_state.saved_datasets[selected_event_dataset]
                                        st.session_state.merged_events_by_group[st.session_state.active_group_index] = dataset["data"]
                                        st.session_state.dynamic_event_filters_by_group[st.session_state.active_group_index] = dataset["filters"]
                                        st.success(f"Event dataset '{dataset['name']}' loaded into current group!")
                                with event_dataset_cols[2]:
                                    if st.button("🗑️ Delete Event Dataset", key=f"delete_saved_event_dataset_{group_idx}"):
                                        if selected_event_dataset:
                                            del st.session_state.saved_datasets[selected_event_dataset]
                                            st.success("Event dataset deleted successfully!")
                                        st.rerun()
                            # Optionally, add visualization tabs for event data here if needed
                        else:
                            st.info("No saved event datasets yet. Use the save functionality above to create your first event dataset.")

                # Player Data Tab
                with data_tabs[1]:
                    if st.session_state.csv_data_by_group[group_idx]:
                        st.markdown("### Player Data")
                        
                        # Initialize filters in session state if not present
                        if "player_filters" not in st.session_state:
                            st.session_state.player_filters = [{} for _ in range(3)]
                        
                        # Get the combined data
                        combined_df = None
                        for file_name, df in st.session_state.csv_data_by_group[group_idx].items():
                            if combined_df is None:
                                combined_df = df
                            else:
                                combined_df = pd.concat([combined_df, df], ignore_index=True)
                        
                        if combined_df is not None:
                            # Add grouping options
                            st.markdown("#### Grouping Options")
                            group_cols = st.columns(3)
                            with group_cols[0]:
                                group_by = st.radio(
                                    "Group by:",
                                    options=[
                                        "Player Only",
                                        "Player + Position",
                                        "Player + Club",
                                        "Player + Position + Club"
                                    ],
                                    key=f"group_by_{group_idx}"
                                )
                            
                            # Add filters if the required columns exist
                            if all(col in combined_df.columns for col in ['nickname', 'primary_position_name', 'club']):
                                filter_cols = st.columns(4)
                                
                                # Nickname filter
                                with filter_cols[0]:
                                    nicknames = sorted(combined_df['nickname'].unique())
                                    current_nicknames = st.session_state.player_filters[group_idx].get('nickname', [])
                                    # Filter out any nicknames that don't exist in current data
                                    valid_nicknames = [n for n in current_nicknames if n in nicknames]
                                    selected_nicknames = st.multiselect(
                                        "Player",
                                        options=nicknames,
                                        default=valid_nicknames,
                                        key=f"nickname_filter_{group_idx}"
                                    )
                                    st.session_state.player_filters[group_idx]['nickname'] = selected_nicknames
                                
                                # Position filter
                                with filter_cols[1]:
                                    positions = sorted(combined_df['primary_position_name'].unique())
                                    current_positions = st.session_state.player_filters[group_idx].get('position', [])
                                    # Filter out any positions that don't exist in current data
                                    valid_positions = [p for p in current_positions if p in positions]
                                    selected_positions = st.multiselect(
                                        "Position",
                                        options=positions,
                                        default=valid_positions,
                                        key=f"position_filter_{group_idx}"
                                    )
                                    st.session_state.player_filters[group_idx]['position'] = selected_positions
                                
                                # Club filter
                                with filter_cols[2]:
                                    clubs = sorted(combined_df['club'].unique())
                                    current_clubs = st.session_state.player_filters[group_idx].get('club', [])
                                    # Filter out any clubs that don't exist in current data
                                    valid_clubs = [c for c in current_clubs if c in clubs]
                                    selected_clubs = st.multiselect(
                                        "Club",
                                        options=clubs,
                                        default=valid_clubs,
                                        key=f"club_filter_{group_idx}"
                                    )
                                    st.session_state.player_filters[group_idx]['club'] = selected_clubs
                                
                                # Playtime filter
                                with filter_cols[3]:
                                    if 'Playtime (mins)' in combined_df.columns:
                                        min_playtime = combined_df['Playtime (mins)'].min()
                                        max_playtime = combined_df['Playtime (mins)'].max()
                                        current_playtime = st.session_state.player_filters[group_idx].get('playtime', (float(min_playtime), float(max_playtime)))
                                        # Ensure current playtime range is within valid bounds
                                        valid_min = max(float(min_playtime), current_playtime[0])
                                        valid_max = min(float(max_playtime), current_playtime[1])
                                        playtime_range = st.slider(
                                            "Playtime (mins)",
                                            min_value=float(min_playtime),
                                            max_value=float(max_playtime),
                                            value=(valid_min, valid_max),
                                            key=f"playtime_filter_{group_idx}"
                                        )
                                        st.session_state.player_filters[group_idx]['playtime'] = playtime_range
                                
                                # Apply filters
                                filtered_df = combined_df.copy()
                                if selected_nicknames:
                                    filtered_df = filtered_df[filtered_df['nickname'].isin(selected_nicknames)]
                                if selected_positions:
                                    filtered_df = filtered_df[filtered_df['primary_position_name'].isin(selected_positions)]
                                if selected_clubs:
                                    filtered_df = filtered_df[filtered_df['club'].isin(selected_clubs)]
                                if 'Playtime (mins)' in combined_df.columns:
                                    filtered_df = filtered_df[
                                        (filtered_df['Playtime (mins)'] >= playtime_range[0]) &
                                        (filtered_df['Playtime (mins)'] <= playtime_range[1])
                                    ]
                                
                                # Create aggregation dictionary
                                agg_dict = {}
                                for col in filtered_df.columns:
                                    if col not in ['nickname', 'primary_position_name', 'club', 'match_code_source']:
                                        if pd.api.types.is_numeric_dtype(filtered_df[col]):
                                            agg_dict[col] = 'sum'
                                        else:
                                            agg_dict[col] = 'first'
                                
                                # Determine group columns based on selection
                                group_cols = ['nickname']  # Always group by nickname
                                if group_by in ["Player + Position", "Player + Position + Club"]:
                                    group_cols.append('primary_position_name')
                                if group_by in ["Player + Club", "Player + Position + Club"]:
                                    group_cols.append('club')
                                
                                # Perform grouping
                                filtered_df = filtered_df.groupby(group_cols, as_index=False).agg(agg_dict)
                                
                                # Add save dataset functionality
                                st.markdown("### Save Filtered Player Dataset")
                                save_cols = st.columns([3, 1])
                                with save_cols[0]:
                                    dataset_name = st.text_input(
                                        "Dataset Name",
                                        key=f"player_dataset_name_{group_idx}_player_save_section_",
                                        placeholder="Enter a name for this filtered player dataset"
                                    )
                                with save_cols[1]:
                                    if st.button("💾 Save Player Dataset", key=f"save_player_dataset_{group_idx}_player_save_section_"):
                                        if dataset_name:
                                            # Create a unique key for this dataset
                                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            dataset_key = f"{dataset_name}_{timestamp}"
                                            
                                            # Calculate per 90 metrics
                                            per_90_df = filtered_df.copy()
                                            if 'Playtime (mins)' in per_90_df.columns:
                                                # Calculate the per 90 factor
                                                per_90_factor = 90 / per_90_df['Playtime (mins)']
                                                
                                                # Apply per 90 calculation to all numeric columns except Playtime
                                                for col in per_90_df.columns:
                                                    if pd.api.types.is_numeric_dtype(per_90_df[col]) and col != 'Playtime (mins)':
                                                        per_90_df[f'{col}_per_90'] = per_90_df[col] * per_90_factor
                                            
                                            # Calculate percentiles for all numeric columns
                                            percentile_df = per_90_df.copy()
                                            for col in percentile_df.columns:
                                                if pd.api.types.is_numeric_dtype(percentile_df[col]):
                                                    # Calculate percentile rank (0-100)
                                                    percentile_df[f'{col}_percentile'] = percentile_df[col].rank(pct=True) * 100
                                            
                                            # Store the dataset and its metadata
                                            st.session_state.saved_player_datasets[dataset_key] = {
                                                "name": dataset_name,
                                                "data": filtered_df,  # Original data
                                                "per_90_data": per_90_df,  # Per 90 metrics
                                                "percentile_data": percentile_df,  # Percentile data
                                                "group": group_idx,
                                                "group_by": group_by,
                                                "filters": st.session_state.player_filters[group_idx].copy(),
                                                "timestamp": timestamp,
                                                "row_count": len(filtered_df)
                                            }
                                            st.success(f"Player dataset '{dataset_name}' saved successfully with per 90 metrics and percentiles!")
                                            st.rerun()
                                        else:
                                            st.error("Please enter a name for the dataset")
                                
                                # Display filtered and grouped data
                                st.dataframe(filtered_df)
                                
                                # Download button for filtered data
                                csv = filtered_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download filtered player data as CSV",
                                    data=csv,
                                    file_name="filtered_player_data.csv",
                                    mime='text/csv',
                                    key=f"download_filtered_player_data_{group_idx}"
                                )
                            else:
                                # Display original data if required columns don't exist
                                st.dataframe(combined_df)
                                csv = combined_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download player data as CSV",
                                    data=csv,
                                    file_name="player_data.csv",
                                    mime='text/csv',
                                    key=f"download_player_data_{group_idx}"
                                )
                    else:
                        st.info(f"No player data (CSV files) loaded for {st.session_state.group_names[group_idx]}. Click 'Load Events' for this group in the sidebar.")

                    # Add Saved Player Datasets Repository section (and management)
                    st.markdown("---")
                    st.header("Saved Player Datasets Repository")
                    if st.session_state.saved_player_datasets:
                        saved_player_datasets_info = []
                        for key, dataset in st.session_state.saved_player_datasets.items():
                            saved_player_datasets_info.append({
                                "Name": dataset["name"],
                                "Group": st.session_state.group_names[dataset["group"]],
                                "Group By": dataset["group_by"],
                                "Rows": dataset["row_count"],
                                "Saved At": datetime.datetime.strptime(dataset["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"),
                                "Key": key
                            })
                        saved_player_datasets_df = pd.DataFrame(saved_player_datasets_info)
                        st.dataframe(saved_player_datasets_df, hide_index=True)
                        st.markdown("### Player Dataset Management")
                        dataset_cols = st.columns([3, 1, 1])
                        with dataset_cols[0]:
                            selected_dataset = st.selectbox(
                                "Select Player Dataset",
                                options=list(st.session_state.saved_player_datasets.keys()),
                                format_func=lambda x: f"{st.session_state.saved_player_datasets[x]['name']} ({st.session_state.saved_player_datasets[x]['row_count']} rows)",
                                key=f"select_player_dataset_{group_idx}"
                            )
                        with dataset_cols[1]:
                            if st.button("📥 Load Player Dataset", key=f"load_saved_player_dataset_{group_idx}"):
                                if selected_dataset:
                                    dataset = st.session_state.saved_player_datasets[selected_dataset]
                                    st.session_state.csv_data_by_group[st.session_state.active_group_index] = {"loaded_player_data.csv": dataset["data"]}
                                    st.session_state.player_filters[st.session_state.active_group_index] = dataset["filters"]
                                    st.success(f"Player dataset '{dataset['name']}' loaded into current group!")
                        with dataset_cols[2]:
                            if st.button("🗑️ Delete Player Dataset", key=f"delete_saved_player_dataset_{group_idx}"):
                                if selected_dataset:
                                    del st.session_state.saved_player_datasets[selected_dataset]
                                    st.success("Player dataset deleted successfully!")
                                st.rerun()
                        # Add Visualization Tabs for Player Data (keep as is)
                        if selected_dataset:
                            st.markdown("### Player Data Visualizations")
                            is_player_dataset = selected_dataset in st.session_state.saved_player_datasets
                            dataset = st.session_state.saved_player_datasets[selected_dataset] if is_player_dataset else st.session_state.saved_datasets[selected_dataset]
                            available_metrics = ["Raw Metrics"]
                            if "per_90_data" in dataset:
                                available_metrics.append("Per 90 Metrics")
                            if "percentile_data" in dataset:
                                available_metrics.append("Percentiles")
                            metric_type = st.radio(
                                "Select Metric Type:",
                                options=available_metrics,
                                horizontal=True,
                                key=f"metric_type_{selected_dataset}_{group_idx}"
                            )
                            if metric_type == "Raw Metrics":
                                display_df = dataset["data"]
                            elif metric_type == "Per 90 Metrics" and "per_90_data" in dataset:
                                display_df = dataset["per_90_data"]
                            elif metric_type == "Percentiles" and "percentile_data" in dataset:
                                display_df = dataset["percentile_data"]
                            else:
                                display_df = dataset["data"]
                            viz_tabs = st.tabs([
                                "Bar Chart",
                                "Pizza Chart",
                                "Radar Chart",
                                "Table View"
                            ])
                            with viz_tabs[0]:
                                render_bar_tab(st, display_df, color_picker, download_figure, key_prefix=f"{group_idx}_{selected_dataset}")
                            with viz_tabs[1]:
                                render_pizza_tab(st, display_df, color_picker, download_figure, key_prefix=f"{group_idx}_{selected_dataset}")
                            with viz_tabs[2]:
                                render_radar_tab(st, display_df, color_picker, download_figure, key_prefix=f"{group_idx}_{selected_dataset}")
                            with viz_tabs[3]:
                                render_table_tab(st, display_df, color_picker, download_figure, key_prefix=f"{group_idx}_{selected_dataset}")
                        else:
                            st.info("No saved datasets available. Save some datasets first to use this feature.")
                    else:
                        st.info("No saved player datasets yet. Use the save functionality above to create your first player dataset.")

# --- Match Selection List (Main Page - Keep as before) ---
if not st.session_state.show_event_data:
    st.markdown("---")
    st.header("Available Matches for Selection")
    filtered_matches_for_ui_list = []
    active_match_list_filters = st.session_state.group_filters[active_group_index]
    _ml_selected_tournaments = active_match_list_filters['tournaments']
    _ml_selected_seasons = active_match_list_filters['seasons']
    _ml_selected_teams = active_match_list_filters['teams']
    _ml_selected_players = active_match_list_filters['players']
    _ml_selected_stadiums = active_match_list_filters['stadiums']
    _ml_selected_referees = active_match_list_filters['referees']
    _ml_selected_result = active_match_list_filters['result']
    _ml_selected_player_status = active_match_list_filters['player_status']

    for match in all_matches_for_selection_ui:
        home = match['teams']['home']
        away = match['teams']['away']
        match_info = match['match_info']
        home_name = home['team_info']['name']
        away_name = away['team_info']['name']
        home_score = home['score']
        away_score = away['score']

        # Apply filters
        tournament_pass = not _ml_selected_tournaments or match['tournament_name'] in _ml_selected_tournaments
        season_pass = not _ml_selected_seasons or match['season_name'] in _ml_selected_seasons
        team_pass = not _ml_selected_teams or home_name in _ml_selected_teams or away_name in _ml_selected_teams
        stadium_pass = not _ml_selected_stadiums or match_info.get('stadium') in _ml_selected_stadiums
        referee_pass = not _ml_selected_referees or match_info.get('referee') in _ml_selected_referees

        # Player filter logic
        player_pass = True
        if _ml_selected_players:
            player_found = False
            home_starters = [p['player_name'] for p in home['players'].get('starting', [])]
            home_subs = [p['player_name'] for p in home['players'].get('subs', [])]
            away_starters = [p['player_name'] for p in away['players'].get('starting', [])]
            away_subs = [p['player_name'] for p in away['players'].get('subs', [])]

            if _ml_selected_player_status == "Starter":
                if any(p_name in _ml_selected_players for p_name in home_starters + away_starters):
                    player_found = True
            elif _ml_selected_player_status == "Sub":
                if any(p_name in _ml_selected_players for p_name in home_subs + away_subs):
                    player_found = True
            else:  # "All" player statuses
                if any(p_name in _ml_selected_players for p_name in home_starters + home_subs + away_starters + away_subs):
                    player_found = True
            player_pass = player_found

        # Result filter logic
        result_pass = True
        if _ml_selected_result != "All" and _ml_selected_teams:
            match_result_satisfied = False
            for team_name_filter in _ml_selected_teams:
                if team_name_filter == home_name:
                    if (_ml_selected_result == "Win" and home_score > away_score) or \
                       (_ml_selected_result == "Draw" and home_score == away_score) or \
                       (_ml_selected_result == "Loss" and home_score < away_score):
                        match_result_satisfied = True
                        break
                elif team_name_filter == away_name:
                    if (_ml_selected_result == "Win" and away_score > home_score) or \
                       (_ml_selected_result == "Draw" and home_score == away_score) or \
                       (_ml_selected_result == "Loss" and away_score < home_score):
                        match_result_satisfied = True
                        break
            result_pass = match_result_satisfied
        elif _ml_selected_result != "All" and not _ml_selected_teams:
            result_pass = False

        # Add match to filtered list if all filters pass
        if tournament_pass and season_pass and team_pass and stadium_pass and referee_pass and player_pass and result_pass:
            filtered_matches_for_ui_list.append(match)

    # Group matches by season and tournament
    seasons_grouped = {}
    for match_item_sel_ui in filtered_matches_for_ui_list:
        key = (match_item_sel_ui['tournament_name'], match_item_sel_ui['season_name'])
        if key not in seasons_grouped:
            seasons_grouped[key] = []
        seasons_grouped[key].append(match_item_sel_ui)

    seasons_sorted = sorted(seasons_grouped.keys(), reverse=True)

    # Display appropriate message based on filter results
    if not seasons_grouped and (any(_ml_selected_tournaments) or any(_ml_selected_seasons) or 
                              any(_ml_selected_teams) or any(_ml_selected_players) or 
                              any(_ml_selected_stadiums) or any(_ml_selected_referees) or 
                              _ml_selected_result != "All" or _ml_selected_player_status != "All"):
        st.info("No matches found for the current Match List filters. Adjust filters in the sidebar (under '3. Filters for Match List Below').")
    elif not seasons_grouped and not all_matches_for_selection_ui:
        st.error("No matches available in the master data source. Please check `DATA_PATH`.")
    elif not seasons_grouped and all_matches_for_selection_ui:
        st.info("All matches are currently filtered out by the Match List filters, or no filters are set. Adjust filters in the sidebar.")

    # Display matches with pagination
    ITEMS_PER_PAGE = 10
    for tournament_name, season_name in seasons_sorted:
        expander_key = f"expander_matchlist_{tournament_name}_{season_name}"
        expander_state_key = f"page_matchlist_{tournament_name}_{season_name}"
        
        if expander_state_key not in st.session_state:
            st.session_state[expander_state_key] = 1

        with st.expander(f"{season_name} - {tournament_name}", expanded=False):
            matches_in_season_group = sorted(
                seasons_grouped[(tournament_name, season_name)],
                key=lambda x: (x['match_info']['date'], x['match_info'].get('time', '00:00:00')),
                reverse=True
            )
            
            total_items = len(matches_in_season_group)
            total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))
            current_page = st.session_state[expander_state_key]

            # Pagination controls
            cols_pagination = st.columns([2, 1, 1, 1, 2])
            with cols_pagination[1]:
                st.button("←", key=f"prev_matchlist_{expander_state_key}",
                         on_click=change_page, args=(expander_state_key, -1),
                         disabled=current_page <= 1)
            with cols_pagination[2]:
                st.write(f"Page {current_page}/{total_pages}", style={"text-align": "center"})
            with cols_pagination[3]:
                st.button("→", key=f"next_matchlist_{expander_state_key}",
                         on_click=change_page, args=(expander_state_key, 1),
                         disabled=current_page >= total_pages)

            st.markdown("***")

            # Bulk actions
            action_selectbox_key = f"bulk_action_select_matchlist_{expander_state_key}"
            action_options = ["--Select Action--", "Select All Matches", "Deselect All Matches", "Select Last N Matches"]
            st.selectbox("Season Actions (Match List):",
                        options=action_options,
                        key=action_selectbox_key,
                        index=0,
                        on_change=handle_bulk_action_change,
                        args=(action_selectbox_key, expander_state_key, matches_in_season_group, st.session_state.active_group_index))

            # Last N matches selection
            show_last_n_key = f"show_last_n_input_matchlist_{expander_state_key}"
            if st.session_state.get(show_last_n_key, False):
                num_last_matches_key = f"num_last_matchlist_{expander_state_key}"
                if num_last_matches_key not in st.session_state:
                    st.session_state[num_last_matches_key] = 5

                cols_last_n = st.columns([1, 2])
                with cols_last_n[0]:
                    st.number_input("Number (N):",
                                  min_value=1,
                                  max_value=len(matches_in_season_group),
                                  key=num_last_matches_key,
                                  label_visibility="visible")
                with cols_last_n[1]:
                    st.button("Confirm Select Last N",
                             key=f"confirm_last_n_matchlist_{expander_state_key}",
                             on_click=handle_select_last_n,
                             args=(expander_state_key, matches_in_season_group, st.session_state.active_group_index))

            st.markdown("***")

            # Display paginated matches
            start_idx = (current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            paginated_matches_for_display = matches_in_season_group[start_idx:end_idx]

            for match_card_item in paginated_matches_for_display:
                home = match_card_item['teams']['home']
                away = match_card_item['teams']['away']
                match_info = match_card_item['match_info']
                match_identifier_for_checkbox = {
                    "match_id": match_info['match_id'],
                    "season_id": match_card_item['season_id']
                }
                match_key_for_checkbox = f"match_cb_{match_info['match_id']}_{match_card_item['season_id']}_group_{st.session_state.active_group_index}"

                with st.container():
                    match_cols = st.columns([1, 12])
                    with match_cols[0]:
                        is_selected_in_active_group = any(
                            item['match_id'] == match_identifier_for_checkbox['match_id'] and
                            item['season_id'] == match_identifier_for_checkbox['season_id']
                            for item in st.session_state.selected_matches_by_group[st.session_state.active_group_index]
                        )
                        st.checkbox("",
                                  value=is_selected_in_active_group,
                                  key=match_key_for_checkbox,
                                  on_change=handle_individual_checkbox,
                                  args=(match_key_for_checkbox, match_identifier_for_checkbox, st.session_state.active_group_index))

                    with match_cols[1]:
                        # Date and time display
                        date_str = match_info.get('date')
                        time_str = match_info.get('time')
                        date_display = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y") if date_str else "N/A"
                        time_display = datetime.datetime.strptime(time_str, "%H:%M:%S").strftime("%I:%M %p") if time_str and time_str != "00:00:00" else ""
                        st.markdown(f"<p style='text-align: center; font-size:16px; color:#666'>{date_display} {('• '+time_display) if time_display else ''}</p>",
                                  unsafe_allow_html=True)

                        # Team names and scores
                        home_name = home['team_info']['name']
                        away_name = away['team_info']['name']
                        home_score = home['score']
                        away_score = away['score']
                        styled_score_html = style_score(home_score, away_score)

                        # Team logos
                        home_logo_url = home['team_info'].get('logo')
                        away_logo_url = away['team_info'].get('logo')
                        home_logo_html = f'<img src="{home_logo_url}" style="height: 25px; width: auto; margin-left: 5px;" alt="{home_name} logo">' if home_logo_url else ""
                        away_logo_html = f'<img src="{away_logo_url}" style="height: 25px; width: auto; margin-right: 5px;" alt="{away_name} logo">' if away_logo_url else ""

                        # Match layout
                        html_layout = f"""<div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 5px;">
                            <div style="flex: 5; display: flex; align-items: center; justify-content: flex-end;">
                                <span style="font-size:20px;">{home_name}</span>{home_logo_html}
                            </div>
                            <div style="flex: 2; text-align: center;">
                                <span style="{styled_score_html}">{home_score} - {away_score}</span>
                            </div>
                            <div style="flex: 5; display: flex; align-items: center; justify-content: flex-start;">
                                {away_logo_html}<span style="font-size:20px;">{away_name}</span>
                            </div>
                        </div>"""
                        st.markdown(html_layout, unsafe_allow_html=True)

                        # Goals display
                        if home.get("goals") or away.get("goals"):
                            home_goals_html = "".join([f"<div style='text-align:right; font-size:15px;'>{g.get('scorer', 'N/A')} {g.get('minute', '')}\'</div>" for g in home.get("goals", [])])
                            away_goals_html = "".join([f"<div style='text-align:left; font-size:15px;'>{g.get('scorer', 'N/A')} {g.get('minute', '')}\'</div>" for g in away.get("goals", [])])
                            goal_layout_html = f"""<div style="display: flex; width: 100%; margin-top: 5px; justify-content: center; align-items: flex-start;">
                                <div style="flex: 5; max-width: 40%;">{home_goals_html}</div>
                                <div style="flex: 2; text-align:center; min-width: 50px; padding-top: 5px;">⚽️</div>
                                <div style="flex: 5; max-width: 40%;">{away_goals_html}</div>
                            </div>"""
                            st.markdown(goal_layout_html, unsafe_allow_html=True)
                            st.markdown(" ")

                        # Stadium and referee info
                        stadium_display = match_info.get('stadium', 'N/A')
                        referee_display = match_info.get('referee', 'N/A')
                        st.markdown(f"<p style='text-align: center; color: #aaa; font-size:15px;'>🏟️ {stadium_display} | 🧑‍⚖️ {referee_display}</p>",
                                  unsafe_allow_html=True)
                        st.divider()