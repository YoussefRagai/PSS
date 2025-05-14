import streamlit as st
from .pitch_tab import show_pitch_tab
from .scatter_tab import show_scatter_tab
from .table_tab import show_table_tab
from .bar_tab import show_bar_tab
from .radar_tab import show_radar_tab
from .pizza_tab import show_pizza_tab
from .data_loader import load_master_seasons, get_available_matches

# Initialize session state
if 'master_seasons' not in st.session_state:
    st.session_state.master_seasons = load_master_seasons()
if 'available_matches' not in st.session_state:
    st.session_state.available_matches = get_available_matches()
if 'saved_datasets' not in st.session_state:
    st.session_state.saved_datasets = {}

st.set_page_config(
    page_title="Football Analysis Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("Football Analysis Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Pitch Analysis", 
    "Scatter Plot", 
    "Table View",
    "Bar Charts",
    "Radar Charts",
    "Pizza Charts"
])

with tab1:
    show_pitch_tab()

with tab2:
    show_scatter_tab()

with tab3:
    show_table_tab()

with tab4:
    show_bar_tab()

with tab5:
    show_radar_tab()

with tab6:
    show_pizza_tab() 