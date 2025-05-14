import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Radar
import time
import hashlib

# Note: color_picker and download_figure are expected to be passed as arguments

# Calculate a hash of parameters to detect changes
def compute_parameters_hash(params, datasets):
    param_str = str(params) + str(datasets)
    return hashlib.md5(param_str.encode()).hexdigest()

# Cache the radar creation to avoid recomputation
@st.cache_resource
def create_radar_chart(_params, _datasets, _style_params):
    """Create and cache the radar chart"""
    # Note: Removed _chart_hash as it's implicitly handled by cache_resource
    try:
        fig, ax = plt.subplots(figsize=(_style_params.get('width', 10), _style_params.get('height', 10)), dpi=72)
        
        lower = [_style_params.get('range_min', 0)] * len(_params)
        upper = [_style_params.get('range_max', 100)] * len(_params)
        
        radar = Radar(
            _params, lower, upper,
            num_rings=_style_params.get('num_rings', 4),
            ring_width=_style_params.get('ring_width', 3),
            center_circle_radius=_style_params.get('center_circle_radius', 1),
            round_int=[True] * len(_params) # Assuming integer rounding is desired
        )
        
        radar.setup_axis(ax=ax, facecolor=_style_params.get('background_color', '#f9f9f9'))
        radar.draw_circles(ax=ax, facecolor=_style_params.get('ring_color', '#cccccc'), alpha=0.1)
        radar.spoke(ax=ax, color=_style_params.get('line_color', '#999999'), 
                   linestyle=_style_params.get('line_style', '-'), zorder=2)
        
        vertices_list = []
        for i, dataset in enumerate(_datasets):
            values = dataset.get('values', [])
            name = dataset.get('name', f'Data {i+1}')
            color = dataset.get('color', f'#1f77b4')
            
            if len(values) == len(_params):
                radar_poly, vertices = radar.draw_radar_solid(
                    values, ax=ax, 
                    kwargs={
                        'facecolor': color, 'alpha': _style_params.get('alpha', 0.7), 
                        'edgecolor': color, 'label': name, 'linewidth': 2
                    }
                )
                vertices_list.append(vertices)
                
                if _style_params.get('show_markers', True):
                    ax.scatter(
                        vertices[:, 0], vertices[:, 1],
                        c=color, edgecolors='white', marker='o', 
                        s=_style_params.get('marker_size', 60), 
                        zorder=i+10, alpha=_style_params.get('alpha', 0.7) + 0.2
                    )
        
        radar.draw_range_labels(ax=ax, fontsize=10)
        radar.draw_param_labels(ax=ax, fontsize=10)
        
        if _style_params.get('show_legend', True) and len(_datasets) > 1:
            ax.legend(loc=_style_params.get('legend_loc', 'upper right'))
        
        return fig, ax, vertices_list
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")
        st.exception(e)
        return None, None, []

def render_radar_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("📊 Radar Chart Customization")
    
    # Initialize session state variables if they don't exist
    if 'radar_params' not in st.session_state:
        st.session_state.radar_params = ["Shooting", "Passing", "Dribbling", "Defending", "Physical"]
    if 'radar_range' not in st.session_state:
        st.session_state.radar_range = (0, 100)
    if 'radar_datasets' not in st.session_state:
        st.session_state.radar_datasets = []
    if 'radar_style' not in st.session_state:
        st.session_state.radar_style = {
            'num_rings': 4,
            'ring_width': 3,
            'center_circle_radius': 1,
            'background_color': '#f9f9f9',
            'ring_color': '#cccccc',
            'line_color': '#999999',
            'line_style': '-',
            'alpha': 0.7,
            'show_legend': True,
            'legend_loc': 'upper right',
            'show_markers': True,
            'marker_size': 60,
            'width': 10,
            'height': 10
        }
    
    # --- Parameter Input Form ---
    with st.form(f"radar_parameters_form_{key_prefix}"):
        st.subheader("Parameters")
        st.text_area(
            "Enter parameters (comma-separated)", 
            value=",".join(st.session_state.radar_params),
            key=f"radar_params_input_area_{key_prefix}"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Minimum Value", 0, 100, st.session_state.radar_range[0], key=f"radar_min_{key_prefix}")
        with col2:
            st.slider("Maximum Value", 0, 100, st.session_state.radar_range[1], key=f"radar_max_{key_prefix}")
        
        params_submitted = st.form_submit_button("Set Parameters")
        if params_submitted:
            params_input_value = st.session_state.radar_params_input_area 
            new_params = [p.strip() for p in params_input_value.split(',') if p.strip()]
            if new_params:
                 st.session_state.radar_params = new_params
                 st.session_state.radar_range = (st.session_state.radar_min, st.session_state.radar_max)
                 st.session_state.radar_datasets = [] 
                 st.toast("Parameters updated. Please re-enter dataset values.") # Use toast
                 st.rerun()
            else:
                 st.warning("Please enter at least one parameter.")
    
    params = st.session_state.radar_params
    range_min, range_max = st.session_state.radar_range
    num_params = len(params)

    # --- Multi-dataset Form ---
    with st.form(f"radar_datasets_form_{key_prefix}"):
        st.subheader("Data Sets")
        num_datasets = st.number_input(
            "Number of data sets", min_value=1, max_value=10, 
            value=max(1, len(st.session_state.radar_datasets)), 
            key=f"radar_num_datasets_{key_prefix}", help="Number of players/teams to compare"
        )
        
        datasets_container = st.container()
        all_datasets = []
        current_datasets = st.session_state.get('radar_datasets', [])
        data_valid = True

        for i in range(num_datasets):
             default_name = current_datasets[i]['name'] if i < len(current_datasets) else f"Data {i+1}"
             default_color = current_datasets[i]['color'] if i < len(current_datasets) else f"#{abs(hash(default_name)) % 0xFFFFFF:06X}"
             default_values_list = current_datasets[i]['values'] if i < len(current_datasets) and len(current_datasets[i].get('values', [])) == num_params else ([0] * num_params)
             default_values_str = ",".join(map(str, default_values_list))

             with datasets_container:
                 st.markdown(f"--- Data Set {i+1} ---")
                 col1, col2 = st.columns([3, 1])
                 with col1:
                      dataset_name = st.text_input(f"Name##{i}", value=default_name, key=f"radar_name_{i}_{key_prefix}")
                      values_input = st.text_area(
                           f"Values for '{dataset_name}' ({num_params} expected)##{i}", 
                           value=default_values_str, key=f"radar_values_{i}_{key_prefix}"
                      )
                 with col2:
                      dataset_color = color_picker(f"Color##{i}", default_color, key=f"radar_dataset_color_{i}_{key_prefix}")
                 
                 try:
                      values = [float(v.strip()) for v in values_input.split(',') if v.strip()]
                      if len(values) == num_params:
                           all_datasets.append({'name': dataset_name, 'values': values, 'color': dataset_color})
                      else:
                           st.warning(f"Dataset '{dataset_name}': Expected {num_params} values, found {len(values)}. Ignoring.")
                           data_valid = False # Mark data as invalid for this dataset
                 except ValueError:
                      st.error(f"Dataset '{dataset_name}': Please enter numeric values only. Ignoring.")
                      data_valid = False # Mark data as invalid

        datasets_submitted = st.form_submit_button("Update Data Sets")
        if datasets_submitted:
             if data_valid and len(all_datasets) == num_datasets: 
                  st.session_state.radar_datasets = all_datasets
                  st.toast("Datasets updated.")
                  st.rerun()
             elif not data_valid:
                  st.error("Some dataset entries have incorrect value counts or non-numeric values. Please correct and resubmit.")
             # Implicit else: no datasets entered, do nothing

    # --- Radar Chart Style Form ---
    with st.form(f"radar_style_form_{key_prefix}"):
        st.subheader("Chart Style")
        current_style = st.session_state.radar_style
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Rings & Background**")
            num_rings = st.slider("Num Rings", 3, 10, current_style.get('num_rings', 4), key=f"radar_num_rings_{key_prefix}")
            ring_width = st.slider("Ring Width", 1, 10, current_style.get('ring_width', 3), key=f"radar_ring_width_{key_prefix}")
            center_circle_radius = st.slider("Center Radius", 0, 10, current_style.get('center_circle_radius', 1), key=f"radar_center_radius_{key_prefix}")
            background_color = color_picker("BG Color", current_style.get('background_color', "#f9f9f9"), key=f"radar_bg_color_{key_prefix}")
            ring_color = color_picker("Ring Color", current_style.get('ring_color', "#cccccc"), key=f"radar_ring_color_{key_prefix}")
            
        with col2:
            st.markdown("**Lines & Opacity**")
            line_color = color_picker("Spoke Color", current_style.get('line_color', "#999999"), key=f"radar_line_color_{key_prefix}")
            line_style_options = ["-", "--", "-.", ":"]
            line_style_index = line_style_options.index(current_style.get('line_style', "-"))
            line_style = st.selectbox("Line Style", line_style_options, index=line_style_index, key=f"radar_line_style_{key_prefix}")
            alpha = st.slider("Fill Opacity", 0.0, 1.0, current_style.get('alpha', 0.7), key=f"radar_alpha_{key_prefix}")
            show_markers = st.checkbox("Show Markers", current_style.get('show_markers', True), key=f"radar_markers_{key_prefix}")
            marker_size_val = current_style.get('marker_size', 60)
            marker_size = st.slider("Marker Size", 20, 200, marker_size_val, key=f"radar_marker_size_{key_prefix}", disabled=not show_markers)

        with col3:
            st.markdown("**Legend & Size**")
            show_legend = st.checkbox("Show Legend", current_style.get('show_legend', True), key=f"radar_legend_{key_prefix}")
            legend_loc_options = ["upper right", "upper left", "lower left", "lower right", 
                                  "center left", "center right", "upper center", "lower center", "best"]
            legend_loc_index = legend_loc_options.index(current_style.get('legend_loc', "upper right"))
            legend_loc = st.selectbox("Legend Loc", legend_loc_options, index=legend_loc_index, key=f"radar_legend_loc_{key_prefix}", disabled=not show_legend)
            chart_width = st.slider("Chart Width", 4, 20, current_style.get('width', 10), key=f"radar_chart_width_{key_prefix}")
            chart_height = st.slider("Chart Height", 4, 20, current_style.get('height', 10), key=f"radar_chart_height_{key_prefix}")
        
        style_submitted = st.form_submit_button("Apply Style")
        if style_submitted:
            updated_style_params = {
                'num_rings': num_rings, 'ring_width': ring_width, 'center_circle_radius': center_circle_radius,
                'background_color': background_color, 'ring_color': ring_color, 'line_color': line_color,
                'line_style': line_style, 'alpha': alpha,
                'show_legend': show_legend, 'legend_loc': legend_loc,
                'show_markers': show_markers, 'marker_size': marker_size, # Store size even if disabled 
                'width': chart_width, 'height': chart_height,
                'range_min': st.session_state.radar_range[0], 'range_max': st.session_state.radar_range[1]
            }
            st.session_state.radar_style = updated_style_params
            st.toast("Chart style updated.")
            st.rerun()
    
    # --- Generate Radar Chart ---
    valid_datasets = st.session_state.get('radar_datasets', [])
    if valid_datasets:
        chart_placeholder = st.empty()
        with chart_placeholder, st.spinner("Generating radar chart..."):
            start_time = time.time()
            # Call the cached function
            fig, ax, vertices_list = create_radar_chart(
                st.session_state.radar_params,
                valid_datasets,
                st.session_state.radar_style
            )
            render_time = time.time() - start_time
            
            if fig is not None:
                st.pyplot(fig)
                st.caption(f"Render time: {render_time:.3f}s")
                buf = download_figure(fig, dpi=150)
                st.download_button(
                    label="Download Radar Chart (High Quality)", data=buf,
                    file_name="radar_comparison.png", mime="image/png", key="download_radar"
                )
    elif not params:
         st.warning("Please set parameters first.")
    else:
        st.info("Enter valid data sets to generate the radar chart.")
