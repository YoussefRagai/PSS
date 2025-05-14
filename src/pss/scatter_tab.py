import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import json
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

# Note: color_picker and download_figure are expected to be passed as arguments

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic_func(x, a, b, c):
    return a * np.log(b * x) + c

def power_func(x, a, b, c):
    return a * np.power(x, b) + c

def render_scatter_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("📊 Scatter Plot Customization")
    
    # Initialize session state variables if they don't exist
    if f'selected_datasets_{key_prefix}' not in st.session_state:
        st.session_state[f'selected_datasets_{key_prefix}'] = []
    if f'saved_plot_configs_{key_prefix}' not in st.session_state:
        st.session_state[f'saved_plot_configs_{key_prefix}'] = {}

    # Add dataset selection section
    with st.expander("📁 Dataset Selection", expanded=True):
        if 'saved_datasets' in st.session_state and st.session_state.saved_datasets:
            available_datasets = list(st.session_state.saved_datasets.keys())
            selected_datasets = st.multiselect(
                "Select Datasets to Plot",
                options=available_datasets,
                format_func=lambda x: f"{st.session_state.saved_datasets[x]['name']} ({st.session_state.saved_datasets[x]['row_count']} rows)",
                key=f"scatter_dataset_selector_{key_prefix}"
            )
            st.session_state[f'selected_datasets_{key_prefix}'] = selected_datasets
        else:
            st.info("No saved datasets available. Save some datasets first to use this feature.")
            selected_datasets = []

    # Initialize variables
    x_column = None
    y_column = None
    event_column = None
    dataset_styles = {}

    # Add data plotting section
    with st.expander("📊 Data Plotting", expanded=True):
        if selected_datasets:
            # Get numeric columns from the first dataset for coordinate selection
            first_dataset = st.session_state.saved_datasets[selected_datasets[0]]['data']
            numeric_columns = first_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("X-axis", numeric_columns, key=f"scatter_x_column_{key_prefix}")
                with col2:
                    y_column = st.selectbox("Y-axis", numeric_columns, key=f"scatter_y_column_{key_prefix}")

                # Event type selection
                event_columns = [col for col in first_dataset.columns if col not in [x_column, y_column]]
                event_column = st.selectbox("Color By", event_columns, key=f"scatter_event_column_{key_prefix}")
                
                # Dataset-specific styling
                st.markdown("---")
                st.subheader("Dataset Styling")
                
                # Create tabs for each dataset's styling
                dataset_tabs = st.tabs([f"Style for {st.session_state.saved_datasets[key]['name']}" for key in selected_datasets])
                
                for idx, dataset_key in enumerate(selected_datasets):
                    dataset = st.session_state.saved_datasets[dataset_key]
                    with dataset_tabs[idx]:
                        # Get unique event types for this dataset
                        event_types = dataset['data'][event_column].unique().tolist()
                        
                        # Color selection for each event type
                        event_colors = {}
                        for event in event_types:
                            event_colors[event] = color_picker(
                                f"Color for {event}", 
                                "#1f77b4", 
                                key=f"scatter_color_{dataset_key}_{event}_{key_prefix}"
                            )
                        
                        # Common styling options
                        col1, col2 = st.columns(2)
                        with col1:
                            point_size = st.slider(
                                "Point Size", 
                                50, 500, 100, 
                                key=f"scatter_point_size_{dataset_key}_{key_prefix}"
                            )
                        with col2:
                            point_alpha = st.slider(
                                "Opacity", 
                                0.0, 1.0, 0.7, 
                                key=f"scatter_alpha_{dataset_key}_{key_prefix}"
                            )
                
                        # Marker options
                        show_markers = st.checkbox(
                            "Show Markers", 
                            True, 
                            key=f"scatter_show_markers_{dataset_key}_{key_prefix}"
                        )
                        if show_markers:
                            marker_style = st.selectbox(
                                "Marker Style",
                                ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "|", "_"],
                                key=f"scatter_marker_style_{dataset_key}_{key_prefix}"
                            )
                        
                        dataset_styles[dataset_key] = {
                            'event_colors': event_colors,
                            'point_size': point_size,
                            'point_alpha': point_alpha,
                            'show_markers': show_markers,
                            'marker_style': marker_style if show_markers else None
                        }
            else:
                st.warning("Need at least 2 numeric columns for coordinates")
        else:
            st.info("Select at least one dataset to plot.")

    # Add plot customization section
    with st.expander("🎨 Plot Customization", expanded=False):
        # Title options
        show_title = st.checkbox("Show Title", True, key=f"scatter_show_title_{key_prefix}")
        if show_title:
            title_text = st.text_input("Plot Title", "Scatter Plot", key=f"scatter_title_text_{key_prefix}")
            title_size = st.slider("Title Size", 10, 30, 16, key=f"scatter_title_size_{key_prefix}")
            title_color = color_picker("Title Color", "#000000", key=f"scatter_title_color_{key_prefix}")
            title_position = st.selectbox(
                "Title Position",
                ["center", "left", "right"],
                key=f"scatter_title_position_{key_prefix}"
            )

        # Legend options
        show_legend = st.checkbox("Show Legend", True, key=f"scatter_show_legend_{key_prefix}")
        if show_legend:
            legend_position = st.selectbox(
                "Legend Position",
                [
                    "Outside Right",
                    "Outside Left",
                    "Outside Top",
                    "Outside Bottom",
                    "Upper Right",
                    "Upper Left",
                    "Lower Right",
                    "Lower Left"
                ],
                key=f"scatter_legend_position_{key_prefix}"
            )
            legend_title = st.text_input("Legend Title", "", key=f"scatter_legend_title_{key_prefix}")
            legend_font_size = st.slider("Legend Font Size", 8, 20, 12, key=f"scatter_legend_font_size_{key_prefix}")
            legend_frame = st.checkbox("Show Legend Frame", True, key=f"scatter_legend_frame_{key_prefix}")
            legend_alpha = st.slider("Legend Background Alpha", 0.0, 1.0, 0.8, key=f"scatter_legend_alpha_{key_prefix}")
        
        # Axis options
        st.markdown("---")
        st.subheader("Axis Customization")
        show_grid = st.checkbox("Show Grid", True, key=f"scatter_show_grid_{key_prefix}")
        if show_grid:
            col1, col2 = st.columns(2)
            with col1:
                grid_alpha = st.slider("Grid Opacity", 0.0, 1.0, 0.2, key=f"scatter_grid_alpha_{key_prefix}")
                show_minor_grid = st.checkbox("Show Minor Grid Lines", False, key=f"scatter_show_minor_grid_{key_prefix}")
                if show_minor_grid:
                    minor_grid_alpha = st.slider("Minor Grid Opacity", 0.0, 1.0, 0.1, key=f"scatter_minor_grid_alpha_{key_prefix}")
            with col2:
                grid_style = st.selectbox(
                    "Grid Style",
                    ["Solid", "Dashed", "Dotted", "Dash-dot"],
                    key=f"scatter_grid_style_{key_prefix}"
                )
                grid_color = color_picker("Grid Color", "#cccccc", key=f"scatter_grid_color_{key_prefix}")
        
        axis_font_size = st.slider("Axis Label Font Size", 8, 20, 12, key=f"scatter_axis_font_size_{key_prefix}")
    
        # Additional axis options
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("X-axis Options")
            x_min = st.number_input("X-axis Min", value=None, placeholder="Auto", key=f"scatter_x_min_{key_prefix}")
            x_max = st.number_input("X-axis Max", value=None, placeholder="Auto", key=f"scatter_x_max_{key_prefix}")
            x_rotation = st.slider("X-axis Label Rotation", 0, 90, 0, key=f"scatter_x_rotation_{key_prefix}")
            x_scale = st.selectbox(
                "X-axis Scale",
                ["Linear", "Log", "Symlog"],
                key=f"scatter_x_scale_{key_prefix}"
            )
            x_tick_format = st.selectbox(
                "X-axis Tick Format",
                ["Default", "Scientific", "Percentage", "Integer", "Custom"],
                key=f"scatter_x_tick_format_{key_prefix}"
            )
            if x_tick_format == "Custom":
                x_tick_format_str = st.text_input("X-axis Format String", "{:.2f}", key=f"scatter_x_tick_format_str_{key_prefix}")
            x_tick_interval = st.number_input("X-axis Tick Interval", value=None, placeholder="Auto", key=f"scatter_x_tick_interval_{key_prefix}")
            x_tick_count = st.number_input("X-axis Number of Ticks", value=None, placeholder="Auto", key=f"scatter_x_tick_count_{key_prefix}")
            x_tick_rotation = st.slider("X-axis Tick Rotation", 0, 90, 0, key=f"scatter_x_tick_rotation_{key_prefix}")
            
        with col2:
            st.subheader("Y-axis Options")
            y_min = st.number_input("Y-axis Min", value=None, placeholder="Auto", key=f"scatter_y_min_{key_prefix}")
            y_max = st.number_input("Y-axis Max", value=None, placeholder="Auto", key=f"scatter_y_max_{key_prefix}")
            y_rotation = st.slider("Y-axis Label Rotation", 0, 90, 0, key=f"scatter_y_rotation_{key_prefix}")
            y_scale = st.selectbox(
                "Y-axis Scale",
                ["Linear", "Log", "Symlog"],
                key=f"scatter_y_scale_{key_prefix}"
            )
            y_tick_format = st.selectbox(
                "Y-axis Tick Format",
                ["Default", "Scientific", "Percentage", "Integer", "Custom"],
                key=f"scatter_y_tick_format_{key_prefix}"
            )
            if y_tick_format == "Custom":
                y_tick_format_str = st.text_input("Y-axis Format String", "{:.2f}", key=f"scatter_y_tick_format_str_{key_prefix}")
            y_tick_interval = st.number_input("Y-axis Tick Interval", value=None, placeholder="Auto", key=f"scatter_y_tick_interval_{key_prefix}")
            y_tick_count = st.number_input("Y-axis Number of Ticks", value=None, placeholder="Auto", key=f"scatter_y_tick_count_{key_prefix}")
            y_tick_rotation = st.slider("Y-axis Tick Rotation", 0, 90, 0, key=f"scatter_y_tick_rotation_{key_prefix}")
        
        # Background color
            bg_color = color_picker("Background Color", "#ffffff", key=f"scatter_bg_color_{key_prefix}")

        # Trend line options
        st.markdown("---")
        st.subheader("Trend Line")
        show_trendline = st.checkbox("Show Trend Line", False, key=f"scatter_show_trendline_{key_prefix}")
        if show_trendline:
            col1, col2 = st.columns(2)
            with col1:
                trendline_type = st.selectbox(
                    "Trend Line Type",
                    ["Linear", "Polynomial", "Exponential", "Logarithmic", "Power"],
                    key=f"scatter_trendline_type_{key_prefix}"
                )
                if trendline_type == "Polynomial":
                    polynomial_degree = st.slider("Polynomial Degree", 2, 10, 2, key=f"scatter_polynomial_degree_{key_prefix}")
                trendline_color = color_picker("Trend Line Color", "#ff0000", key=f"scatter_trendline_color_{key_prefix}")
            with col2:
                trendline_width = st.slider("Trend Line Width", 0.5, 5.0, 2.0, key=f"scatter_trendline_width_{key_prefix}")
                trendline_alpha = st.slider("Trend Line Opacity", 0.0, 1.0, 0.8, key=f"scatter_trendline_alpha_{key_prefix}")
            
            st.markdown("---")
            st.subheader("Equation Display")
            show_equation = st.checkbox("Show Equation", True, key=f"scatter_show_equation_{key_prefix}")
            if show_equation:
                col1, col2 = st.columns(2)
                with col1:
                    equation_position = st.selectbox(
                        "Equation Position",
                        ["Top Left", "Top Right", "Bottom Left", "Bottom Right"],
                        key=f"scatter_equation_position_{key_prefix}"
                    )
                    equation_font_size = st.slider("Equation Font Size", 8, 20, 10, key=f"scatter_equation_font_size_{key_prefix}")
                with col2:
                    equation_color = color_picker("Equation Color", "#000000", key=f"scatter_equation_color_{key_prefix}")
                    equation_background = st.checkbox("Show Equation Background", True, key=f"scatter_equation_background_{key_prefix}")
                    if equation_background:
                        equation_bg_color = color_picker("Equation Background Color", "#ffffff", key=f"scatter_equation_bg_color_{key_prefix}")
                        equation_bg_alpha = st.slider("Equation Background Opacity", 0.0, 1.0, 0.8, key=f"scatter_equation_bg_alpha_{key_prefix}")
            
            show_r2 = st.checkbox("Show R² Value", True, key=f"scatter_show_r2_{key_prefix}")
            if show_r2:
                r2_position = st.selectbox(
                    "R² Position",
                    ["Next to Equation", "Below Equation", "Separate"],
                    key=f"scatter_r2_position_{key_prefix}"
                )
                if r2_position == "Separate":
                    r2_font_size = st.slider("R² Font Size", 8, 20, 10, key=f"scatter_r2_font_size_{key_prefix}")
                    r2_color = color_picker("R² Color", "#000000", key=f"scatter_r2_color_{key_prefix}")
    
        # Save/Load Configuration
        st.markdown("---")
        st.subheader("Save/Load Configuration")
        config_name = st.text_input("Configuration Name", key=f"scatter_config_name_{key_prefix}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Configuration", key=f"scatter_save_config_{key_prefix}"):
                if config_name:
                    config = {
                        'title': {
                            'show': show_title,
                            'text': title_text if show_title else "",
                            'size': title_size if show_title else 16,
                            'color': title_color if show_title else "#000000",
                            'position': title_position if show_title else "center"
                        },
                        'legend': {
                            'show': show_legend,
                            'position': legend_position if show_legend else "Outside Right",
                            'title': legend_title if show_legend else "",
                            'font_size': legend_font_size if show_legend else 12,
                            'frame': legend_frame if show_legend else True,
                            'alpha': legend_alpha if show_legend else 0.8
                        },
                        'axis': {
                            'show_grid': show_grid,
                            'grid_alpha': grid_alpha,
                            'font_size': axis_font_size,
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max,
                            'x_rotation': x_rotation,
                            'y_rotation': y_rotation
                        },
                        'background': {
                            'color': bg_color
                        },
                        'trendline': {
                            'show': show_trendline,
                            'type': trendline_type if show_trendline else "Linear",
                            'color': trendline_color if show_trendline else "#ff0000",
                            'width': trendline_width if show_trendline else 2.0,
                            'alpha': trendline_alpha if show_trendline else 0.8,
                            'show_equation': show_equation if show_trendline else True,
                            'show_r2': show_r2 if show_trendline else True
                        },
                        'datasets': {
                            key: {
                                'colors': {str(k): v for k, v in styles['event_colors'].items()},
                                'point_size': styles['point_size'],
                                'point_alpha': styles['point_alpha'],
                                'show_markers': styles['show_markers'],
                                'marker_style': styles['marker_style']
                            } for key, styles in dataset_styles.items()
                        }
                    }
                    st.session_state[f'saved_plot_configs_{key_prefix}'][config_name] = config
                    st.success(f"Configuration '{config_name}' saved successfully!")
                else:
                    st.warning("Please enter a configuration name.")
        
        with col2:
            if st.session_state[f'saved_plot_configs_{key_prefix}']:
                selected_config = st.selectbox(
                    "Load Configuration",
                    options=list(st.session_state[f'saved_plot_configs_{key_prefix}'].keys()),
                    key=f"scatter_load_config_{key_prefix}"
                )
                if st.button("Load Selected Configuration", key=f"scatter_load_selected_config_{key_prefix}"):
                    config = st.session_state[f'saved_plot_configs_{key_prefix}'][selected_config]
                    # Update all the UI elements with the loaded configuration
                    st.session_state[f'scatter_show_title_{key_prefix}'] = config['title']['show']
                    st.session_state[f'scatter_title_text_{key_prefix}'] = config['title']['text']
                    st.session_state[f'scatter_title_size_{key_prefix}'] = config['title']['size']
                    st.session_state[f'scatter_title_color_{key_prefix}'] = config['title']['color']
                    st.session_state[f'scatter_title_position_{key_prefix}'] = config['title']['position']
                    
                    st.session_state[f'scatter_show_legend_{key_prefix}'] = config['legend']['show']
                    st.session_state[f'scatter_legend_position_{key_prefix}'] = config['legend']['position']
                    st.session_state[f'scatter_legend_title_{key_prefix}'] = config['legend']['title']
                    st.session_state[f'scatter_legend_font_size_{key_prefix}'] = config['legend']['font_size']
                    st.session_state[f'scatter_legend_frame_{key_prefix}'] = config['legend']['frame']
                    st.session_state[f'scatter_legend_alpha_{key_prefix}'] = config['legend']['alpha']
                    
                    st.session_state[f'scatter_show_grid_{key_prefix}'] = config['axis']['show_grid']
                    st.session_state[f'scatter_grid_alpha_{key_prefix}'] = config['axis']['grid_alpha']
                    st.session_state[f'scatter_axis_font_size_{key_prefix}'] = config['axis']['font_size']
                    st.session_state[f'scatter_x_min_{key_prefix}'] = config['axis']['x_min']
                    st.session_state[f'scatter_x_max_{key_prefix}'] = config['axis']['x_max']
                    st.session_state[f'scatter_y_min_{key_prefix}'] = config['axis']['y_min']
                    st.session_state[f'scatter_y_max_{key_prefix}'] = config['axis']['y_max']
                    st.session_state[f'scatter_x_rotation_{key_prefix}'] = config['axis']['x_rotation']
                    st.session_state[f'scatter_y_rotation_{key_prefix}'] = config['axis']['y_rotation']
                    
                    st.session_state[f'scatter_bg_color_{key_prefix}'] = config['background']['color']
                    
                    st.session_state[f'scatter_show_trendline_{key_prefix}'] = config['trendline']['show']
                    st.session_state[f'scatter_trendline_type_{key_prefix}'] = config['trendline']['type']
                    st.session_state[f'scatter_trendline_color_{key_prefix}'] = config['trendline']['color']
                    st.session_state[f'scatter_trendline_width_{key_prefix}'] = config['trendline']['width']
                    st.session_state[f'scatter_trendline_alpha_{key_prefix}'] = config['trendline']['alpha']
                    st.session_state[f'scatter_show_equation_{key_prefix}'] = config['trendline']['show_equation']
                    st.session_state[f'scatter_show_r2_{key_prefix}'] = config['trendline']['show_r2']
                    
                    # Update dataset styles
                    for dataset_key, style in config['datasets'].items():
                        for event, color in style['colors'].items():
                            st.session_state[f"scatter_color_{dataset_key}_{event}_{key_prefix}"] = color
                        st.session_state[f"scatter_point_size_{dataset_key}_{key_prefix}"] = style['point_size']
                        st.session_state[f"scatter_alpha_{dataset_key}_{key_prefix}"] = style['point_alpha']
                        st.session_state[f"scatter_show_markers_{dataset_key}_{key_prefix}"] = style['show_markers']
                        st.session_state[f"scatter_marker_style_{dataset_key}_{key_prefix}"] = style['marker_style']
                    
                    st.success(f"Configuration '{selected_config}' loaded successfully!")
                    st.experimental_rerun()

    try:
        # Only create the plot if we have the necessary data
        if selected_datasets and x_column and y_column and event_column:
            # Create figure with adjusted size for legend
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Set background color
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            
            # Plot data from all selected datasets
            for dataset_key in selected_datasets:
                dataset = st.session_state.saved_datasets[dataset_key]
                styles = dataset_styles[dataset_key]
                
                # Plot each event type
                for event in dataset['data'][event_column].unique():
                    event_data = dataset['data'][dataset['data'][event_column] == event]
                    if not event_data.empty:
                        ax.scatter(
                            event_data[x_column],
                            event_data[y_column],
                            color=styles['event_colors'][event],
                            s=styles['point_size'],
                            alpha=styles['point_alpha'],
                            label=f"{event} ({dataset['name']})",
                            marker=styles['marker_style'] if styles['show_markers'] else None
                        )
                
                # Add trend line if enabled
                if show_trendline:
                    x_values = dataset['data'][x_column].values
                    y_values = dataset['data'][y_column].values
                    
                    # Remove any NaN or infinite values
                    mask = np.isfinite(x_values) & np.isfinite(y_values)
                    x_clean = x_values[mask]
                    y_clean = y_values[mask]
                    
                    if len(x_clean) > 1:  # Need at least 2 points for a trend line
                        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                        
                        if trendline_type == "Linear":
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                            y_line = slope * x_line + intercept
                            equation = f"y = {slope:.2f}x + {intercept:.2f}"
                            r2 = r_value**2
                        elif trendline_type == "Polynomial":
                            coeffs = np.polyfit(x_clean, y_clean, polynomial_degree)
                            y_line = np.polyval(coeffs, x_line)
                            # Format equation with proper superscripts
                            equation_parts = []
                            for i, coef in enumerate(coeffs):
                                if i == 0:
                                    equation_parts.append(f"{coef:.2f}x^{polynomial_degree}")
                                elif i == polynomial_degree:
                                    equation_parts.append(f"{coef:.2f}")
                                else:
                                    power = polynomial_degree - i
                                    equation_parts.append(f"{coef:+.2f}x^{power}")
                            equation = "y = " + "".join(equation_parts)
                            r2 = np.corrcoef(y_clean, np.polyval(coeffs, x_clean))[0,1]**2
                        elif trendline_type == "Exponential":
                            try:
                                popt, _ = curve_fit(exponential_func, x_clean, y_clean, maxfev=10000)
                                y_line = exponential_func(x_line, *popt)
                                equation = f"y = {popt[0]:.2f}e^({popt[1]:.2f}x) + {popt[2]:.2f}"
                                r2 = np.corrcoef(y_clean, exponential_func(x_clean, *popt))[0,1]**2
                            except:
                                st.warning(f"Could not fit exponential trend line for {dataset['name']}")
                                continue
                        elif trendline_type == "Logarithmic":
                            try:
                                popt, _ = curve_fit(logarithmic_func, x_clean, y_clean, maxfev=10000)
                                y_line = logarithmic_func(x_line, *popt)
                                equation = f"y = {popt[0]:.2f}ln({popt[1]:.2f}x) + {popt[2]:.2f}"
                                r2 = np.corrcoef(y_clean, logarithmic_func(x_clean, *popt))[0,1]**2
                            except:
                                st.warning(f"Could not fit logarithmic trend line for {dataset['name']}")
                                continue
                        elif trendline_type == "Power":
                            try:
                                popt, _ = curve_fit(power_func, x_clean, y_clean, maxfev=10000)
                                y_line = power_func(x_line, *popt)
                                equation = f"y = {popt[0]:.2f}x^({popt[1]:.2f}) + {popt[2]:.2f}"
                                r2 = np.corrcoef(y_clean, power_func(x_clean, *popt))[0,1]**2
                            except:
                                st.warning(f"Could not fit power trend line for {dataset['name']}")
                                continue
                        
                        # Plot trend line
                        ax.plot(x_line, y_line, 
                               color=trendline_color,
                               linewidth=trendline_width,
                               alpha=trendline_alpha,
                               label=f"Trend Line ({dataset['name']})")
                        
                        # Add equation and R² if requested
                        if show_equation or show_r2:
                            label_parts = []
                            if show_equation:
                                label_parts.append(equation)
                            if show_r2:
                                if r2_position == "Next to Equation":
                                    label_parts[-1] += f" (R² = {r2:.3f})"
                                elif r2_position == "Below Equation":
                                    label_parts.append(f"R² = {r2:.3f}")
                                else:  # Separate
                                    ax.text(0.02, 0.98 - 0.05 * (len(st.session_state[f'saved_plot_configs_{key_prefix}']) + 1),
                                           f"R² = {r2:.3f}",
                                           transform=ax.transAxes,
                                           fontsize=r2_font_size,
                                           color=r2_color,
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            # Position mapping
                            position_map = {
                                "Top Left": (0.02, 0.98),
                                "Top Right": (0.98, 0.98),
                                "Bottom Left": (0.02, 0.02),
                                "Bottom Right": (0.98, 0.02)
                            }
                            
                            x_pos, y_pos = position_map[equation_position]
                            ha = 'left' if 'Left' in equation_position else 'right'
                            va = 'top' if 'Top' in equation_position else 'bottom'
                            
                            bbox_props = {}
                            if equation_background:
                                bbox_props = dict(
                                    boxstyle='round',
                                    facecolor=equation_bg_color,
                                    alpha=equation_bg_alpha
                                )
                            
                            ax.text(x_pos, y_pos,
                                   "\n".join(label_parts),
                                   transform=ax.transAxes,
                                   fontsize=equation_font_size,
                                   color=equation_color,
                                   horizontalalignment=ha,
                                   verticalalignment=va,
                                   bbox=bbox_props)
        
            # Add title if enabled
            if show_title:
                ax.set_title(
                    title_text,
                    fontsize=title_size,
                    color=title_color,
                    loc=title_position,
                    pad=20
                )
            
            # Add axis labels
            ax.set_xlabel(x_column, fontsize=axis_font_size, rotation=x_rotation)
            ax.set_ylabel(y_column, fontsize=axis_font_size, rotation=y_rotation)
            
            # Set axis scales
            if x_scale == "Log":
                ax.set_xscale('log')
            elif x_scale == "Symlog":
                ax.set_xscale('symlog')
            
            if y_scale == "Log":
                ax.set_yscale('log')
            elif y_scale == "Symlog":
                ax.set_yscale('symlog')
            
            # Set axis limits if specified
            if x_min is not None:
                ax.set_xlim(left=x_min)
            if x_max is not None:
                ax.set_xlim(right=x_max)
            if y_min is not None:
                ax.set_ylim(bottom=y_min)
            if y_max is not None:
                ax.set_ylim(top=y_max)
            
            # Set tick intervals if specified
            if x_tick_interval is not None:
                ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_interval))
            if y_tick_interval is not None:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
            
            # Set number of ticks if specified
            if x_tick_count is not None:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(x_tick_count))
            if y_tick_count is not None:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(y_tick_count))
            
            # Format axis ticks
            if x_tick_format == "Scientific":
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            elif x_tick_format == "Percentage":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            elif x_tick_format == "Integer":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            elif x_tick_format == "Custom":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: x_tick_format_str.format(x)))
            
            if y_tick_format == "Scientific":
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            elif y_tick_format == "Percentage":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            elif y_tick_format == "Integer":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            elif y_tick_format == "Custom":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: y_tick_format_str.format(x)))
            
            # Rotate tick labels
            ax.tick_params(axis='x', rotation=x_tick_rotation)
            ax.tick_params(axis='y', rotation=y_tick_rotation)
            
            # Add grid if enabled
            if show_grid:
                grid_style_map = {
                    "Solid": "-",
                    "Dashed": "--",
                    "Dotted": ":",
                    "Dash-dot": "-."
                }
                ax.grid(True, alpha=grid_alpha, linestyle=grid_style_map[grid_style], color=grid_color)
                if show_minor_grid:
                    ax.grid(True, which='minor', alpha=minor_grid_alpha, linestyle=grid_style_map[grid_style], color=grid_color)
            
            # Add legend if enabled and we have multiple event types or datasets
            if show_legend and (len(selected_datasets) > 1 or any(len(dataset['data'][event_column].unique()) > 1 for dataset in [st.session_state.saved_datasets[key] for key in selected_datasets])):
                # Map legend position to bbox_to_anchor coordinates
                position_map = {
                    "Outside Right": (1.15, 0.5),
                    "Outside Left": (-0.15, 0.5),
                    "Outside Top": (0.5, 1.15),
                    "Outside Bottom": (0.5, -0.15),
                    "Upper Right": (1.0, 1.0),
                    "Upper Left": (0.0, 1.0),
                    "Lower Right": (1.0, 0.0),
                    "Lower Left": (0.0, 0.0)
                }
                
                bbox_to_anchor = position_map[legend_position]
                loc = 'center' if 'Outside' in legend_position else legend_position.lower().replace(' ', '_')
                
                legend = ax.legend(
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,
                    title=legend_title if legend_title else None,
                    fontsize=legend_font_size,
                    frameon=legend_frame,
                    framealpha=legend_alpha
                )
                if legend_title:
                    plt.setp(legend.get_title(), fontsize=legend_font_size + 2)
            
            # Adjust layout to prevent legend cutoff
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Add download button
            buf, filename = download_figure(fig)
            st.download_button(
                label="Download Scatter Plot",
                data=buf,
                file_name=filename,
                mime="image/png",
                key=f"download_scatter_{key_prefix}"
            )
        else:
            st.info("Please select datasets and configure the plot options to generate the visualization.")
            
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")
        st.exception(e)
