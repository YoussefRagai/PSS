import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch

# Note: color_picker and download_figure are expected to be passed as arguments

def render_pitch_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("📐 Pitch Customization")

    # Initialize session state variables if they don't exist
    if f'pitch_axis_{key_prefix}' not in st.session_state:
        st.session_state[f'pitch_axis_{key_prefix}'] = False
    if f'pitch_labels_{key_prefix}' not in st.session_state:
        st.session_state[f'pitch_labels_{key_prefix}'] = False
    if f'selected_datasets_{key_prefix}' not in st.session_state:
        st.session_state[f'selected_datasets_{key_prefix}'] = []

    # Add dataset selection section
    with st.expander("📁 Dataset Selection", expanded=True):
        if 'saved_datasets' in st.session_state and st.session_state.saved_datasets:
            available_datasets = list(st.session_state.saved_datasets.keys())
            selected_datasets = st.multiselect(
                "Select Datasets to Plot",
                options=available_datasets,
                format_func=lambda x: f"{st.session_state.saved_datasets[x]['name']} ({st.session_state.saved_datasets[x]['row_count']} rows)",
                key=f"pitch_dataset_selector_{key_prefix}"
            )
            st.session_state[f'selected_datasets_{key_prefix}'] = selected_datasets
        else:
            st.info("No saved datasets available. Save some datasets first to use this feature.")
            selected_datasets = []

    # Add data plotting section
    with st.expander("📊 Data Plotting", expanded=True):
        if selected_datasets:
            # Get numeric columns from the first dataset for coordinate selection
            first_dataset = st.session_state.saved_datasets[selected_datasets[0]]['data']
            numeric_columns = first_dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                # Plot type selection
                plot_type = st.selectbox(
                    "Plot Type",
                    ["Scatter", "Lines", "Arrows"],
                    key=f"pitch_plot_type_{key_prefix}"
                )

                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("Start X Coordinate", numeric_columns, key=f"pitch_start_x_column_{key_prefix}")
                with col2:
                    y_column = st.selectbox("Start Y Coordinate", numeric_columns, key=f"pitch_start_y_column_{key_prefix}")
                
                # Add end coordinate selection for Lines and Arrows
                if plot_type in ["Lines", "Arrows"]:
                    st.markdown("---")
                    st.subheader("End Coordinates")
                    col1, col2 = st.columns(2)
                    with col1:
                        end_x_column = st.selectbox("End X Coordinate", numeric_columns, key=f"pitch_end_x_column_{key_prefix}")
                    with col2:
                        end_y_column = st.selectbox("End Y Coordinate", numeric_columns, key=f"pitch_end_y_column_{key_prefix}")
                else:
                    end_x_column = None
                    end_y_column = None
                
                # Event type selection
                event_columns = [col for col in first_dataset.columns if col not in [x_column, y_column, end_x_column if plot_type in ["Lines", "Arrows"] else None, end_y_column if plot_type in ["Lines", "Arrows"] else None]]
                event_column = st.selectbox("Event Type Column", event_columns, key=f"pitch_event_column_{key_prefix}")
                
                # Dataset-specific styling
                st.markdown("---")
                st.subheader("Dataset Styling")
                dataset_styles = {}
                
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
                                key=f"pitch_color_{dataset_key}_{event}_{key_prefix}"
                            )
                        
                        # Common styling options
                        col1, col2 = st.columns(2)
                        with col1:
                            point_size = st.slider(
                                "Point/Line Size", 
                                50, 500, 100, 
                                key=f"pitch_plot_point_size_{dataset_key}_{key_prefix}"
                            )
                        with col2:
                            point_alpha = st.slider(
                                "Opacity", 
                                0.0, 1.0, 0.7, 
                                key=f"pitch_plot_alpha_{dataset_key}_{key_prefix}"
                            )

                        # Plot-specific options
                        if plot_type == "Scatter":
                            show_markers = st.checkbox(
                                "Show Markers", 
                                True, 
                                key=f"pitch_show_markers_{dataset_key}_{key_prefix}"
                            )
                            if show_markers:
                                marker_style = st.selectbox(
                                    "Marker Style",
                                    ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+", "x", "|", "_"],
                                    key=f"pitch_marker_style_{dataset_key}_{key_prefix}"
                                )
                        elif plot_type == "Lines":
                            line_style = st.selectbox(
                                "Line Style",
                                ["-", "--", "-.", ":", "None"],
                                key=f"pitch_plot_line_style_{dataset_key}_{key_prefix}"
                            )
                            line_width = st.slider(
                                "Line Width", 
                                1, 10, 2, 
                                key=f"pitch_plot_line_width_{dataset_key}_{key_prefix}"
                            )
                            show_points = st.checkbox(
                                "Show Points at Start/End", 
                                True, 
                                key=f"pitch_show_line_points_{dataset_key}_{key_prefix}"
                            )
                        elif plot_type == "Arrows":
                            col1, col2 = st.columns(2)
                            with col1:
                                arrow_length = st.slider(
                                    "Arrow Length", 
                                    1, 10, 3, 
                                    key=f"pitch_plot_arrow_length_{dataset_key}_{key_prefix}"
                                )
                            with col2:
                                arrow_width = st.slider(
                                    "Arrow Width", 
                                    1, 5, 2, 
                                    key=f"pitch_plot_arrow_width_{dataset_key}_{key_prefix}"
                                )
                            show_points = st.checkbox(
                                "Show Points at Start/End", 
                                True, 
                                key=f"pitch_show_arrow_points_{dataset_key}_{key_prefix}"
                            )
                        dataset_styles[dataset_key] = {
                            'event_colors': event_colors,
                            'point_size': point_size,
                            'point_alpha': point_alpha,
                            'show_markers': show_markers if plot_type == "Scatter" else None,
                            'marker_style': marker_style if plot_type == "Scatter" and show_markers else None,
                            'line_style': line_style if plot_type == "Lines" else None,
                            'line_width': line_width if plot_type == "Lines" else None,
                            'show_points': show_points if plot_type in ["Lines", "Arrows"] else None,
                            'arrow_length': arrow_length if plot_type == "Arrows" else None,
                            'arrow_width': arrow_width if plot_type == "Arrows" else None
                        }
            else:
                st.warning("Need at least 2 numeric columns for coordinates")
        else:
            st.info("Select at least one dataset to plot.")

    with st.expander("🟩 Pitch Layout"):
        pitch_type = st.selectbox("Pitch Type", ["statsbomb", "opta", "tracab", "wyscout", "custom"], key=f"pitch_layout_type_{key_prefix}")
        
        pitch_length = 105.0
        pitch_width = 68.0
        penalty_box_length = 16.5
        penalty_box_width = 40.3
        
        if pitch_type == "custom":
            col1, col2 = st.columns(2)
            with col1:
                pitch_length = st.number_input("Pitch Length", value=pitch_length, min_value=90.0, max_value=120.0, key=f"pitch_layout_length_{key_prefix}")
                pitch_width = st.number_input("Pitch Width", value=pitch_width, min_value=45.0, max_value=90.0, key=f"pitch_layout_width_{key_prefix}")
            with col2:
                penalty_box_length = st.number_input("Penalty Box Length", value=penalty_box_length, min_value=10.0, max_value=20.0, key=f"pitch_layout_penalty_length_{key_prefix}")
                penalty_box_width = st.number_input("Penalty Box Width", value=penalty_box_width, min_value=30.0, max_value=50.0, key=f"pitch_layout_penalty_width_{key_prefix}")
                
        orientation = st.selectbox("Orientation", ["horizontal", "vertical"], key=f"pitch_layout_orientation_{key_prefix}")
        show_full_pitch = st.radio("Pitch Display", ["Full Pitch", "Half Pitch"], index=0, key=f"pitch_layout_display_{key_prefix}")
        
        half_selection = None 
        if show_full_pitch == "Half Pitch":
            if orientation == "horizontal":
                half_options = ["left", "right"]
                half_selection = st.selectbox("Which half to show", half_options, key=f"pitch_layout_half_h_{key_prefix}")
            else:
                half_options = ["top", "bottom"]
                half_selection = st.selectbox("Which half to show", half_options, key=f"pitch_layout_half_v_{key_prefix}")
        
        goal_type = st.selectbox("Goal Type", ["box", "circle", "line", "none"], key=f"pitch_layout_goal_type_{key_prefix}")
        
        col1, col2 = st.columns(2)
        with col1:
            pad_left = st.slider("pad_left", 0.0, 5.0, 0.5, key=f"pitch_layout_pad_left_{key_prefix}")
            pad_right = st.slider("pad_right", 0.0, 5.0, 0.5, key=f"pitch_layout_pad_right_{key_prefix}")
        with col2:
            pad_top = st.slider("pad_top", 0.0, 5.0, 0.5, key=f"pitch_layout_pad_top_{key_prefix}")
            pad_bottom = st.slider("pad_bottom", 0.0, 5.0, 0.5, key=f"pitch_layout_pad_bottom_{key_prefix}")

    with st.expander("🎨 Style", expanded=False): # Start collapsed
        col1, col2 = st.columns(2)
        with col1:
            pitch_color = color_picker("Pitch Color", "#aabb97", key=f"pitch_style_color_{key_prefix}")
            line_color = color_picker("Line Color", "#ffffff", key=f"pitch_style_line_color_{key_prefix}")
        with col2:
            line_thickness = st.slider("Line Width", 0.5, 5.0, 2.0, key=f"pitch_style_line_width_{key_prefix}")
            line_alpha = st.slider("Line Alpha", 0.0, 1.0, 1.0, key=f"pitch_style_line_alpha_{key_prefix}")
            line_zorder = st.slider("Line Z-Order", 0, 10, 1, key=f"pitch_style_line_zorder_{key_prefix}")
            goal_alpha = st.slider("Goal Alpha", 0.0, 1.0, 1.0, key=f"pitch_style_goal_alpha_{key_prefix}")

    with st.expander("📐 Axes & Labels", expanded=False): # Start collapsed
        axis = st.checkbox("Show Axis", st.session_state.get(f'pitch_axis_{key_prefix}', False), key=f"pitch_axes_show_{key_prefix}")
        label = st.checkbox("Show Half Labels", st.session_state.get(f'pitch_labels_{key_prefix}', False), key=f"pitch_axes_labels_{key_prefix}")
        
        # Add title and legend customization
        st.markdown("---")
        st.subheader("Title & Legend")
        
        # Title options
        show_title = st.checkbox("Show Title", True, key=f"pitch_show_title_{key_prefix}")
        if show_title:
            title_text = st.text_input("Plot Title", "Pitch Visualization", key=f"pitch_title_text_{key_prefix}")
            title_size = st.slider("Title Size", 10, 30, 16, key=f"pitch_title_size_{key_prefix}")
            title_color = color_picker("Title Color", "#000000", key=f"pitch_title_color_{key_prefix}")
            title_position = st.selectbox(
                "Title Position",
                ["center", "left", "right"],
                key=f"pitch_title_position_{key_prefix}"
            )
        
        # Legend options
        show_legend = st.checkbox("Show Legend", True, key=f"pitch_show_legend_{key_prefix}")
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
                key=f"pitch_legend_position_{key_prefix}"
            )
            legend_title = st.text_input("Legend Title", "", key=f"pitch_legend_title_{key_prefix}")
            legend_font_size = st.slider("Legend Font Size", 8, 20, 12, key=f"pitch_legend_font_size_{key_prefix}")
            legend_frame = st.checkbox("Show Legend Frame", True, key=f"pitch_legend_frame_{key_prefix}")
            legend_alpha = st.slider("Legend Background Alpha", 0.0, 1.0, 0.8, key=f"pitch_legend_alpha_{key_prefix}")

    try:
        pitch_class = VerticalPitch if orientation == 'vertical' else Pitch
        half = show_full_pitch == "Half Pitch"
        
        custom_params = {}
        if pitch_type == "custom":
             if 'pitch_length' not in locals() or 'pitch_width' not in locals() or \
                'penalty_box_length' not in locals() or 'penalty_box_width' not in locals():
                 st.warning("Please define custom pitch dimensions in the layout section.")
                 st.stop()
             custom_params = {
                 'pitch_length': pitch_length, 'pitch_width': pitch_width,
                 'penalty_box_length': penalty_box_length, 'penalty_box_width': penalty_box_width
             }

        # Create pitch instance
        pitch = pitch_class(
            pitch_type=pitch_type if pitch_type != "custom" else None, 
            pitch_length=custom_params.get('pitch_length'),
            pitch_width=custom_params.get('pitch_width'),
            half=half,
            goal_type=goal_type,
            pad_left=pad_left, pad_right=pad_right,
            pad_top=pad_top, pad_bottom=pad_bottom,
            line_color=line_color,
            line_zorder=line_zorder,
            line_alpha=line_alpha,
            linewidth=line_thickness,
            pitch_color=pitch_color,
            goal_alpha=goal_alpha,
            axis=axis,
            label=label
        )

        # Draw pitch with adjusted figure size to accommodate legend
        fig, ax = plt.subplots(figsize=(12, 7))  # Increased width for legend
        pitch.draw(ax=ax)
        
        # Add title if enabled
        if show_title:
            ax.set_title(
                title_text,
                fontsize=title_size,
                color=title_color,
                loc=title_position,
                pad=20
            )
        
        # Adjust view for half pitch
        if half and half_selection:
            if orientation == "horizontal":
                if half_selection == "right":
                    ax.set_xlim(pitch_length/2, pitch_length)
                else:  # left
                    ax.set_xlim(0, pitch_length/2)
            else:  # vertical
                if half_selection == "top":
                    ax.set_ylim(pitch_width/2, pitch_width)
                else:  # bottom
                    ax.set_ylim(0, pitch_width/2)

        # Plot data from all selected datasets
        if selected_datasets and 'x_column' in locals() and 'y_column' in locals():
            try:
                for dataset_key in selected_datasets:
                    dataset = st.session_state.saved_datasets[dataset_key]
                    styles = dataset_styles[dataset_key]
                    
                    # Plot based on selected type
                    for event in dataset['data'][event_column].unique():
                        event_data = dataset['data'][dataset['data'][event_column] == event]
                        if not event_data.empty:
                            if plot_type == "Scatter":
                                pitch.scatter(
                                    event_data[x_column], 
                                    event_data[y_column],
                                    ax=ax,
                                    color=styles['event_colors'][event],
                                    s=styles['point_size'],
                                    alpha=styles['point_alpha'],
                                    label=f"{event} ({dataset['name']})",
                                    marker=styles['marker_style'] if styles['show_markers'] else None
                                )
                            
                            elif plot_type == "Lines":
                                pitch.lines(
                                    event_data[x_column],
                                    event_data[y_column],
                                    event_data[end_x_column],
                                    event_data[end_y_column],
                                    ax=ax,
                                    color=styles['event_colors'][event],
                                    alpha=styles['point_alpha'],
                                    label=f"{event} ({dataset['name']})",
                                    linestyle=styles['line_style'],
                                    linewidth=styles['line_width']
                                )
                                if styles['show_points']:
                                    pitch.scatter(
                                        event_data[x_column],
                                        event_data[y_column],
                                        ax=ax,
                                        color=styles['event_colors'][event],
                                        s=styles['point_size']/2,
                                        alpha=styles['point_alpha'],
                                        marker='o'
                                    )
                                    pitch.scatter(
                                        event_data[end_x_column],
                                        event_data[end_y_column],
                                        ax=ax,
                                        color=styles['event_colors'][event],
                                        s=styles['point_size']/2,
                                        alpha=styles['point_alpha'],
                                        marker='o'
                                    )
                            
                            elif plot_type == "Arrows":
                                pitch.arrows(
                                    event_data[x_column],
                                    event_data[y_column],
                                    event_data[end_x_column],
                                    event_data[end_y_column],
                                    ax=ax,
                                    color=styles['event_colors'][event],
                                    width=styles['arrow_width'],
                                    headwidth=styles['arrow_length'],
                                    alpha=styles['point_alpha'],
                                    label=f"{event} ({dataset['name']})"
                                )
                                if styles['show_points']:
                                    pitch.scatter(
                                        event_data[x_column],
                                        event_data[y_column],
                                        ax=ax,
                                        color=styles['event_colors'][event],
                                        s=styles['point_size']/2,
                                        alpha=styles['point_alpha'],
                                        marker='o'
                                    )
                                    pitch.scatter(
                                        event_data[end_x_column],
                                        event_data[end_y_column],
                                        ax=ax,
                                        color=styles['event_colors'][event],
                                        s=styles['point_size']/2,
                                        alpha=styles['point_alpha'],
                                        marker='o'
                                    )
                
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
                    
            except Exception as e:
                st.error(f"Error plotting data: {e}")
                st.exception(e)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Add download button
        buf, filename = download_figure(fig)
        st.download_button(
            label="Download Pitch Image",
            data=buf,
            file_name=filename,
            mime="image/png",
            key=f"download_pitch_{key_prefix}"
        )
        
    except NameError as ne:
         st.error(f"A configuration error occurred: {ne}. Try selecting a standard pitch type first or ensure custom dimensions are set.")
    except Exception as e:
        st.error(f"Error creating pitch visualization: {e}")
        st.exception(e)
