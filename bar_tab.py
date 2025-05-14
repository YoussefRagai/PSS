import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Note: color_picker and download_figure are expected to be passed as arguments

def render_bar_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("📊 Bar Chart Customization")
    
    # Initialize session state variables if they don't exist
    if 'selected_datasets' not in st.session_state:
        st.session_state.selected_datasets = []

    # Add dataset selection section
    with st.expander("📁 Dataset Selection", expanded=True):
        if 'saved_datasets' in st.session_state and st.session_state.saved_datasets:
            available_datasets = list(st.session_state.saved_datasets.keys())
            selected_datasets = st.multiselect(
                "Select Datasets to Plot",
                options=available_datasets,
                format_func=lambda x: f"{st.session_state.saved_datasets[x]['name']} ({st.session_state.saved_datasets[x]['row_count']} rows)",
                key=f"bar_dataset_selector_{key_prefix}"
            )
            st.session_state.selected_datasets = selected_datasets
        else:
            st.info("No saved datasets available. Save some datasets first to use this feature.")
            selected_datasets = []

    # Add data plotting section
    with st.expander("📊 Data Plotting", expanded=True):
        if selected_datasets:
            # Get columns from the first dataset
            first_dataset = st.session_state.saved_datasets[selected_datasets[0]]['data']
            
            # Column selection
            col1, col2 = st.columns(2)
            with col1:
                category_column = st.selectbox("Category Column", first_dataset.columns, key=f"bar_category_column_{key_prefix}")
            with col2:
                value_column = st.selectbox("Value Column", first_dataset.select_dtypes(include=['float64', 'int64']).columns, key=f"bar_value_column_{key_prefix}")
            
            # Dataset-specific styling
            st.markdown("---")
            st.subheader("Dataset Styling")
            dataset_styles = {}
            
            # Create tabs for each dataset's styling
            dataset_tabs = st.tabs([f"Style for {st.session_state.saved_datasets[key]['name']}" for key in selected_datasets])
            
            for idx, dataset_key in enumerate(selected_datasets):
                dataset = st.session_state.saved_datasets[dataset_key]
                with dataset_tabs[idx]:
                    # Color selection
                    bar_color = color_picker(
                        "Bar Color", 
                        "#1f77b4", 
                        key=f"bar_color_{dataset_key}_{key_prefix}"
                    )
                    
                    # Bar styling
                    col1, col2 = st.columns(2)
                    with col1:
                        bar_alpha = st.slider(
                            "Bar Opacity", 
                            0.0, 1.0, 0.7, 
                            key=f"bar_alpha_{dataset_key}_{key_prefix}"
                        )
                    with col2:
                        bar_width = st.slider(
                            "Bar Width", 
                            0.1, 1.0, 0.8, 
                            key=f"bar_width_{dataset_key}_{key_prefix}"
                        )
                    
                    dataset_styles[dataset_key] = {
                        'color': bar_color,
                        'alpha': bar_alpha,
                        'width': bar_width
                    }
        else:
            st.info("Select at least one dataset to plot.")

    # Add plot customization section
    with st.expander("🎨 Plot Customization", expanded=False):
        # Title options
        show_title = st.checkbox("Show Title", True, key=f"bar_show_title_{key_prefix}")
        if show_title:
            title_text = st.text_input("Plot Title", "Bar Chart", key=f"bar_title_text_{key_prefix}")
            title_size = st.slider("Title Size", 10, 30, 16, key=f"bar_title_size_{key_prefix}")
            title_color = color_picker("Title Color", "#000000", key=f"bar_title_color_{key_prefix}")
            title_position = st.selectbox(
                "Title Position",
                ["center", "left", "right"],
                key=f"bar_title_position_{key_prefix}"
            )
        
        # Legend options
        show_legend = st.checkbox("Show Legend", True, key=f"bar_show_legend_{key_prefix}")
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
                key=f"bar_legend_position_{key_prefix}"
            )
            legend_title = st.text_input("Legend Title", "", key=f"bar_legend_title_{key_prefix}")
            legend_font_size = st.slider("Legend Font Size", 8, 20, 12, key=f"bar_legend_font_size_{key_prefix}")
            legend_frame = st.checkbox("Show Legend Frame", True, key=f"bar_legend_frame_{key_prefix}")
            legend_alpha = st.slider("Legend Background Alpha", 0.0, 1.0, 0.8, key=f"bar_legend_alpha_{key_prefix}")
        
        # Axis options
        st.markdown("---")
        st.subheader("Axis Customization")
        show_grid = st.checkbox("Show Grid", True, key=f"bar_show_grid_{key_prefix}")
        grid_alpha = st.slider("Grid Opacity", 0.0, 1.0, 0.2, key=f"bar_grid_alpha_{key_prefix}")
        axis_font_size = st.slider("Axis Label Font Size", 8, 20, 12, key=f"bar_axis_font_size_{key_prefix}")
        
        # Additional axis options
        col1, col2 = st.columns(2)
        with col1:
            x_rotation = st.slider("X-axis Label Rotation", 0, 90, 0, key=f"bar_x_rotation_{key_prefix}")
            x_tick_format = st.selectbox(
                "X-axis Tick Format",
                ["Default", "Scientific", "Percentage", "Integer"],
                key=f"bar_x_tick_format_{key_prefix}"
            )
        with col2:
            y_rotation = st.slider("Y-axis Label Rotation", 0, 90, 0, key=f"bar_y_rotation_{key_prefix}")
            y_tick_format = st.selectbox(
                "Y-axis Tick Format",
                ["Default", "Scientific", "Percentage", "Integer"],
                key=f"bar_y_tick_format_{key_prefix}"
            )
        
        # Bar orientation
        is_horizontal = st.checkbox("Horizontal Bars", False, key=f"bar_horizontal_{key_prefix}")
        
        # Background color
        bg_color = color_picker("Background Color", "#ffffff", key=f"bar_bg_color_{key_prefix}")

    try:
        # Only create the plot if we have the necessary data
        if selected_datasets and 'category_column' in locals() and 'value_column' in locals():
            # Create figure with adjusted size for legend
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Set background color
            ax.set_facecolor(bg_color)
            fig.patch.set_facecolor(bg_color)
            
            # Prepare data
            datasets = []
            for dataset_key in selected_datasets:
                dataset = st.session_state.saved_datasets[dataset_key]
                styles = dataset_styles[dataset_key]
                
                # Group by category and calculate values
                grouped_data = dataset['data'].groupby(category_column)[value_column].mean()
                
                datasets.append({
                    'name': dataset['name'],
                    'categories': grouped_data.index.tolist(),
                    'values': grouped_data.values.tolist(),
                    'styles': styles
                })
            
            # Get all unique categories
            all_categories = sorted(set(cat for d in datasets for cat in d['categories']))
            
            # Set up x positions for grouped bars
            x = np.arange(len(all_categories))
            bar_width = min(0.8, 0.8 / len(datasets))  # Adjust bar width based on number of datasets
            
            # Plot bars for each dataset
            for i, dataset in enumerate(datasets):
                # Map categories to x positions
                cat_to_pos = {cat: pos for pos, cat in enumerate(all_categories)}
                x_positions = [cat_to_pos[cat] for cat in dataset['categories']]
                
                # Calculate bar positions
                offset = (i - len(datasets) / 2 + 0.5) * bar_width
                x_pos = x[x_positions] + offset
                
                # Choose the appropriate bar function and parameters
                bar_func = ax.barh if is_horizontal else ax.bar
                bar_params = {
                    'x': x_pos if not is_horizontal else dataset['values'],
                    'height' if is_horizontal else 'width': bar_width,
                    'color': dataset['styles']['color'],
                    'alpha': dataset['styles']['alpha'],
                    'label': dataset['name']
                }
                
                # Add the other parameter based on orientation
                if is_horizontal:
                    bar_params['y'] = x_pos
                else:
                    bar_params['y'] = dataset['values']
                
                # Create the bars
                bars = bar_func(**bar_params)
            
            # Set axis labels
            if is_horizontal:
                ax.set_xlabel(value_column, fontsize=axis_font_size, rotation=x_rotation)
                ax.set_ylabel(category_column, fontsize=axis_font_size, rotation=y_rotation)
            else:
                ax.set_xlabel(category_column, fontsize=axis_font_size, rotation=x_rotation)
                ax.set_ylabel(value_column, fontsize=axis_font_size, rotation=y_rotation)
            
            # Set x-axis ticks and labels
            if is_horizontal:
                ax.set_yticks(x)
                ax.set_yticklabels(all_categories)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(all_categories)
            
            # Format axis ticks
            if x_tick_format == "Scientific":
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            elif x_tick_format == "Percentage":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            elif x_tick_format == "Integer":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
                
            if y_tick_format == "Scientific":
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            elif y_tick_format == "Percentage":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
            elif y_tick_format == "Integer":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
            
            # Add title if enabled
            if show_title:
                ax.set_title(
                    title_text,
                    fontsize=title_size,
                    color=title_color,
                    loc=title_position,
                    pad=20
                )
            
            # Add grid if enabled
            if show_grid:
                ax.grid(True, alpha=grid_alpha)
            
            # Add legend if enabled
            if show_legend:
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
                label="Download Bar Chart",
                data=buf,
                file_name=filename,
                mime="image/png",
                key=f"download_bar_{key_prefix}"
            )
        else:
            st.info("Please select datasets and configure the plot options to generate the visualization.")
            
    except Exception as e:
        st.error(f"Error creating bar chart: {e}")
        st.exception(e)
