import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def render_pizza_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("🍕 Pizza Chart Customization")
    
    if df is not None:
        # Player selection
        if 'nickname' in df.columns:
            player_nickname = st.selectbox(
                "Select Player",
                options=df['nickname'].unique(),
                key=f"pizza_player_selector_{key_prefix}"
            )
            
            # Filter data for selected player
            player_data = df[df['nickname'] == player_nickname]
            
            # Metric selection
            st.markdown("---")
            st.subheader("Metric Selection")
            
            # Get numeric columns (excluding 'nickname' and other non-metric columns)
            numeric_columns = player_data.select_dtypes(include=['float64', 'int64']).columns
            metric_columns = [col for col in numeric_columns if col not in ['row_id', 'player_id', 'match_id', 'team_id']]
            
            # Allow selecting multiple metrics
            selected_metrics = st.multiselect(
                "Select Metrics to Display",
                options=metric_columns,
                key=f"pizza_metrics_selector_{key_prefix}"
            )
    
            if selected_metrics:
                # Create a new dataframe with just the selected metrics
                plot_data = pd.DataFrame({
                    'Metric': selected_metrics,
                    'Value': [player_data[metric].iloc[0] for metric in selected_metrics]
                })
                
                # Dataset styling
                st.markdown("---")
                st.subheader("Dataset Styling")
                
                # Color selection
                colors = []
                for metric in selected_metrics:
                    color = color_picker(
                        f"Color for {metric}", 
                        f"#{abs(hash(metric)) % 0xFFFFFF:06X}", 
                        key=f"pizza_color_{metric}"
                    )
                    colors.append(color)
                
                # Pie styling
                col1, col2 = st.columns(2)
                with col1:
                    pie_alpha = st.slider(
                        "Slice Opacity", 
                        0.0, 1.0, 0.7, 
                        key="pizza_alpha"
                    )
                with col2:
                    explode = st.slider(
                        "Explode Slices", 
                        0.0, 0.3, 0.0, 
                        key="pizza_explode"
                    )

                # Plot customization
                st.markdown("---")
                st.subheader("🎨 Plot Customization")
                
                # Title options
                show_title = st.checkbox("Show Title", True, key="pizza_show_title")
                if show_title:
                    title_text = st.text_input("Plot Title", f"{player_nickname}'s Performance Metrics", key="pizza_title_text")
                    title_size = st.slider("Title Size", 10, 30, 16, key="pizza_title_size")
                    title_color = color_picker("Title Color", "#000000", key="pizza_title_color")
                    title_position = st.selectbox(
                        "Title Position",
                        ["center", "left", "right"],
                        key="pizza_title_position"
                    )
                
                # Legend options
                show_legend = st.checkbox("Show Legend", True, key="pizza_show_legend")
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
                        key="pizza_legend_position"
                    )
                    legend_title = st.text_input("Legend Title", "Metrics", key="pizza_legend_title")
                    legend_font_size = st.slider("Legend Font Size", 8, 20, 12, key="pizza_legend_font_size")
                    legend_frame = st.checkbox("Show Legend Frame", True, key="pizza_legend_frame")
                    legend_alpha = st.slider("Legend Background Alpha", 0.0, 1.0, 0.8, key="pizza_legend_alpha")
                
                # Additional options
                st.markdown("---")
                st.subheader("Additional Options")
                show_percentages = st.checkbox("Show Percentages", True, key="pizza_show_percentages")
                if show_percentages:
                    percentage_font_size = st.slider("Percentage Font Size", 8, 20, 10, key="pizza_percentage_font_size")
                    percentage_color = color_picker("Percentage Color", "#000000", key="pizza_percentage_color")
                
                show_values = st.checkbox("Show Values", True, key="pizza_show_values")
                if show_values:
                    value_font_size = st.slider("Value Font Size", 8, 20, 10, key="pizza_value_font_size")
                    value_color = color_picker("Value Color", "#000000", key="pizza_value_color")
                
                # Background color
                bg_color = color_picker("Background Color", "#ffffff", key="pizza_bg_color")

                try:
                    # Create figure with adjusted size for legend and high DPI
                    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
                    
                    # Set background color
                    ax.set_facecolor(bg_color)
                    fig.patch.set_facecolor(bg_color)
                    
                    # Create explode array
                    explode = [explode] * len(selected_metrics)
                    
                    # Plot pie chart
                    wedges, texts, autotexts = ax.pie(
                        plot_data['Value'],
                        labels=plot_data['Metric'] if show_legend else None,
                        colors=colors,
                        autopct=lambda p: f'{p:.1f}%' if show_percentages else '',
                        explode=explode,
                        wedgeprops={'alpha': pie_alpha},
                        textprops={
                            'fontsize': percentage_font_size if show_percentages else value_font_size,
                            'color': percentage_color if show_percentages else value_color
                        }
                    )
                    
                    # Add values if requested
                    if show_values:
                        for i, (wedge, value) in enumerate(zip(wedges, plot_data['Value'])):
                            angle = (wedge.theta2 + wedge.theta1) / 2
                            x = wedge.r * 0.8 * np.cos(np.deg2rad(angle))
                            y = wedge.r * 0.8 * np.sin(np.deg2rad(angle))
                            ax.text(
                                x, y,
                                f'{value:.1f}',
                                ha='center',
                                va='center',
                                fontsize=value_font_size,
                                color=value_color
                            )
                    
                    # Add title if enabled
                    if show_title:
                        ax.set_title(
                            title_text,
                            fontsize=title_size,
                            color=title_color,
                            loc=title_position,
                            pad=20
                        )
                    
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
                    
                    # Make the plot circular
                    ax.axis('equal')
                    
                    # Adjust layout to prevent legend cutoff
                    plt.tight_layout()

                    st.pyplot(fig)

                    # Add download button
                    buf, filename = download_figure(fig)
                    st.download_button(
                        label="Download Pizza Chart",
                        data=buf,
                        file_name=filename,
                        mime="image/png",
                        key="download_pizza"
                    )
                except Exception as e:
                    st.error(f"Error creating pizza chart: {e}")
                    st.exception(e)
            else:
                st.info("Please select at least one metric to display.")
        else:
            st.error("No 'nickname' column found in the dataset. Please ensure the dataset contains player nicknames.")
    else:
        st.info("No data available for visualization.")
