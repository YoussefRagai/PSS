import streamlit as st
import pandas as pd
import datetime

# Note: color_picker is expected to be passed as an argument

# Function to convert DF to CSV (moved from main script)
@st.cache_data
def convert_df_to_csv(df_to_convert, selected_columns):
    # Ensure only selected columns are exported
    cols_to_export = [col for col in selected_columns if col in df_to_convert.columns]
    if not cols_to_export:
        # Handle case where selected_columns might be empty or invalid
        cols_to_export = list(df_to_convert.columns)
    return df_to_convert[cols_to_export].to_csv(index=False).encode('utf-8')

def render_table_tab(st, df=None, color_picker=None, download_figure=None, key_prefix=""):
    st.header("📄 Table Customization & Filtering")

    # --- Initialize Table-specific session state --- 
    if 'table_filter_conditions' not in st.session_state:
        st.session_state.table_filter_conditions = []
    if 'table_sort_col' not in st.session_state:
        st.session_state.table_sort_col = "None"
    if 'table_sort_order' not in st.session_state:
        st.session_state.table_sort_order = "Ascending"
    if 'table_secondary_sort' not in st.session_state:
        st.session_state.table_secondary_sort = "None"
    if 'table_secondary_order' not in st.session_state:
        st.session_state.table_secondary_order = "Ascending"
    # appearance_settings, rows_per_page, current_page, selected_columns are initialized earlier/globally

    # --- Check if data is loaded --- 
    if 'original_df' not in st.session_state or st.session_state.original_df is None or st.session_state.original_df.empty:
        st.warning("Please upload a CSV file using the uploader at the top of the page to use the Table tab.")
        return # Stop rendering if no data
    
    df = st.session_state.original_df 
    all_columns = list(df.columns)

    # --- Configuration Expanders --- 
    with st.expander("🎨 Appearance Settings", expanded=False):
        with st.form("appearance_form"):
            settings = st.session_state.appearance_settings
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Show Index", settings.get('show_index', False), key="table_show_index_cb")
                st.number_input("Numeric Precision (for export)", min_value=0, max_value=10, value=settings.get('precision', 2), key="table_precision_num")
            with col2:
                st.number_input("Table Height (px)", min_value=100, value=settings.get('table_height', 400), key="table_height_px")
            
            submitted = st.form_submit_button("Apply Appearance")
            if submitted:
                st.session_state.appearance_settings['show_index'] = st.session_state.table_show_index_cb
                st.session_state.appearance_settings['precision'] = st.session_state.table_precision_num
                st.session_state.appearance_settings['table_height'] = st.session_state.table_height_px
                # Clear removed settings just in case (already done in main file, but safe here too)
                st.session_state.appearance_settings.pop('header_bg_color', None)
                st.session_state.appearance_settings.pop('header_text_color', None)
                st.session_state.appearance_settings.pop('odd_row_color', None)
                st.session_state.appearance_settings.pop('even_row_color', None)
                st.session_state.appearance_settings.pop('text_color', None)
                st.session_state.appearance_settings.pop('highlight_color', None)
                st.rerun()

    with st.expander("📊 Column Selection", expanded=True):
         if 'selected_columns' not in st.session_state or not st.session_state.selected_columns or not all(c in all_columns for c in st.session_state.selected_columns):
             st.session_state.selected_columns = all_columns
         
         def update_selected_columns():
             st.session_state.selected_columns = st.session_state.table_column_select_widget
             
         st.multiselect(
             "Select columns to display (updates immediately)",
             options=all_columns,
             default=st.session_state.selected_columns,
             key="table_column_select_widget",
             on_change=update_selected_columns
         )

    # --- Filtering, Sorting, and Pagination Form --- 
    with st.form(f"table_filter_form_{key_prefix}"):
        st.subheader("Filtering, Sorting & Pagination")
        
        st.markdown("**Filters**")
        filter_cols = st.columns(3) 
        add_filter_pressed = filter_cols[0].form_submit_button("Add Filter")
        remove_filter_pressed = filter_cols[1].form_submit_button("Remove Last Filter", disabled=not st.session_state.table_filter_conditions)
        
        form_filter_conditions = []
        for i, filt_state in enumerate(st.session_state.table_filter_conditions):
             st.markdown(f"_Filter {i+1}_")
             cols = st.columns([2, 2, 3, 1])
             current_filter_config = filt_state.copy()
             
             selected_column = cols[0].selectbox(
                  f"Column##{i}", all_columns, 
                  index=all_columns.index(current_filter_config.get('column')) if current_filter_config.get('column') in all_columns else 0,
                  key=f"filter_col_{i}_{key_prefix}"
             )
             current_filter_config['column'] = selected_column
             
             data_type = 'text' 
             filter_types = ["Equals", "Contains", "Starts with", "Ends with", "In list", "Not equal to"]
             try:
                 col_to_check = df[selected_column].dropna()
                 if not col_to_check.empty:
                     if pd.api.types.is_numeric_dtype(col_to_check):
                         data_type = 'numeric'
                         filter_types = ["Equal to", "Not equal to", "Greater than", "Less than", "Between"]
                     elif pd.api.types.is_datetime64_any_dtype(col_to_check) or pd.api.types.is_timedelta64_dtype(col_to_check):
                         if pd.to_datetime(col_to_check, errors='coerce').notna().any():
                             data_type = 'datetime'
                             filter_types = ["Equal to", "Not equal to", "After", "Before", "Between"]
             except KeyError:
                  st.warning(f"Column '{selected_column}' not found during type check.")
             
             current_filter_config['data_type'] = data_type
             
             valid_type = current_filter_config.get('type') in filter_types
             current_type_index = filter_types.index(current_filter_config.get('type')) if valid_type else 0
             if not valid_type:
                  current_filter_config['type'] = filter_types[0]
                  current_filter_config['value'] = '' 
                  if data_type == 'numeric': current_filter_config['value'] = 0.0
                  if data_type == 'datetime': current_filter_config['value'] = datetime.date.today()
                  current_type_index = 0
                  
             selected_type = cols[1].selectbox(f"Condition##{i}", filter_types, index=current_type_index, key=f"filter_type_{i}_{key_prefix}")
             current_filter_config['type'] = selected_type
             
             current_value = current_filter_config.get('value')
             
             # --- Value Input Widget Logic --- 
             if selected_type == "Between":
                 if data_type == 'numeric':
                      default_val = current_value if isinstance(current_value, list) and len(current_value) == 2 else [0.0, 0.0]
                      val1 = cols[2].number_input(f"Min##{i}", value=float(default_val[0]), key=f"filter_val_min_{i}_{key_prefix}")
                      val2 = cols[3].number_input(f"Max##{i}", value=float(default_val[1]), key=f"filter_val_max_{i}_{key_prefix}")
                      current_filter_config['value'] = [val1, val2]
                 elif data_type == 'datetime':
                      default_val = current_value if isinstance(current_value, list) and len(current_value) == 2 else [datetime.date.today(), datetime.date.today()]
                      try: 
                          date_val1 = default_val[0] if isinstance(default_val[0], datetime.date) else pd.to_datetime(default_val[0]).date()
                          date_val2 = default_val[1] if isinstance(default_val[1], datetime.date) else pd.to_datetime(default_val[1]).date()
                      except:
                          date_val1, date_val2 = datetime.date.today(), datetime.date.today()
                      val1 = cols[2].date_input(f"Start Date##{i}", value=date_val1, key=f"filter_val_start_{i}_{key_prefix}")
                      val2 = cols[3].date_input(f"End Date##{i}", value=date_val2, key=f"filter_val_end_{i}_{key_prefix}")
                      current_filter_config['value'] = [val1, val2]
                 else: 
                     current_filter_config['value'] = cols[2].text_input(f"Value##{i}", value=str(current_value if current_value is not None else ''), key=f"filter_val_{i}_{key_prefix}")
             elif selected_type == "In list" and data_type == 'text':
                  default_val_str = ",".join(map(str, current_value)) if isinstance(current_value, list) else str(current_value if current_value is not None else '')
                  current_filter_config['value'] = cols[2].text_area(f"Values (comma-separated)##{i}", value=default_val_str, key=f"filter_val_{i}_{key_prefix}")
             elif data_type == 'numeric':
                  safe_numeric_value = 0.0
                  if current_value is not None:
                      try: safe_numeric_value = float(current_value)
                      except (ValueError, TypeError): pass # Keep default 0.0
                  current_filter_config['value'] = cols[2].number_input(f"Value##{i}", value=safe_numeric_value, key=f"filter_val_{i}_{key_prefix}")
             elif data_type == 'datetime':
                  try: default_date = current_value if isinstance(current_value, datetime.date) else pd.to_datetime(current_value).date()
                  except: default_date = datetime.date.today()
                  current_filter_config['value'] = cols[2].date_input(f"Date##{i}", value=default_date, key=f"filter_val_{i}_{key_prefix}")
             elif data_type == 'text': 
                  current_filter_config['value'] = cols[2].text_input(f"Value##{i}", value=str(current_value if current_value is not None else ''), key=f"filter_val_{i}_{key_prefix}")
             else: 
                  current_filter_config['value'] = cols[2].text_input(f"Value (fallback)##{i}", value=str(current_value if current_value is not None else ''), key=f"filter_val_{i}_{key_prefix}")
             # --- End Value Input Widget Logic ---

             form_filter_conditions.append(current_filter_config)

        st.markdown("--- **Sorting**")
        sort_options = ["None"] + all_columns
        sort_col1, sort_col2, sort_col3, sort_col4 = st.columns(4)
        primary_sort_col_widget = sort_col1.selectbox("Primary Sort Column", sort_options, index=sort_options.index(st.session_state.table_sort_col) if st.session_state.table_sort_col in sort_options else 0, key=f"table_primary_sort_col_widget_{key_prefix}")
        primary_sort_order_widget = sort_col2.selectbox("Order", ["Ascending", "Descending"], index=0 if st.session_state.table_sort_order == "Ascending" else 1, key=f"table_primary_sort_order_widget_{key_prefix}")
        secondary_sort_col_widget = sort_col3.selectbox("Secondary Sort Column", sort_options, index=sort_options.index(st.session_state.table_secondary_sort) if st.session_state.table_secondary_sort in sort_options else 0, key=f"table_secondary_sort_col_widget_{key_prefix}")
        secondary_sort_order_widget = sort_col4.selectbox("Order", ["Ascending", "Descending"], index=0 if st.session_state.table_secondary_order == "Ascending" else 1, key=f"table_secondary_sort_order_widget_{key_prefix}")

        st.markdown("--- **Pagination**")
        rows_per_page_widget = st.number_input("Rows Per Page", min_value=5, max_value=200, value=st.session_state.rows_per_page, step=5, key=f"table_rows_per_page_widget_{key_prefix}")

        st.markdown("--- ")
        apply_button = st.form_submit_button("Apply All Changes")

    # --- Form Processing (Outside Form) ---
    if add_filter_pressed:
        default_col = all_columns[0]
        default_value = '' 
        try:
            col_data = df[default_col].dropna()
            if not col_data.empty:
                if pd.api.types.is_numeric_dtype(col_data): default_value = 0.0
                elif pd.api.types.is_datetime64_any_dtype(col_data) or pd.api.types.is_timedelta64_dtype(col_data):
                    if pd.to_datetime(col_data, errors='coerce').notna().any(): default_value = datetime.date.today()
        except Exception: pass
        st.session_state.table_filter_conditions.append({'column': default_col, 'type': 'Equals', 'value': default_value})
        st.rerun()
        
    if remove_filter_pressed:
        if st.session_state.table_filter_conditions: st.session_state.table_filter_conditions.pop()
        st.rerun()
    
    if apply_button:
        st.session_state.table_filter_conditions = form_filter_conditions 
        st.session_state.table_sort_col = primary_sort_col_widget
        st.session_state.table_sort_order = primary_sort_order_widget
        st.session_state.table_secondary_sort = secondary_sort_col_widget
        st.session_state.table_secondary_order = secondary_sort_order_widget
        st.session_state.rows_per_page = rows_per_page_widget
        st.session_state.current_page = 1
        st.session_state.excel_ready = False # Reset export flag
        st.rerun() 

    # --- Apply Filtering & Sorting based on current session state --- 
    processed_df = df.copy()
    active_filters = st.session_state.table_filter_conditions
    
    # Apply Filters
    for filter_config in active_filters:
         col = filter_config['column']
         filter_type = filter_config['type']
         value = filter_config['value']
         data_type = filter_config.get('data_type', 'text')
         if col not in processed_df.columns: continue
         try: 
             if data_type == 'numeric':
                 numeric_col = pd.to_numeric(processed_df[col], errors='coerce')
                 if filter_type == "Equal to": processed_df = processed_df[numeric_col == value]
                 elif filter_type == "Not equal to": processed_df = processed_df[numeric_col != value]
                 elif filter_type == "Greater than": processed_df = processed_df[numeric_col > value]
                 elif filter_type == "Less than": processed_df = processed_df[numeric_col < value]
                 elif filter_type == "Between": processed_df = processed_df[numeric_col.between(value[0], value[1], inclusive='both')]
             elif data_type == 'datetime':
                 datetime_col = pd.to_datetime(processed_df[col], errors='coerce').dt.date
                 if isinstance(value, list):
                     comp_val1 = value[0] if isinstance(value[0], datetime.date) else pd.to_datetime(value[0]).date()
                     comp_val2 = value[1] if isinstance(value[1], datetime.date) else pd.to_datetime(value[1]).date()
                     if filter_type == "Between": processed_df = processed_df[datetime_col.between(comp_val1, comp_val2, inclusive='both')]
                 else:
                     comp_val = value if isinstance(value, datetime.date) else pd.to_datetime(value).date()
                     if filter_type == "Equal to": processed_df = processed_df[datetime_col == comp_val]
                     elif filter_type == "Not equal to": processed_df = processed_df[datetime_col != comp_val]
                     elif filter_type == "After": processed_df = processed_df[datetime_col > comp_val]
                     elif filter_type == "Before": processed_df = processed_df[datetime_col < comp_val]
             elif data_type == 'text':
                 text_col = processed_df[col].astype(str).str.lower()
                 value_str = str(value).lower()
                 if filter_type == "Equals": processed_df = processed_df[text_col == value_str]
                 elif filter_type == "Contains": processed_df = processed_df[text_col.str.contains(value_str, na=False)]
                 elif filter_type == "Starts with": processed_df = processed_df[text_col.str.startswith(value_str, na=False)]
                 elif filter_type == "Ends with": processed_df = processed_df[text_col.str.endswith(value_str, na=False)]
                 elif filter_type == "In list":
                     value_list = [v.strip().lower() for v in str(value).split(',') if v.strip()] 
                     processed_df = processed_df[text_col.isin(value_list)]
         except Exception as filter_apply_err: pass # Avoid spamming warnings on rerun
    
    # Apply Sorting
    try:
        sort_columns, sort_ascending = [], []
        if st.session_state.table_sort_col != "None" and st.session_state.table_sort_col in processed_df.columns:
            sort_columns.append(st.session_state.table_sort_col)
            sort_ascending.append(st.session_state.table_sort_order == "Ascending")
        if st.session_state.table_secondary_sort != "None" and st.session_state.table_secondary_sort in processed_df.columns and st.session_state.table_secondary_sort != st.session_state.table_sort_col:
            sort_columns.append(st.session_state.table_secondary_sort)
            sort_ascending.append(st.session_state.table_secondary_order == "Ascending")
        if sort_columns: processed_df = processed_df.sort_values(by=sort_columns, ascending=sort_ascending, na_position='last')
    except Exception as sort_err: pass # Avoid spamming warnings
    
    st.session_state.filtered_df = processed_df # Update state with final processed data

    # --- Display Section --- 
    st.subheader("Filtered & Sorted Data")
    total_rows = len(processed_df)
    rows_per_page = st.session_state.rows_per_page
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page) # Ceiling division

    current_page = st.session_state.current_page
    if current_page > total_pages: current_page = total_pages
    if current_page < 1: current_page = 1
    st.session_state.current_page = current_page # Update state if corrected

    st.markdown(f"Total Rows: {total_rows}")
    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
         if st.button("◀ Previous", disabled=(current_page <= 1), key="prev_page_btn"):
              st.session_state.current_page -= 1
              st.rerun()
    with page_col2:
         st.write(f"Page {current_page} of {total_pages}")
    with page_col3:
         if st.button("Next ▶", disabled=(current_page >= total_pages), key="next_page_btn"):
              st.session_state.current_page += 1
              st.rerun()
    
    start_idx = (current_page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    display_df = processed_df.iloc[start_idx:end_idx]

    display_cols = [col for col in st.session_state.selected_columns if col in display_df.columns]
    display_df_final = display_df[display_cols] if display_cols else display_df # Show all if selection invalid

    st.dataframe(
         display_df_final,
         height=st.session_state.appearance_settings.get('table_height', 400),
         use_container_width=True,
         hide_index=not st.session_state.appearance_settings.get('show_index', False)
    )

    # --- Export Section --- 
    st.subheader("Export Data")
    export_cols = st.columns(2)
    
    # Use the helper function defined at the start of the file
    csv_export_current = convert_df_to_csv(display_df_final, st.session_state.selected_columns)
    csv_export_filtered = convert_df_to_csv(processed_df, st.session_state.selected_columns)

    with export_cols[0]:
        st.download_button(
            label="Download Current Page View (CSV)", data=csv_export_current,
            file_name='table_current_page.csv', mime='text/csv', key='download_current_page'
        )
    with export_cols[1]:
        st.download_button(
            label="Download Filtered Data (CSV)", data=csv_export_filtered,
            file_name='table_filtered_data.csv', mime='text/csv', key='download_filtered_data'
        )
