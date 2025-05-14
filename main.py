import streamlit as st
from src.pss.app import initialize_app

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Football Analysis Dashboard",
            page_icon="⚽",
            layout="wide"
        )
        initialize_app()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all data files are in the correct location.") 