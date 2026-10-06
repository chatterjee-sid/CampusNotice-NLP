"""
app.py

Streamlit application interface for CampusNotice AI.
"""

import streamlit as st

def main():
    st.set_page_config(page_title="CampusNotice AI", layout="wide")
    st.title("CampusNotice AI")
    st.write("Welcome to CampusNotice AI dashboard.")

if __name__ == "__main__":
    main()
