#
# Upload PNG option
# Take text from PNG
# Analyze Text based off of Database stuff
#
from pathlib import Path
import streamlit as st
import os

st.title("File Uploader")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Save the file locally
    folder = Path("uploads")
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved {uploaded_file.name}!")
