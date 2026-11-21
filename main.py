#
# Upload PNG option
# Take text from PNG
# Analyze Text based off of Database stuff
#

import streamlit as st

st.title("File Uploader")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Save the file locally
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved {uploaded_file.name}!")
