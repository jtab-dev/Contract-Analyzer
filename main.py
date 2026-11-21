#
# Upload PNG option
# Take text from PNG
# Analyze Text based off of Database stuff
#
from pathlib import Path
import os
import time
import shutil

import streamlit as st

# Upload a File
st.title("Upload a File")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    folder = Path("uploads")
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / uploaded_file.name
    if file_path.suffix.lower() in {".pdf", ".png"}:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}!")
time.sleep(0.05)

# Start Analyzing Uploaded File
empty = True
for _ in os.scandir("uploads"):
    empty = False
    break
if not empty:
    message = "Generating Analysis..."
    placeholder = st.empty()
    current_text = ""
    for char in message:
        current_text += char
        placeholder.write(current_text)
        time.sleep(0.02)

# --------- MACHINE LEARNING PIPELINE ----------------
# --------- MACHINE LEARNING PIPELINE ----------------
# --------- MACHINE LEARNING PIPELINE ----------------

# Put Uploaded Files into the Archive
for file in os.scandir("uploads"):
    shutil.move("uploads/" + file.name, "archive/" + file.name)
