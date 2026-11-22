import streamlit as st
import fitz
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set up VADER
analyzer = SentimentIntensityAnalyzer()

# Streamlit page
st.set_page_config(page_title="Insurance Policy Analyzer")

st.title("AI Health Insurance Policy Analyzer")

st.write("Upload a PDF to extract its text and analyse individual clauses.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your insurance policy PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Open the PDF
    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    extracted_text = ""

    # Extract text from every page
    for page_number, page in enumerate(pdf_document):
        page_text = page.get_text()
        extracted_text += page_text + "\n"

    st.success("PDF uploaded and text extracted successfully!")

    # Display extracted text
    st.subheader("Extracted PDF Text")
    st.text_area(
        "Document contents",
        extracted_text,
        height=250
    )

    # Split text into individual sentences
    clauses = re.split(r'(?<=[.!?])\s+', extracted_text)

    # Remove empty sentences
    clauses = [
        clause.strip()
        for clause in clauses
        if clause.strip()
    ]

    st.subheader("Sentiment Analysis")

    # Analyse each clause
    for clause in clauses:

        scores = analyzer.polarity_scores(clause)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"

        elif compound <= -0.05:
            sentiment = "Negative"

        else:
            sentiment = "Neutral"

        # Display results
        with st.expander(clause[:100]):

            st.write("Full clause:")
            st.write(clause)

            st.write("Positive score:", scores["pos"])
            st.write("Negative score:", scores["neg"])
            st.write("Neutral score:", scores["neu"])
            st.write("Compound score:", compound)

            st.write("Classification:", sentiment)
