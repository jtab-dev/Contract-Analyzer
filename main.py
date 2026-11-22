# Upload PNG option
# Take text from PNG
# Analyze Text based off of Database stuff

from pathlib import Path
import os
import re
import time
import shutil
import sys

import streamlit as st
import pymupdf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk_data_dir = os.path.join(sys.prefix, "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)

analyzer = SentimentIntensityAnalyzer()


# ------- DATABASE ------------
MEDICAL_LEXICON = {
    # Reassuring / positive descriptions
    "recovered": 2.5,
    "recovery": 2.2,
    "improved": 2.0,
    "improvement": 2.0,
    "stable": 1.0,
    "controlled": 1.2,
    "normal": 1.2,
    "healthy": 2.0,
    "resolved": 1.8,
    "effective": 1.8,
    "successful": 2.0,
    "healing": 1.8,
    "manageable": 0.8,
    "uncomplicated": 0.8,
    "adequate": 0.8,
    "optimal": 1.5,
    "responding": 1.0,
    "symptom_free": 2.0,
    "pain_free": 2.0,
    # Concerning descriptions / symptoms
    "severe": -2.5,
    "critical": -3.0,
    "worsening": -2.5,
    "deteriorating": -2.5,
    "complication": -1.8,
    "complications": -1.8,
    "infection": -1.2,
    "inflammation": -1.0,
    "abnormal": -1.5,
    "uncontrolled": -2.0,
    "recurrence": -1.8,
    "relapse": -2.2,
    "hospitalization": -1.2,
    "emergency": -1.5,
    "adverse": -1.5,
    "declining": -1.8,
    "dysfunction": -1.8,
    "failure": -2.5,
    "pain": -1.5,
    "headache": -0.8,
    "fatigue": -0.8,
    "nausea": -1.0,
    "vomiting": -1.5,
    "dizziness": -0.8,
    "weakness": -0.8,
    "breathlessness": -1.8,
    "bleeding": -2.0,
    "fever": -1.0,
    "shortness of breath": -1.8,
    "swelling": -1.0,
    "rash": -0.8,
    "dehydration": -1.2,
    "fainting": -1.8,
    "seizure": -2.5,
    "unresponsive": -2.0,
    # Diagnoses and clinical terms: neutral, not inherently sentiment
    "hypertension": 0.0,
    "essential hypertension": 0.0,
    "prediabetes": 0.0,
    "diabetes": 0.0,
    "type 1 diabetes": 0.0,
    "type 2 diabetes": 0.0,
    "asthma": 0.0,
    "mild intermittent asthma": 0.0,
    "allergic rhinitis": 0.0,
    "seasonal allergic rhinitis": 0.0,
    "urinary tract infection": 0.0,
    "uti": 0.0,
    "chronic": 0.0,
    "mild": 0.0,
    "moderate": 0.0,
    "diagnosis": 0.0,
    "condition": 0.0,
    "symptom": 0.0,
    "symptoms": 0.0,
    "medical history": 0.0,
    "comorbidity": 0.0,
    "comorbidities": 0.0,
    # Medication / treatment terms
    "medication": 0.0,
    "medications": 0.0,
    "treatment": 0.0,
    "therapy": 0.0,
    "prescribed": 0.0,
    "dosage": 0.0,
    "antibiotic": 0.0,
    "antibiotics": 0.0,
    "inhaler": 0.0,
    "rescue inhaler": 0.0,
    "side effect": -1.2,
    "side effects": -1.2,
    "allergic reaction": -1.8,
    "contraindicated": -1.8,
    "treatment failure": -2.5,
    # Laboratory / test language
    "within range": 1.0,
    "normal range": 1.0,
    "elevated": -0.8,
    "elevated levels": -1.0,
    "abnormal result": -1.5,
    "abnormal results": -1.5,
    "borderline": -0.4,
    "deficiency": -1.0,
    "insufficient": -0.8,
    "test result": 0.0,
    "laboratory": 0.0,
    "lab result": 0.0,
}

INSURANCE_POLICY_LEXICON = {
    # Potentially favorable administrative language
    "covered": 0.8,
    "coverage": 0.5,
    "eligible": 1.0,
    "approved": 1.5,
    "authorized": 1.0,
    "payable": 1.0,
    "reimbursable": 1.0,
    "reimbursement": 0.8,
    "benefit": 0.8,
    "benefits": 0.8,
    "in-network": 0.5,
    "preventive": 0.5,
    "medically necessary": 0.5,
    # Potentially unfavorable administrative language
    "excluded": -1.5,
    "exclusion": -1.5,
    "ineligible": -1.5,
    "denied": -2.0,
    "denial": -2.0,
    "rejected": -1.5,
    "noncovered": -1.5,
    "uncovered": -1.5,
    "nonpayable": -1.5,
    "nonreimbursable": -1.5,
    "restricted": -0.8,
    "prohibited": -1.5,
    "termination": -1.0,
    "lapse": -1.0,
    "cancellation": -1.2,
    # Usually neutral contract / plan terms
    "deductible": 0.0,
    "copayment": 0.0,
    "copay": 0.0,
    "coinsurance": 0.0,
    "premium": 0.0,
    "out-of-pocket": 0.0,
    "maximum": 0.0,
    "limit": 0.0,
    "limitation": 0.0,
    "waiting period": 0.0,
    "prior authorization": 0.0,
    "preauthorization": 0.0,
    "network": 0.0,
    "provider": 0.0,
    "claim": 0.0,
    "policyholder": 0.0,
    "insured": 0.0,
    "dependent": 0.0,
    "rider": 0.0,
    "endorsement": 0.0,
    "subrogation": 0.0,
    "coordination of benefits": 0.0,
    "pre-existing": 0.0,
    "preexisting": 0.0,
    "experimental": 0.0,
    "investigational": 0.0,
}

INSURANCE_POLICY_LEXICON = {
    # Potentially favorable administrative language
    "covered": 0.8,
    "coverage": 0.5,
    "eligible": 1.0,
    "approved": 1.5,
    "authorized": 1.0,
    "payable": 1.0,
    "reimbursable": 1.0,
    "reimbursement": 0.8,
    "benefit": 0.8,
    "benefits": 0.8,
    "in-network": 0.5,
    "preventive": 0.5,
    "medically necessary": 0.5,
    # Potentially unfavorable administrative language
    "excluded": -1.5,
    "exclusion": -1.5,
    "ineligible": -1.5,
    "denied": -2.0,
    "denial": -2.0,
    "rejected": -1.5,
    "noncovered": -1.5,
    "uncovered": -1.5,
    "nonpayable": -1.5,
    "nonreimbursable": -1.5,
    "restricted": -0.8,
    "prohibited": -1.5,
    "termination": -1.0,
    "lapse": -1.0,
    "cancellation": -1.2,
    # Usually neutral contract / plan terms
    "deductible": 0.0,
    "copayment": 0.0,
    "copay": 0.0,
    "coinsurance": 0.0,
    "premium": 0.0,
    "out-of-pocket": 0.0,
    "maximum": 0.0,
    "limit": 0.0,
    "limitation": 0.0,
    "waiting period": 0.0,
    "prior authorization": 0.0,
    "preauthorization": 0.0,
    "network": 0.0,
    "provider": 0.0,
    "claim": 0.0,
    "policyholder": 0.0,
    "insured": 0.0,
    "dependent": 0.0,
    "rider": 0.0,
    "endorsement": 0.0,
    "subrogation": 0.0,
    "coordination of benefits": 0.0,
    "pre-existing": 0.0,
    "preexisting": 0.0,
    "experimental": 0.0,
    "investigational": 0.0,
}

POTENTIAL_COVERAGE_CUES = [
    "covered services",
    "is covered",
    "are covered",
    "we will pay",
    "the plan pays",
    "eligible expenses",
    "covered benefit",
    "covered benefits",
    "medically necessary",
    "subject to the terms",
    "subject to the limits",
    "in-network services",
    "preventive care",
    "prescription drug coverage",
    "hospital services",
    "emergency services",
    "diagnostic testing",
    "physician services",
]

POTENTIAL_EXCLUSION_CUES = [
    "not covered",
    "is not covered",
    "are not covered",
    "excluded from coverage",
    "coverage is excluded",
    "we will not pay",
    "the plan will not pay",
    "not eligible for benefits",
    "non-covered services",
    "noncovered services",
    "experimental or investigational",
    "cosmetic",
    "services not medically necessary",
    "outside the provider network",
    "out-of-network",
    "waiting period",
    "pre-existing condition exclusion",
    "subject to an exclusion",
]

CONDITIONAL_COVERAGE_CUES = [
    "prior authorization required",
    "preauthorization required",
    "subject to prior authorization",
    "referral required",
    "after the deductible",
    "subject to deductible",
    "subject to copayment",
    "subject to coinsurance",
    "up to the plan maximum",
    "within the annual limit",
    "only when medically necessary",
    "when provided by an in-network provider",
]

COVERAGE_LEXICON = {phrase: 1.5 for phrase in POTENTIAL_COVERAGE_CUES}
EXCLUSION_LEXICON = {phrase: -2.0 for phrase in POTENTIAL_EXCLUSION_CUES}
CONDITIONAL_LEXICON = {phrase: -0.5 for phrase in CONDITIONAL_COVERAGE_CUES}

ALL_CUSTOM_LEXICONS = [
    MEDICAL_LEXICON,
    INSURANCE_POLICY_LEXICON,
    COVERAGE_LEXICON,
    EXCLUSION_LEXICON,
    CONDITIONAL_LEXICON,
]


# ADDING TO VADER LEXICON
parsed_lexicon = {}
phrases = []

for lex in ALL_CUSTOM_LEXICONS:
    for term, score in lex.items():
        clean_term = term.lower().strip().replace(" ", "_")
        parsed_lexicon[clean_term] = float(score)
        if "_" in clean_term:
            phrases.append(clean_term.replace("_", " "))
analyzer.lexicon.update(parsed_lexicon)


# -------- Function Definitions -------------
def get_pdf_text(pdf_path: str) -> list[str]:
    result = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            raw = str(page.get_text("text"))
            clean = " ".join(raw.split())
            result.append(clean)

    return result


def get_page_sentences(pages: list[str]) -> list[str]:
    result = [sentence for para in pages for sentence in nltk.sent_tokenize(para)]
    return result


def analyze_domain_sentiment(text: str) -> dict:
    processed_text = text.lower()
    for phrase in phrases:
        # Match whole phrases and join them with underscores
        pattern = r"\b" + re.escape(phrase) + r"\b"
        processed_text = re.sub(pattern, phrase.replace(" ", "_"), processed_text)

    return analyzer.polarity_scores(processed_text)


def analyze_sentiment(sentences: list[str]):
    results = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            continue
        if not sentence.strip():
            continue
        scores = analyze_domain_sentiment(sentence)

        results.append(
            {
                "sentence": sentence,
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "compound": scores["compound"],
            }
        )
    return results


def get_color(data: list[dict]):
    for item in data:
        sentence = item["sentence"]
        score = item["compound"]

        if score > 0.05:
            color = "green"
        elif score < -0.05:
            color = "red"
        else:
            color = "gray"

        safe_sentence = (
            sentence.replace("\\", "\\\\")
            .replace("[", r"\[")
            .replace("]", r"\]")
            .replace("$", r"\$")
        )

        st.write(f":{color}[{safe_sentence}] '({score})'")


def get_total(sentences: list[str]) -> float:
    total = 0.0
    for sentence in sentences:
        amounts = re.findall(
            r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", sentence
        )
        for amount in amounts:
            total += float(amount.replace(",", ""))
    return total


AILMENT_SERVICE_REQUIREMENTS = {
    "essential hypertension": ["primary care", "generic formulary", "prescription"],
    "prediabetes": ["primary care", "diagnostic laboratory", "preventive"],
    "mild intermittent asthma": ["primary care", "generic formulary", "prescription"],
    "seasonal allergic rhinitis": ["primary care", "generic formulary", "prescription"],
    "uncomplicated urinary tract infection": [
        "primary care",
        "generic formulary",
        "prescription",
    ],
}


def extract_policy_benefits(policy_array: list[str], cues: list[str]) -> list[str]:
    """Finds policy chunks triggered by coverage cues and captures subsequent benefit rows."""
    relevant_chunks = []
    in_benefits_section = False
    in_exclusions_section = False

    for chunk in policy_array:
        lower_chunk = chunk.lower()

        # Stop capturing if we hit exclusions
        if "exclusions and limitations" in lower_chunk:
            in_benefits_section = False
            in_exclusions_section = True
            continue
        if "prior authorization and utilization" in lower_chunk:
            in_exclusions_section = False

        # Trigger capture when any cue matches
        if any(cue in lower_chunk for cue in cues) and not in_exclusions_section:
            in_benefits_section = True
            relevant_chunks.append(chunk)
        elif in_benefits_section and not in_exclusions_section:
            # Capture table rows immediately following a "Covered Benefits" header
            if any(
                term in lower_chunk
                for term in [
                    "copay",
                    "coinsurance",
                    "deductible",
                    "formulary",
                    "preventive",
                ]
            ):
                relevant_chunks.append(chunk)
            else:
                in_benefits_section = False

    return relevant_chunks


def extract_ailments(medical_array: list[str]) -> list[dict]:
    """Isolates the diagnosis block and extracts active/resolved conditions."""
    ailments = []
    for chunk in medical_array:
        if "medical history and active diagnoses" in chunk.lower():
            # Match pattern: <Condition Name> <MM/DD/YYYY> <Active|Resolved>
            matches = re.findall(
                r"([A-Za-z\s]+?)\s+(\d{2}/\d{2}/\d{4})\s+(Active|Resolved)", chunk
            )
            for condition, date, status in matches:
                clean_name = condition.replace(
                    "Diagnosis Date identified Current status Clinical notes", ""
                ).strip()
                ailments.append({"ailment": clean_name, "status": status, "date": date})
    return ailments


def verify_coverage(medical_array: list[str], policy_array: list[str]) -> list[dict]:
    benefit_chunks = extract_policy_benefits(policy_array, POTENTIAL_COVERAGE_CUES)
    combined_benefits_text = " ".join(benefit_chunks).lower()
    ailments = extract_ailments(medical_array)

    verification_results = []
    for item in ailments:
        key = item["ailment"].lower()
        required_services = AILMENT_SERVICE_REQUIREMENTS.get(key, ["primary care"])

        matched_services = [
            svc for svc in required_services if svc in combined_benefits_text
        ]

        is_covered = len(matched_services) > 0
        verification_results.append(
            {
                "ailment": item["ailment"],
                "status": item["status"],
                "is_covered": is_covered,
                "matched_policy_services": matched_services,
                "missing_services": [
                    s for s in required_services if s not in matched_services
                ],
            }
        )

    return verification_results


# ---------- FRONT STACK -----------------
# Upload a File
st.title("Contract Analyzer")

willing_cost = st.number_input(
    "How much are you able to spend on health insurance, annually?",
    value=None,
    placeholder="Input in USD ($)",
)

uploaded_file = st.file_uploader("Choose a file to analyze")
folder = Path("uploads")
folder.mkdir(parents=True, exist_ok=True)

if uploaded_file is not None:
    file_path = folder / uploaded_file.name
    if file_path.suffix.lower() in {".pdf", ".png"}:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved {uploaded_file.name}!")
time.sleep(0.05)

# --------- MACHINE LEARNING PIPELINE ----------------
if uploaded_file is not None:
    message = "Generating Analysis..."
    placeholder = st.empty()
    current_text = ""
    for char in message:
        current_text += char
        placeholder.write(current_text)
        time.sleep(0.02)

    file_path = folder / uploaded_file.name
    new = get_pdf_text(file_path)
    new_parsed = get_page_sentences(new)
    results = analyze_sentiment(new_parsed)
    # --------- MACHINE LEARNING PIPELINE ----------------

    # --------- OUTPUT ------------ Financial Analysis Then Coverage Analysis
    # Financial Analysis
    if willing_cost is not None:
        st.title("Financial Analysis")
        cost = get_total(new_parsed)
        if willing_cost > cost:
            st.write(f":{'green'}[{'You can afford this health insurance !!!!!!!!'}]")
            st.write(f"${willing_cost}     >     ${cost}")
        else:
            st.write(
                f":{'red'}[{'You cannot afford this health insurance :((((((((((((((('}]"
            )
            st.write(f"${willing_cost}     <     ${cost}")
    # Coverage Analysis
    st.title("Coverage Analysis")
    st.write(f":{'green'}[Green] = Covered | :{'red'}[Red] = Not Covered")

    history = get_pdf_text("archive/medical_history.pdf")
    history_parsed = get_page_sentences(history)

    coverage = verify_coverage(history_parsed, new_parsed)
    for item in coverage:
        if item.get("is_covered"):
            color = "green"
        else:
            color = "red"
        st.write(f":{color}[{item.get('ailment')}] : {item.get('status')}")

    # Color PDF Text
    st.title("Full-text Overview")
    get_color(results)
# --------- OUTPUT ------------
