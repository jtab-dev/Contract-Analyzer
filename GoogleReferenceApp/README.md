# ClauseVault — US Health Insurance ML & Policy Intelligence (Python)

A Python implementation of the **US Health Insurance Policy Analyzer, Affordability Calculator, Medical Coverage Matcher, and TF-IDF + Multinomial Naive Bayes Sentiment Classifier** (zero LLM/AI API calls).

## Features
- **3-Step Intake**: Accepts US Health Insurance policies (PDF, PNG/JPG/WEBP OCR, or text), Medical History documents, and Financial Background documents (W-2, paystub, budget).
- **TF-IDF + Multinomial Naive Bayes (`ml_health_engine.py`)**:
  - Splits policies into individual clauses.
  - Vectorizes unigrams + bigrams with TF-IDF (`TfidfVectorizer`, preserving negation and `$`/`%` tokens).
  - Classifies each clause as **Positive**, **Neutral**, or **Negative** with calibrated probabilities using `MultinomialNB(alpha=1.0)`.
  - Aggregates negative clauses into **Most Common Unfavorable Conditions** (e.g., *Pre-Existing Condition Exclusions*, *Prior Authorization & Step Therapy*, *Balance Billing Exposure*, *Lifetime Caps*).
- **Affordability & Buy Decision**: Evaluates monthly premium, annual deductible, and out-of-pocket maximum against household net pay, fixed obligations, Savings/HSA balance, 2026 FPL estimate, and the **ACA 8.5% income affordability threshold**.
- **Medical Eligibility & Coverage Matrix**: Maps extracted chronic conditions, prescriptions, and anticipated care against policy terms to output **Apply / Apply with Caution / Do Not Apply**.
- **Contract & Risk Engine (`contract_engine.py`)**: Extracts clauses, parties, key dates, contradictions, missing ACA disclosures, authenticity checks, and version diffs.
- **Google Firestore & GLEIF Registry (`firebase_store.py`)**: Persists policy records, custom risk rules, and immutable lock-in snapshots in Cloud Firestore.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run automated ML & policy pipeline tests
python3 test_pipeline.py

# Start FastAPI server on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note on PDF & OCR**: For scanned image OCR, install the system `tesseract-ocr` binary (`brew install tesseract` on macOS or `sudo apt-get install tesseract-ocr` on Ubuntu/Debian).
