"1. First, split each page into sentences

import nltk

nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

sentences = []

for page_num, page_text in enumerate(pages):
    page_sentences = sent_tokenize(page_text)

    for sentence in page_sentences:
        sentences.append({
            "page": page_num + 1,
            "text": sentence
        })
        
        
"""    2. Clean the sentences
    Medical PDF extraction often produces:
    Extra whitespace
    Unnecessary line breaks
    Bullet characters
    Broken words (e.g. Substanc e)
    Incorrectly formatted dates
    You can start with:
"""

import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[●•]', '', text)
    text = text.strip()

    return text

for item in sentences:
    item["text"] = clean_text(item["text"])

sentences = [
    item for item in sentences
    if item["text"]
]

"""3. Extract features for your ML model
This is where things get interesting.

For example, your dataset might look like this:

Sentence

Potential category

Essential hypertension was identified in March 2024.

Diagnosis

Lisinopril 10 mg once daily.

Medication

Patient reports developing a rash after penicillin.

Allergy

HbA1c 5.9%.

Laboratory

Blood pressure 132/84 mmHg.

Vital signs

You can then transform text into numerical features using:

TF-IDF – good baseline for traditional ML.

Bag of Words – simple word-frequency representation.

Word2Vec – captures word relationships.

BioWordVec – biomedical word embeddings.

For a traditional ML baseline, I'd start with TF-IDF."""

from sklearn.feature_extraction.text import TfidfVectorizer

texts = [item["text"] for item in sentences]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(texts)

print(X.shape)
