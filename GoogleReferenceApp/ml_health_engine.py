from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.naive_bayes import MultinomialNB  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# 1. Labeled US Health Insurance Clause Training Corpus (68 Clauses)
# ---------------------------------------------------------------------------

TRAINING_CORPUS: List[Tuple[str, str]] = [
    # POSITIVE (24)
    ("All ACA-mandated preventive wellness exams immunizations and routine screenings are covered at $0 copay 100% covered with no deductible.", "Positive"),
    ("Pre-existing conditions including diabetes asthma hypertension and heart disease are fully covered from day one with no waiting period.", "Positive"),
    ("There are zero lifetime dollar limits and zero annual benefit caps on essential health benefits.", "Positive"),
    ("Insulin prescriptions and diabetes testing supplies are capped at $35 per 30-day supply with deductible waived.", "Positive"),
    ("Outpatient mental health therapy and behavioral health counseling are covered on parity at $0 telehealth copay.", "Positive"),
    ("Generic Tier 1 medications are covered at $10 copay with deductible waived.", "Positive"),
    ("Primary care office visits require only a low $25 copay before deductible is met.", "Positive"),
    ("Emergency services and ancillary radiology or laboratory services are protected under the federal No Surprises Act with zero balance billing.", "Positive"),
    ("Once the out-of-pocket maximum is reached the insurer pays 100% of covered essential health benefits for the remainder of the year.", "Positive"),
    ("Diagnostic outpatient bloodwork and routine preventive imaging require no prior authorization.", "Positive"),
    ("Maternity care newborn coverage and pediatric dental and vision benefits are fully included as essential health benefits.", "Positive"),
    ("Urgent care visits and telehealth virtual consultations are covered at a flat $0 to $30 copay with deductible waived.", "Positive"),
    ("In-network specialist visits including endocrinology and cardiology are covered at a fixed $50 copay without deductible.", "Positive"),
    ("Physical therapy and rehabilitative services are covered up to 60 visits per plan year with no referral required.", "Positive"),
    ("Independent external appeal rights are guaranteed within 180 days at zero cost to the member.", "Positive"),
    ("Preventive cancer screenings mammograms and colonoscopies are paid 100% in-network with $0 member cost share.", "Positive"),
    ("Policy is guaranteed renewable regardless of changes in health status or medical claims history.", "Positive"),
    ("Brand medications on the preferred formulary have predictable copays capped at $35 without percentage coinsurance.", "Positive"),
    ("Hospital emergency room visits worldwide are covered at in-network cost-sharing rates without balance billing.", "Positive"),
    ("Durable medical equipment including glucose monitors and CPAP supplies are covered at 80% after deductible.", "Positive"),
    ("Deductible waived for primary care visits generic prescriptions and outpatient mental health appointments.", "Positive"),
    ("Annual wellness checkups and routine laboratory panels are covered 100% with no out-of-pocket charge.", "Positive"),
    ("Continuity of care protects ongoing cancer or pregnancy treatment for 90 days if a provider leaves the network.", "Positive"),
    ("Out-of-pocket maximum caps total annual member spending for covered in-network care.", "Positive"),
    # NEGATIVE (24)
    ("Any medical condition diabetes asthma hypertension or illness diagnosed or treated within five years prior is strictly excluded as a pre-existing condition.", "Negative"),
    ("This policy is not ACA-compliant minimum essential coverage under federal law and omits essential health benefits.", "Negative"),
    ("Failure to obtain prior authorization at least 72 hours before non-emergency MRI or surgery results in a 50% benefit penalty charged to policyholder.", "Negative"),
    ("Total plan lifetime maximum benefit is strictly capped at $250,000 and expenses exceeding the cap are excluded and paid 100% by policyholder.", "Negative"),
    ("Outpatient prescription drugs including insulin inhalers and specialty biologics are not covered under this policy.", "Negative"),
    ("Mental health counseling substance use treatment preventive wellness exams and maternity care are completely excluded.", "Negative"),
    ("A 12-month waiting period applies to outpatient surgery imaging and diagnostic specialist consultations.", "Negative"),
    ("Out-of-network providers are not covered and policyholder is responsible for 100% of hospital charges and balance billing.", "Negative"),
    ("Out-of-network diagnostic lab testing is subject to balance billing above usual and customary rates.", "Negative"),
    ("Specialty biologics require 30% to 50% coinsurance after deductible and mandatory fail-first step therapy before dispensing.", "Negative"),
    ("Policyholder must pay a high $10,000 individual deductible and 50% coinsurance on all hospital and specialist claims.", "Negative"),
    ("Policy automatically renews at insurer then-current rate unless cancelled in writing 60 days prior to expiration.", "Negative"),
    ("Insurer reserves the right to rescind or cancel coverage or impose undisclosed administrative surcharges without prior notice.", "Negative"),
    ("Emergency room copay is $750 plus 40% coinsurance after deductible is satisfied.", "Negative"),
    ("Annual maximum benefit payout is limited to $100,000 per calendar year with no out-of-pocket cap.", "Negative"),
    ("Pre-existing condition exclusion bars reimbursement for chronic illness medications and follow-up visits for 12 months.", "Negative"),
    ("Step therapy forces patient to fail cheaper alternative drugs before insurer approves prescribed specialist medication.", "Negative"),
    ("Claims denied if prior authorization penalty applies or referral from primary physician is not filed in advance.", "Negative"),
    ("Zero out-of-network coverage and surprise balance billing charges do not count toward the annual deductible.", "Negative"),
    ("Policyholder pays 100% of prescription costs until high deductible is met with no insulin cap protection.", "Negative"),
    ("Rehabilitation mental health and maternity services are excluded riders requiring separate supplemental fees.", "Negative"),
    ("Coverage terminates immediately upon non-payment with no grace period and forfeits prepaid fees.", "Negative"),
    ("Experimental investigational or off-label therapies are strictly excluded at insurer sole discretion.", "Negative"),
    ("Late payment penalty and administrative fee charged retroactively if premium arrives after the 5th of the month.", "Negative"),
    # NEUTRAL (20)
    ("This health insurance policy is issued effective January 1, 2026 and expires December 31, 2026.", "Neutral"),
    ("This policy is governed by the laws of the state of California and applicable federal regulations.", "Neutral"),
    ("Monthly premium payments are due on the 1st day of each calendar month via electronic funds transfer or check.", "Neutral"),
    ("In witness whereof the insurer and policyholder execute this certificate as of the effective date.", "Neutral"),
    ("Member identification cards and network provider directories are maintained online and updated monthly.", "Neutral"),
    ("Plan year runs on a standard calendar year basis from January 1 through December 31.", "Neutral"),
    ("Coordination of benefits determines primary and secondary payer order when member holds dual group coverage.", "Neutral"),
    ("Notices required under this policy shall be delivered in writing to the address on file.", "Neutral"),
    ("Definitions section establishes meanings for allowed amount coinsurance copayment deductible and network provider.", "Neutral"),
    ("Claims must be submitted on standard CMS-1500 or UB-04 billing forms within 180 days of service date.", "Neutral"),
    ("Enrolled member must notify insurer within 30 days of qualifying life events such as marriage birth or relocation.", "Neutral"),
    ("Subrogation and reimbursement provisions apply when a third-party liability settlement recovers medical expenses.", "Neutral"),
    ("Grievance procedures outline mailing addresses and customer service telephone hours for member inquiries.", "Neutral"),
    ("Entire contract clause states that the policy certificate application and schedule of benefits constitute the full agreement.", "Neutral"),
    ("Headings and section numbers are inserted for reference convenience and do not alter substantive provisions.", "Neutral"),
    ("Premium billing statements are issued 15 days prior to each monthly due date.", "Neutral"),
    ("Dependent eligibility extends to biological adopted or stepchildren through age 26 under plan rules.", "Neutral"),
    ("Conversion and continuation rights under COBRA are administered according to federal statutory timelines.", "Neutral"),
    ("Records retention and HIPAA privacy notices govern disclosure of protected health information.", "Neutral"),
    ("Annual open enrollment window occurs between November 1 and January 15 unless a special enrollment applies.", "Neutral"),
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "will", "with", "under", "shall", "may", "such", "any", "all", "each",
    "per", "if", "into", "than", "their", "they", "when", "where", "which", "who",
}


def tokenize_clause(text: str) -> List[str]:
    normalized = re.sub(r"\$(\d[\d,]*)", r" dollar_\1 ", text.lower())
    normalized = re.sub(r"(\d+)\s*%", r" pct_\1 ", normalized)
    normalized = re.sub(r"[^a-z0-9_$]+", " ", normalized).strip()
    if not normalized:
        return []
    unigrams: List[str] = []
    for tok in normalized.split():
        if (
            tok in ("no", "not", "never", "zero")
            or tok.startswith("dollar_")
            or tok.startswith("pct_")
            or (tok not in STOPWORDS and len(tok) >= 2)
        ):
            unigrams.append(tok)
    features = list(unigrams)
    for i in range(len(unigrams) - 1):
        features.append(f"{unigrams[i]}_{unigrams[i + 1]}")
    return features


# ---------------------------------------------------------------------------
# 2. Dual-Backend TF-IDF + Multinomial Naive Bayes Classifier
#    Uses scikit-learn when installed, with a pure-Python fallback
# ---------------------------------------------------------------------------

class PurePythonSentimentModel:
    CLASSES = ["Positive", "Negative", "Neutral"]

    def __init__(self, corpus: List[Tuple[str, str]], alpha: float = 1.0):
        self.alpha = alpha
        texts = [t for t, _ in corpus]
        labels = [l for _, l in corpus]
        n_docs = len(texts)

        df_counts: Dict[str, int] = {}
        for doc in texts:
            for term in set(tokenize_clause(doc)):
                df_counts[term] = df_counts.get(term, 0) + 1

        self.feature_names = sorted(df_counts.keys())
        self.vocab = {t: i for i, t in enumerate(self.feature_names)}
        self.idf = [math.log((1.0 + n_docs) / (1.0 + df_counts[t])) + 1.0 for t in self.feature_names]

        v_size = len(self.feature_names)
        class_counts = {c: 0 for c in self.CLASSES}
        feat_sums = {c: [0.0] * v_size for c in self.CLASSES}
        total_weights = {c: 0.0 for c in self.CLASSES}

        for doc, lbl in zip(texts, labels):
            class_counts[lbl] += 1
            vec = self._vectorize(doc)
            for idx, val in vec.items():
                feat_sums[lbl][idx] += val
                total_weights[lbl] += val

        self.class_log_prior = {c: math.log(class_counts[c] / n_docs) for c in self.CLASSES}
        self.feature_log_prob: Dict[str, List[float]] = {}
        for c in self.CLASSES:
            denom = total_weights[c] + self.alpha * v_size
            self.feature_log_prob[c] = [
                math.log((feat_sums[c][i] + self.alpha) / denom) for i in range(v_size)
            ]

    def _vectorize(self, doc: str) -> Dict[int, float]:
        tf: Dict[int, int] = {}
        for tok in tokenize_clause(doc):
            idx = self.vocab.get(tok)
            if idx is not None:
                tf[idx] = tf.get(idx, 0) + 1
        weighted: Dict[int, float] = {}
        norm_sq = 0.0
        for idx, count in tf.items():
            val = count * self.idf[idx]
            weighted[idx] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for idx in weighted:
                weighted[idx] /= norm
        return weighted

    def predict(self, text: str) -> Dict[str, Any]:
        vec = self._vectorize(text)
        log_joint: Dict[str, float] = {}
        for c in self.CLASSES:
            score = self.class_log_prior[c]
            lp = self.feature_log_prob[c]
            for idx, val in vec.items():
                score += val * lp[idx]
            log_joint[c] = score

        max_log = max(log_joint.values())
        exp_pos = math.exp(log_joint["Positive"] - max_log)
        exp_neg = math.exp(log_joint["Negative"] - max_log)
        exp_neu = math.exp(log_joint["Neutral"] - max_log)
        total = exp_pos + exp_neg + exp_neu or 1.0

        pos_p = round(exp_pos / total, 3)
        neg_p = round(exp_neg / total, 3)
        neu_p = max(0.0, round(1.0 - pos_p - neg_p, 3))

        sentiment = "Neutral"
        if pos_p >= neg_p and pos_p >= neu_p:
            sentiment = "Positive"
        elif neg_p >= pos_p and neg_p >= neu_p:
            sentiment = "Negative"

        scored: List[Tuple[str, float]] = []
        others = [c for c in self.CLASSES if c != sentiment]
        for idx, val in vec.items():
            term = self.feature_names[idx]
            if term.startswith("dollar_") or term.startswith("pct_"):
                continue
            cls_lp = self.feature_log_prob[sentiment][idx]
            other_lp = sum(self.feature_log_prob[o][idx] for o in others) / len(others)
            scored.append((term.replace("_", " "), val * (cls_lp - other_lp)))
        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "sentiment": sentiment,
            "positiveProb": pos_p,
            "negativeProb": neg_p,
            "neutralProb": neu_p,
            "topFeatures": [t for t, _ in scored[:5]],
        }


class SklearnSentimentModel:
    def __init__(self, corpus: List[Tuple[str, str]], alpha: float = 1.0):
        texts = [t for t, _ in corpus]
        labels = [l for _, l in corpus]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_clause,
            token_pattern=None,
            lowercase=False,
            smooth_idf=True,
            norm="l2",
        )
        X = self.vectorizer.fit_transform(texts)
        self.model = MultinomialNB(alpha=alpha)
        self.model.fit(X, labels)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        self.classes = list(self.model.classes_)

    def predict(self, text: str) -> Dict[str, Any]:
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        prob_map = {cls: float(p) for cls, p in zip(self.classes, probs)}
        pos_p = round(prob_map.get("Positive", 0.0), 3)
        neg_p = round(prob_map.get("Negative", 0.0), 3)
        neu_p = max(0.0, round(1.0 - pos_p - neg_p, 3))

        sentiment = "Neutral"
        if pos_p >= neg_p and pos_p >= neu_p:
            sentiment = "Positive"
        elif neg_p >= pos_p and neg_p >= neu_p:
            sentiment = "Negative"

        cls_idx = self.classes.index(sentiment)
        other_idxs = [i for i, c in enumerate(self.classes) if c != sentiment]
        row = X.tocoo()
        scored: List[Tuple[str, float]] = []
        for idx, val in zip(row.col, row.data):
            term = self.feature_names[idx]
            if term.startswith("dollar_") or term.startswith("pct_"):
                continue
            cls_lp = float(self.model.feature_log_prob_[cls_idx, idx])
            other_lp = sum(float(self.model.feature_log_prob_[oi, idx]) for oi in other_idxs) / len(other_idxs)
            scored.append((term.replace("_", " "), float(val) * (cls_lp - other_lp)))
        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "sentiment": sentiment,
            "positiveProb": pos_p,
            "negativeProb": neg_p,
            "neutralProb": neu_p,
            "topFeatures": [t for t, _ in scored[:5]],
        }


_MODEL = SklearnSentimentModel(TRAINING_CORPUS) if HAS_SKLEARN else PurePythonSentimentModel(TRAINING_CORPUS)


# ---------------------------------------------------------------------------
# 3. Clause Splitting & Unfavorable Condition Clustering
# ---------------------------------------------------------------------------

UNFAVORABLE_THEMES: List[Tuple[str, re.Pattern[str]]] = [
    ("Pre-Existing Condition Exclusions & Waiting Periods", re.compile(r"pre-?existing|waiting\s+period|prior\s+to\s+the\s+effective\s+date", re.I)),
    ("Prior Authorization & Step Therapy Barriers", re.compile(r"prior\s+authorization|step\s+therapy|fail-?first|benefit\s+penalty|72\s+hours", re.I)),
    ("Out-of-Network / Balance Billing Exposure", re.compile(r"out-?of-?network|balance\s+billing|usual\s+and\s+customary|zero\s+out-of-network", re.I)),
    ("High Deductible & Coinsurance Cost-Share", re.compile(r"deductible|coinsurance|\$\d[\d,]*\s+individual", re.I)),
    ("Annual / Lifetime Benefit Caps", re.compile(r"lifetime\s+maximum|benefit\s+cap|capped\s+at|annual\s+maximum|exceeding\s+\$", re.I)),
    ("Prescription Formulary & Specialty Tier Restrictions", re.compile(r"prescription|insulin|biologics|formulary|specialty|tier\s+[34]", re.I)),
    ("Essential Health Benefit Exclusions (Mental Health / Maternity / Preventive)", re.compile(r"not\s+aca-compliant|excluded|mental\s+health|maternity|preventive", re.I)),
]


def split_policy_into_clauses(policy_text: str, parsed_clauses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if parsed_clauses:
        for c in parsed_clauses:
            title = c.get("title", "")
            body = c.get("excerpt") or c.get("plainSummary") or ""
            combined = f"{title}: {body}".strip() if title else body.strip()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if len(s.strip()) >= 20]
            if sentences:
                for s in sentences:
                    items.append({"sectionRef": f"Sec {c.get('sectionNumber', '')}", "text": s[:480]})
            elif len(combined) >= 15:
                items.append({"sectionRef": f"Sec {c.get('sectionNumber', '')}", "text": combined[:480]})

    if len(items) < 3:
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", policy_text)
            if len(s.strip()) >= 25 and not re.match(r"^IN WITNESS WHEREOF", s.strip(), re.I)
        ]
        for idx, s in enumerate(raw_sentences[:60]):
            items.append({"sectionRef": f"Clause {idx + 1}", "text": s[:480]})
    return items[:75]


def run_ml_sentiment_analysis(
    policy_text: str,
    parsed_clauses: List[Dict[str, Any]],
    user_context_text: str = "",
) -> Dict[str, Any]:
    clause_units = split_policy_into_clauses(policy_text, parsed_clauses)
    classifications: List[Dict[str, Any]] = []
    for idx, unit in enumerate(clause_units, start=1):
        pred = _MODEL.predict(unit["text"])
        classifications.append({
            "clauseIndex": idx,
            "sectionRef": unit["sectionRef"],
            "text": unit["text"],
            **pred,
        })

    total = len(classifications)
    pos_c = sum(1 for c in classifications if c["sentiment"] == "Positive")
    neg_c = sum(1 for c in classifications if c["sentiment"] == "Negative")
    neu_c = sum(1 for c in classifications if c["sentiment"] == "Neutral")

    pos_pct = round((pos_c / total) * 100) if total else 0
    neg_pct = round((neg_c / total) * 100) if total else 0
    neu_pct = max(0, 100 - pos_pct - neg_pct) if total else 0

    overall = "Balanced (Neutral)"
    if neg_pct >= 40 or (neg_c > pos_c and neg_pct >= 30):
        overall = "Unfavorable (Negative)"
    elif pos_pct > neg_c and pos_pct >= 35:
        overall = "Favorable (Positive)"

    neg_clauses = [c for c in classifications if c["sentiment"] == "Negative"]
    clusters: Dict[str, Dict[str, Any]] = {}
    user_lower = user_context_text.lower()

    for nc in neg_clauses:
        matched = False
        for theme_name, pattern in UNFAVORABLE_THEMES:
            if pattern.search(nc["text"]):
                matched = True
                entry = clusters.setdefault(theme_name, {"count": 0, "sampleClause": nc["text"], "keywords": []})
                entry["count"] += 1
                for f in nc["topFeatures"]:
                    if f not in entry["keywords"]:
                        entry["keywords"].append(f)
        if not matched:
            fallback = "Auto-Renewal, Rescission & Administrative Restrictions"
            entry = clusters.setdefault(fallback, {"count": 0, "sampleClause": nc["text"], "keywords": []})
            entry["count"] += 1
            for f in nc["topFeatures"]:
                if f not in entry["keywords"]:
                    entry["keywords"].append(f)

    neg_total = len(neg_clauses) or 1
    unfavorable_list = []
    for cond, data in clusters.items():
        kws = data["keywords"][:5]
        if "Pre-Existing" in cond and re.search(r"diabetes|asthma|hypertension", user_lower):
            kws = ["Affects your history", *kws[:5]]
        unfavorable_list.append({
            "condition": cond,
            "count": data["count"],
            "sharePct": round((data["count"] / neg_total) * 100),
            "sampleClause": data["sampleClause"][:320],
            "topKeywords": kws,
        })
    unfavorable_list.sort(key=lambda x: x["count"], reverse=True)

    return {
        "modelName": f"TF-IDF (Unigrams + Bigrams) + Multinomial Naive Bayes ({'scikit-learn' if HAS_SKLEARN else 'stdlib'}, alpha=1.0)",
        "vocabularySize": len(_MODEL.feature_names),
        "trainingCorpusSize": len(TRAINING_CORPUS),
        "totalClausesAnalyzed": total,
        "positiveCount": pos_c,
        "neutralCount": neu_c,
        "negativeCount": neg_c,
        "positivePct": pos_pct,
        "neutralPct": neu_pct,
        "negativePct": neg_pct,
        "overallSentiment": overall,
        "mostCommonUnfavorableConditions": unfavorable_list,
        "clauseClassifications": classifications,
    }


# ---------------------------------------------------------------------------
# 4. US Plan Costs, Medical History & Financial Background Parsers
# ---------------------------------------------------------------------------

def _parse_dollar(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        val = float(re.sub(r"[$,\s]", "", raw))
        return val if val >= 0 else None
    except ValueError:
        return None


def extract_plan_costs_and_type(policy_text: str) -> Tuple[str, bool, Dict[str, Any]]:
    lower = policy_text.lower()
    if re.search(r"short-?term\s+limited\s+duration|not\s+aca-compliant", lower):
        plan_type = "Short-Term Medical"
    elif re.search(r"medicare|medigap|part\s+[abcd]", lower):
        plan_type = "Medicare / Medigap"
    elif re.search(r"hdhp|high\s+deductible\s+health\s+plan|hsa-eligible", lower):
        plan_type = "HDHP + HSA"
    elif re.search(r"\bhmo\b|health\s+maintenance\s+organization", lower):
        plan_type = "HMO / Managed Care"
    elif re.search(r"marketplace|aca\s+silver|aca\s+gold|aca\s+bronze|affordable\s+care\s+act", lower):
        plan_type = "ACA Marketplace"
    elif re.search(r"\bppo\b|\bepo\b|preferred\s+provider", lower):
        plan_type = "Employer PPO / EPO"
    elif re.search(r"dental|vision\s+plan", lower):
        plan_type = "Dental / Vision"
    else:
        plan_type = "General US Health Plan"

    aca_compliant = (
        not re.search(r"not\s+aca-compliant", lower)
        and plan_type != "Short-Term Medical"
        and bool(
            re.search(r"affordable\s+care\s+act|\baca\b|essential\s+health\s+benefits|no\s+waiting\s+period|no\s+surprises\s+act", lower)
            or re.search(r"pre-?existing\s+conditions?[^.\n]{0,80}covered\s+from\s+day\s+one", lower)
        )
    )

    prem_m = re.search(r"monthly\s+premium[^$\n]{0,30}\$([\d,]+(?:\.\d{2})?)", policy_text, re.I) or re.search(
        r"\$([\d,]+(?:\.\d{2})?)\s*(?:per\s+month|/mo|monthly\s+premium)", policy_text, re.I
    )
    ded_m = re.search(r"annual\s+deductible[^$\n]{0,40}\$([\d,]+(?:\.\d{2})?)", policy_text, re.I) or re.search(
        r"\$([\d,]+(?:\.\d{2})?)\s*(?:individual\s+)?(?:in-network\s+)?deductible", policy_text, re.I
    )
    oop_m = re.search(r"out-?of-?pocket\s+max(?:imum)?[^$\n]{0,40}\$([\d,]+(?:\.\d{2})?)", policy_text, re.I) or re.search(
        r"\$([\d,]+(?:\.\d{2})?)\s*(?:individual\s+)?out-?of-?pocket", policy_text, re.I
    )
    pcp_m = re.search(r"primary\s+care[^.\n$]{0,60}(\$\d+[^\n.;]*copay|\d+%)", policy_text, re.I)
    spec_m = re.search(r"specialist\s+visits?[^.\n$]{0,60}(\$\d+[^\n.;]*copay|\d+%)", policy_text, re.I)
    coins_m = re.search(r"(\d+%)\s*coinsurance", policy_text, re.I)

    network_type = "Not specified"
    if re.search(r"\bppo\b", policy_text, re.I):
        network_type = "PPO (In & Out-of-Network)"
    elif re.search(r"\bepo\b", policy_text, re.I):
        network_type = "EPO (In-Network Only)"
    elif re.search(r"\bhmo\b", policy_text, re.I):
        network_type = "HMO (Referral Required)"
    elif re.search(r"zero\s+out-of-network", policy_text, re.I):
        network_type = "Closed Network (No Out-of-Network)"

    costs = {
        "monthlyPremium": _parse_dollar(prem_m.group(1) if prem_m else None),
        "annualDeductible": _parse_dollar(ded_m.group(1) if ded_m else None),
        "outOfPocketMax": _parse_dollar(oop_m.group(1) if oop_m else None),
        "primaryCareCopay": pcp_m.group(1).strip() if pcp_m else "Not specified",
        "specialistCopay": spec_m.group(1).strip() if spec_m else "Not specified",
        "coinsuranceRate": coins_m.group(1).strip() if coins_m else "Not specified",
        "networkType": network_type,
    }
    return plan_type, aca_compliant, costs


def parse_medical_history(medical_text: str) -> Dict[str, Any]:
    if not medical_text.strip():
        return {
            "hasMedicalHistory": False,
            "conditions": [],
            "medications": [],
            "anticipatedCare": [],
            "sourceSummary": "No medical history document provided.",
        }
    lower = medical_text.lower()
    cond_catalog = [
        ("Type 2 Diabetes Mellitus", r"type\s*2\s*diabetes|diabetes\s+mellitus|\bdiabetes\b"),
        ("Type 1 Diabetes", r"type\s*1\s*diabetes"),
        ("Asthma", r"\basthma\b"),
        ("Hypertension (High Blood Pressure)", r"hypertension|high\s+blood\s+pressure"),
        ("Seasonal Allergic Rhinitis / Allergies", r"allergic\s+rhinitis|\ballerg"),
        ("Anxiety / Depression / Mental Health", r"depression|anxiety|mental\s+health|bipolar|adhd|therapy"),
        ("Heart / Cardiovascular Disease", r"cardiovascular|heart\s+disease|coronary|arrhythmia"),
        ("Cancer / Oncology History", r"cancer|oncology|carcinoma|chemotherapy"),
        ("Pregnancy / Maternity Care", r"pregnan|maternity|prenatal|obstetric"),
    ]
    med_catalog = [
        ("Insulin (Glargine / Rapid-Acting)", r"\binsulin\b"),
        ("Metformin", r"\bmetformin\b"),
        ("Albuterol HFA Inhaler", r"\balbuterol\b|inhaler"),
        ("Specialty Biologics", r"\bbiologic|humira|dupixent|ozempic|semaglutide\b"),
        ("Blood Pressure / Statin Medication", r"lisinopril|losartan|atorvastatin|statin|amlodipine"),
    ]
    care_catalog = [
        ("Endocrinologist Specialist Visits", r"endocrinolog"),
        ("Diagnostic Lab Work (HbA1c / Metabolic Panel)", r"hba1c|lab\s+work|blood\s+work|metabolic\s+panel|screening"),
        ("Outpatient Mental Health Therapy", r"mental\s+health\s+therapy|telehealth\s+sessions|counseling"),
        ("Preventive Wellness Exam & Immunizations", r"preventive|wellness\s+exam|immunization|flu"),
        ("MRI / Diagnostic Imaging", r"\bmri\b|ct\s+scan|imaging|radiology"),
    ]

    conditions = [lbl for lbl, pat in cond_catalog if re.search(pat, lower)]
    medications = [lbl for lbl, pat in med_catalog if re.search(pat, lower)]
    care = [lbl for lbl, pat in care_catalog if re.search(pat, lower)]

    return {
        "hasMedicalHistory": True,
        "conditions": conditions,
        "medications": medications,
        "anticipatedCare": care,
        "sourceSummary": f"Extracted {len(conditions)} condition(s), {len(medications)} active medication(s), and {len(care)} anticipated care need(s).",
    }


def parse_financial_background(financial_text: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = overrides or {}
    if not financial_text.strip() and not any(overrides.values()):
        return {
            "hasFinancialBackground": False,
            "annualIncome": None,
            "monthlyNetIncome": None,
            "monthlyObligations": None,
            "liquidSavings": None,
            "householdSize": 1,
            "sourceSummary": "No financial background document provided.",
        }

    ann_m = re.search(r"(?:annual\s+(?:gross\s+)?income|w-?2\s+(?:wages|income)|gross\s+pay)[^$\n]{0,30}\$([\d,]+(?:\.\d{2})?)", financial_text, re.I)
    net_m = re.search(r"(?:monthly\s+net\s+(?:take-?home\s+)?pay|monthly\s+income)[^$\n]{0,30}\$([\d,]+(?:\.\d{2})?)", financial_text, re.I)
    obl_m = re.search(r"(?:monthly\s+(?:fixed\s+)?obligations|monthly\s+expenses)[^$\n]{0,55}\$([\d,]+(?:\.\d{2})?)", financial_text, re.I)
    sav_m = re.search(r"(?:savings|hsa\s+balance|emergency\s+fund)[^$\n]{0,40}\$([\d,]+(?:\.\d{2})?)", financial_text, re.I)
    hh_m = re.search(r"household\s+size\s*[:\s]*(\d{1,2})", financial_text, re.I)

    annual_income = overrides.get("annualIncome") or _parse_dollar(ann_m.group(1) if ann_m else None)
    monthly_net = overrides.get("monthlyNetIncome") or _parse_dollar(net_m.group(1) if net_m else None)
    if monthly_net is None and annual_income is not None:
        monthly_net = round((annual_income * 0.76) / 12)
    monthly_obl = overrides.get("monthlyObligations") if overrides.get("monthlyObligations") is not None else _parse_dollar(obl_m.group(1) if obl_m else None)
    savings = overrides.get("liquidSavings") if overrides.get("liquidSavings") is not None else _parse_dollar(sav_m.group(1) if sav_m else None)
    hh_size = int(overrides.get("householdSize") or (hh_m.group(1) if hh_m else 1))

    parts = []
    if annual_income is not None:
        parts.append(f"Annual Gross: ${annual_income:,.0f}")
    if monthly_net is not None:
        parts.append(f"Monthly Net: ${monthly_net:,.0f}")
    if monthly_obl is not None:
        parts.append(f"Fixed Obligations: ${monthly_obl:,.0f}/mo")
    if savings is not None:
        parts.append(f"Savings/HSA: ${savings:,.0f}")
    parts.append(f"Household Size: {hh_size}")

    return {
        "hasFinancialBackground": any(v is not None for v in (annual_income, monthly_net, savings)),
        "annualIncome": annual_income,
        "monthlyNetIncome": monthly_net,
        "monthlyObligations": monthly_obl,
        "liquidSavings": savings,
        "householdSize": hh_size,
        "sourceSummary": " · ".join(parts),
    }


# ---------------------------------------------------------------------------
# 5. Affordability & Medical Eligibility Decision Evaluators
# ---------------------------------------------------------------------------

def evaluate_affordability(costs: Dict[str, Any], fin: Dict[str, Any], is_health_policy: bool) -> Dict[str, Any]:
    monthly_prem = costs["monthlyPremium"]
    annual_prem = round(monthly_prem * 12, 2) if monthly_prem is not None else None
    worst_case = (
        round(annual_prem + (costs["outOfPocketMax"] or costs["annualDeductible"] or 0), 2)
        if annual_prem is not None
        else None
    )

    if not is_health_policy or not fin["hasFinancialBackground"] or monthly_prem is None:
        return {
            "verdict": "Financial Info Needed",
            "canAfford": False,
            "shouldBuy": False,
            "estimatedMonthlyPremium": monthly_prem,
            "estimatedAnnualPremium": annual_prem,
            "worstCaseAnnualCost": worst_case,
            "premiumPercentOfIncome": None,
            "acaAffordabilityThresholdPct": 8.5,
            "deductibleToSavingsRatio": None,
            "rationale": "Provide a valid health policy with premium terms plus household income/savings to evaluate affordability.",
            "budgetBreakdown": [],
        }

    gross = fin["annualIncome"] or round(((fin["monthlyNetIncome"] or 0) * 12) / 0.76)
    prem_pct = round(((annual_prem or 0) / gross) * 100, 1) if gross > 0 else None
    m_net = fin["monthlyNetIncome"] or round((gross * 0.76) / 12)
    disposable = (m_net - fin["monthlyObligations"]) if fin["monthlyObligations"] is not None else m_net * 0.3
    ded_ratio = (
        round(costs["annualDeductible"] / fin["liquidSavings"], 2)
        if costs["annualDeductible"] is not None and fin["liquidSavings"]
        else None
    )

    fpl = 15650 + max(0, fin["householdSize"] - 1) * 5500
    fpl_pct = round((gross / fpl) * 100)
    breakdown = [
        f"Monthly Premium: ${monthly_prem:,.0f}/mo (${annual_prem:,.0f}/yr)",
        f"Income Share vs. ACA 8.5% Benchmark: {prem_pct}% of ${gross:,.0f} gross income",
        f"Estimated Monthly Disposable Margin: ${disposable:,.0f}/mo (Remaining: ${disposable - monthly_prem:,.0f}/mo)",
        f"2026 FPL Estimate: ~{fpl_pct}% FPL (Household of {fin['householdSize']})",
    ]
    if ded_ratio is not None:
        breakdown.append(f"Deductible Liquidity Coverage: ${costs['annualDeductible']:,.0f} vs ${fin['liquidSavings']:,.0f} Savings/HSA ({ded_ratio}x)")

    meets_aca = prem_pct is not None and prem_pct <= 8.5
    if meets_aca and (disposable - monthly_prem) >= 250 and (ded_ratio is None or ded_ratio <= 1.15):
        verdict, ok = "Affordable — Recommended to Buy", True
        rationale = f"Premium (${monthly_prem:,.0f}/mo, {prem_pct}% of gross income) meets the 8.5% ACA threshold with healthy cash flow and savings coverage."
    elif (disposable - monthly_prem) >= 50 and (prem_pct is None or prem_pct <= 12.0):
        verdict, ok = "Borderline — Tight Monthly Budget", True
        rationale = f"Premium fits monthly cash flow (${disposable - monthly_prem:,.0f}/mo remaining), though high deductible ratio ({ded_ratio}x savings) creates risk."
    else:
        verdict, ok = "Unaffordable — High Financial Strain", False
        rationale = f"Premium (${monthly_prem:,.0f}/mo) strains monthly budget or exceeds ACA affordability limits."

    return {
        "verdict": verdict,
        "canAfford": ok,
        "shouldBuy": ok,
        "estimatedMonthlyPremium": monthly_prem,
        "estimatedAnnualPremium": annual_prem,
        "worstCaseAnnualCost": worst_case,
        "premiumPercentOfIncome": prem_pct,
        "acaAffordabilityThresholdPct": 8.5,
        "deductibleToSavingsRatio": ded_ratio,
        "rationale": rationale,
        "budgetBreakdown": breakdown,
    }


def evaluate_medical_eligibility(
    policy_text: str, aca_compliant: bool, costs: Dict[str, Any], med: Dict[str, Any], is_health_policy: bool
) -> Dict[str, Any]:
    if not is_health_policy or not med["hasMedicalHistory"]:
        return {
            "verdict": "Medical History Needed",
            "shouldApply": False,
            "coverageHistorySummary": "Provide both a health insurance policy and medical history document.",
            "conditionMatches": [],
            "recommendationReasons": [],
        }

    lower = policy_text.lower()
    has_pre_ex_excl = bool(re.search(r"pre-?existing\s+condition\s+exclusion|strictly\s+excluded\s+for\s+the\s+entire", lower))
    excl_rx = bool(re.search(r"prescription\s+(?:drugs?|medications?)\s+(?:are\s+)?(?:not\s+covered|excluded)", lower))
    excl_mh = bool(re.search(r"mental\s+health[^.\n]{0,90}excluded", lower))
    excl_prev = bool(re.search(r"preventive\s+wellness[^.\n]{0,90}excluded", lower))
    has_wait = bool(re.search(r"\d+[-\s]+month\s+waiting\s+period", lower))
    has_insulin_cap = bool(re.search(r"insulin[^.\n]{0,80}\$35", lower))
    has_pa_penalty = bool(re.search(r"50%\s+benefit\s+penalty", lower))

    matches: List[Dict[str, str]] = []
    for cond in med["conditions"]:
        if has_pre_ex_excl or ("mental" in cond.lower() and excl_mh):
            matches.append({
                "conditionOrNeed": cond,
                "coverageStatus": "Excluded / Waiting Period",
                "estimatedCostShare": "100% Member Paid",
                "policyReference": "Pre-Existing Exclusion Clause",
                "explanation": f"Policy explicitly excludes pre-existing conditions or {cond}.",
            })
        elif aca_compliant:
            matches.append({
                "conditionOrNeed": cond,
                "coverageStatus": "Covered",
                "estimatedCostShare": costs["specialistCopay"],
                "policyReference": "ACA Chronic Condition Coverage",
                "explanation": f"ACA-compliant coverage covers {cond} from day one with zero waiting period.",
            })
        else:
            matches.append({
                "conditionOrNeed": cond,
                "coverageStatus": "Not Specified",
                "estimatedCostShare": costs["coinsuranceRate"],
                "policyReference": "General Medical Benefits",
                "explanation": "Verify network and pre-existing coverage rules.",
            })

    for rx in med["medications"]:
        if excl_rx:
            matches.append({
                "conditionOrNeed": f"Rx: {rx}",
                "coverageStatus": "Excluded / Waiting Period",
                "estimatedCostShare": "100% Out-of-Pocket",
                "policyReference": "Prescription Exclusion",
                "explanation": f"Outpatient prescriptions including {rx} are excluded.",
            })
        elif "insulin" in rx.lower() and has_insulin_cap:
            matches.append({
                "conditionOrNeed": f"Rx: {rx}",
                "coverageStatus": "Covered",
                "estimatedCostShare": "$35 / 30-day cap",
                "policyReference": "Formulary Tier 2",
                "explanation": "Protected by $35 monthly insulin cap.",
            })
        else:
            matches.append({
                "conditionOrNeed": f"Rx: {rx}",
                "coverageStatus": "Covered" if re.search(r"formulary|tier\s*1", lower) else "Not Specified",
                "estimatedCostShare": "Formulary Copay",
                "policyReference": "Pharmacy Schedule",
                "explanation": "Standard outpatient pharmacy formulary rules apply.",
            })

    for care in med["anticipatedCare"]:
        if "preventive" in care.lower():
            matches.append({
                "conditionOrNeed": care,
                "coverageStatus": "Excluded / Waiting Period" if excl_prev else "Covered",
                "estimatedCostShare": "100% Member Paid" if excl_prev else "$0 In-Network",
                "policyReference": "Preventive Care",
                "explanation": "Excluded under short-term plan." if excl_prev else "100% covered under ACA preventive rules.",
            })
        elif "lab" in care.lower() and "balance billing" in lower:
            matches.append({
                "conditionOrNeed": care,
                "coverageStatus": "Covered with Restrictions",
                "estimatedCostShare": "In-Network Covered / OON Risk",
                "policyReference": "Diagnostic Lab Rules",
                "explanation": "Use in-network labs to avoid out-of-network balance billing.",
            })
        elif ("mri" in care.lower() or "surgery" in care.lower()) and (has_pa_penalty or has_wait):
            matches.append({
                "conditionOrNeed": care,
                "coverageStatus": "Excluded / Waiting Period" if has_wait else "Covered with Restrictions",
                "estimatedCostShare": "50% Penalty if Unapproved",
                "policyReference": "Prior Authorization",
                "explanation": "Requires prior authorization or waiting period.",
            })
        else:
            matches.append({
                "conditionOrNeed": care,
                "coverageStatus": "Excluded / Waiting Period" if has_pre_ex_excl else "Covered",
                "estimatedCostShare": costs["specialistCopay"],
                "policyReference": "Medical Benefits",
                "explanation": "Excluded if linked to pre-existing condition." if has_pre_ex_excl else "Covered subject to plan cost-sharing.",
            })

    excl_cnt = sum(1 for m in matches if m["coverageStatus"] == "Excluded / Waiting Period")
    restr_cnt = sum(1 for m in matches if m["coverageStatus"] == "Covered with Restrictions")
    cov_cnt = sum(1 for m in matches if m["coverageStatus"] == "Covered")

    if not aca_compliant or excl_cnt >= 2 or (has_pre_ex_excl and med["conditions"]):
        verdict, should_apply = "Do Not Apply — Critical Exclusions or Gaps", False
    elif restr_cnt > 0 or excl_cnt == 1:
        verdict, should_apply = "Apply with Caution — Verify Prior Auth & Network", True
    else:
        verdict, should_apply = "Apply — Strong Medical Fit", True

    reasons = [
        "ACA-compliant coverage guarantees day-one pre-existing condition protection."
        if aca_compliant
        else "Policy is NOT ACA-compliant Minimum Essential Coverage.",
        f"{cov_cnt} covered, {restr_cnt} restricted, {excl_cnt} excluded out of {len(matches)} matched needs.",
    ]

    return {
        "verdict": verdict,
        "shouldApply": should_apply,
        "coverageHistorySummary": f"Matched {len(med['conditions'])} condition(s), {len(med['medications'])} Rx, {len(med['anticipatedCare'])} care item(s): {cov_cnt} Covered, {restr_cnt} Restricted, {excl_cnt} Excluded.",
        "conditionMatches": matches,
        "recommendationReasons": reasons,
    }


def analyze_health_insurance_and_sentiment(
    policy_text: str,
    is_health_policy: bool,
    parsed_clauses: List[Dict[str, Any]],
    medical_text: str = "",
    financial_text: str = "",
    financial_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan_type, aca_compliant, costs = extract_plan_costs_and_type(policy_text)
    med_profile = parse_medical_history(medical_text)
    fin_profile = parse_financial_background(financial_text, financial_overrides)
    affordability = evaluate_affordability(costs, fin_profile, is_health_policy)
    eligibility = evaluate_medical_eligibility(policy_text, aca_compliant, costs, med_profile, is_health_policy)
    ml_sentiment = run_ml_sentiment_analysis(policy_text, parsed_clauses, f"{medical_text}\n{financial_text}")

    return {
        "planType": plan_type if is_health_policy else "Non-Insurance Document",
        "acaCompliant": aca_compliant if is_health_policy else False,
        "extractedPlanCosts": costs,
        "medicalProfile": med_profile,
        "financialProfile": fin_profile,
        "affordability": affordability,
        "eligibility": eligibility,
        "mlSentiment": ml_sentiment,
    }
