from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple

from ml_health_engine import analyze_health_insurance_and_sentiment


def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, int]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install 'pypdf' (pip install pypdf) to extract PDF text.") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p).strip(), len(reader.pages)


def extract_image_ocr_text(img_bytes: bytes) -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install 'Pillow' and 'pytesseract' (plus system tesseract) for image OCR.") from exc

    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img).strip()


def extract_text_from_inputs(
    files: Optional[List[Dict[str, str]]] = None,
    raw_text: Optional[str] = None,
) -> str:
    segments: List[str] = []
    for f in files or []:
        mime = (f.get("mimeType") or "").lower()
        b64 = re.sub(r"^data:[^;]+;base64,", "", f.get("base64Data") or "")
        raw = base64.b64decode(b64)
        if mime == "text/plain":
            segments.append(raw.decode("utf-8", errors="replace").strip())
        elif mime == "application/pdf":
            text, page_count = extract_pdf_text(raw)
            if len(re.sub(r"\s+", "", text)) < 15:
                raise ValueError(
                    f'"{f.get("fileName", "document.pdf")}" ({page_count} page(s)) has no embedded text layer. Export pages as PNG/JPG for OCR.'
                )
            segments.append(text)
        elif mime.startswith("image/"):
            ocr_text = extract_image_ocr_text(raw)
            if ocr_text:
                segments.append(ocr_text)

    if raw_text and raw_text.strip():
        segments.append(raw_text.strip())

    combined = "\n\n".join(s for s in segments if s).strip()
    if not combined:
        raise ValueError("No readable text could be extracted from the uploaded inputs.")
    return combined[:120000]


def classify_document(text: str) -> Dict[str, Any]:
    lower = text.lower()
    first_lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 2][:6]
    raw_title = re.sub(r"\s+", " ", first_lines[0] if first_lines else "Uploaded Document")[:140]

    health_signals = [
        "health insurance", "summary of benefits", "monthly premium", "annual deductible",
        "out-of-pocket maximum", "coinsurance", "copay", "prior authorization",
        "pre-existing condition", "essential health benefits", "policyholder",
        "insurer", "formulary", "in-network", "out-of-network", "aca-compliant",
        "short-term limited duration", "medicare",
    ]
    contract_signals = [
        "agreement", "hereby agrees", "entered into", "in witness whereof",
        "terms and conditions", "termination", "governing law", "indemnify",
        "limitation of liability", "confidential information", "whereas",
        "obligations", "party", "parties", "policy", "effective date",
    ]
    non_contract_signals = [
        "tax invoice", "invoice #", "patient billing statement", "itemized charges",
        "subtotal", "total paid", "receipt", "paid in full", "amount due",
        "thank you for visiting", "thank you for your business",
    ]

    health_hits = sum(1 for s in health_signals if s in lower)
    contract_hits = sum(1 for s in contract_signals if s in lower)
    non_hits = sum(1 for s in non_contract_signals if s in lower)

    is_health = health_hits >= 4 and not (non_hits >= 2 and health_hits < 5)
    is_contract = is_health or (contract_hits >= 3 and not (non_hits >= 2 and contract_hits < 4))

    if not is_contract:
        doc_type = (
            "Medical Billing Statement / Invoice"
            if ("billing statement" in lower or "cpt" in lower)
            else "Invoice / Billing Statement"
            if "invoice" in lower
            else "Non-Insurance Document"
        )
        return {
            "isContract": False,
            "isHealthPolicy": False,
            "documentType": doc_type,
            "category": "Non-Insurance Document",
            "title": raw_title,
            "confidenceNote": f"Detected {non_hits} billing/non-policy indicators and only {health_hits} health policy terms.",
        }

    if is_health:
        health_rules = [
            ("Short-Term Medical", "Short-Term Limited Duration Health Plan (Non-ACA)", ["short-term limited duration", "not aca-compliant", "pre-existing condition exclusion", "lifetime maximum benefit"]),
            ("ACA Marketplace", "ACA Marketplace Health Insurance Policy", ["affordable care act", "aca", "marketplace", "silver 70", "gold 80", "bronze 60", "essential health benefits"]),
            ("HDHP + HSA", "High Deductible Health Plan (HDHP + HSA)", ["hdhp", "high deductible health plan", "hsa-eligible"]),
            ("HMO / Managed Care", "HMO Managed Care Health Plan", ["hmo", "health maintenance organization"]),
            ("Medicare / Medigap", "Medicare Advantage / Medigap Policy", ["medicare", "medigap"]),
            ("Employer PPO / EPO", "US PPO / EPO Health Insurance Policy", ["ppo", "epo", "preferred provider"]),
        ]
        best_cat, best_doc, best_score = "ACA Marketplace", "US Health Insurance Policy", 0
        for cat, doc_type, terms in health_rules:
            score = sum(1 for t in terms if t in lower)
            if score > best_score:
                best_cat, best_doc, best_score = cat, doc_type, score
        return {
            "isContract": True,
            "isHealthPolicy": True,
            "documentType": best_doc,
            "category": best_cat,
            "title": raw_title,
            "confidenceNote": f"Classified as {best_doc} ({health_hits} policy signals).",
        }

    return {
        "isContract": True,
        "isHealthPolicy": False,
        "documentType": "General Contract / Agreement",
        "category": "Other",
        "title": raw_title,
        "confidenceNote": f"Classified as general contract ({contract_hits} covenant signals).",
    }


def parse_clauses(text: str) -> List[Dict[str, str]]:
    clauses: List[Dict[str, str]] = []
    heading_re = re.compile(
        r"^(?:(ARTICLE|SECTION|EXHIBIT|SCHEDULE)\s+[A-Z0-9]+(?:\.[0-9]+)*|(\d+(?:\.\d+){0,2})[.)]?\s+([A-Z][A-Za-z0-9\s,&/\-]{2,70}))(?:\s*[—:\-]\s*(.*))?$"
    )
    current: Optional[Dict[str, Any]] = None

    def flush() -> None:
        if not current:
            return
        body = re.sub(r"\s+", " ", " ".join(current["bodyLines"])).strip()
        if not body and len(current["title"]) < 3:
            return
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", body) if s]
        clauses.append({
            "sectionNumber": current["sectionNumber"] or str(len(clauses) + 1),
            "title": current["title"][:140],
            "category": "Policy Provision",
            "plainSummary": " ".join(sentences[:2])[:400] or current["title"],
            "excerpt": body[:450],
        })

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^IN WITNESS WHEREOF", line, re.I):
            flush()
            current = None
            break
        m = heading_re.match(line)
        if m and len(line) < 180:
            flush()
            if m.group(1):
                parts = re.split(r"[—:\-]", line, maxsplit=1)
                current = {"sectionNumber": parts[0].strip()[:30], "title": (parts[1].strip() if len(parts) > 1 else parts[0].strip())[:120], "bodyLines": []}
            else:
                current = {"sectionNumber": m.group(2) or "", "title": (m.group(3) or "").strip(), "bodyLines": [m.group(4).strip()] if m.group(4) else []}
        elif current is not None:
            current["bodyLines"].append(line)
    flush()
    return clauses[:60]


def extract_parties(text: str) -> List[Dict[str, Any]]:
    parties: List[Dict[str, Any]] = []
    seen = set()
    role_re = re.compile(r"^(INSURER|POLICYHOLDER|SUBSCRIBER|CARRIER|LANDLORD|TENANT|PROVIDER|CUSTOMER|EMPLOYER|EMPLOYEE)\s*:\s*(.+)$", re.I | re.M)
    for m in role_re.finditer(text):
        role = m.group(1).strip()
        rest = m.group(2).strip()
        name = re.split(r"[(\n,]", rest)[0].strip()[:100]
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        is_org = bool(re.search(r"\b(Inc\.|LLC|Ltd\.|Corp\.|Plan|Indemnity|Company)\b", name, re.I))
        parties.append({
            "name": name,
            "role": role.capitalize(),
            "entityType": "Organization" if is_org else "Individual" if role.upper() in ("POLICYHOLDER", "SUBSCRIBER") else "Unknown",
            "addressOrIdentifier": "Address not specified" if "address not specified" in rest.lower() else "",
            "signed": "/s/" in text[m.start():m.start() + 250],
            "signatoryName": "",
            "signatoryDate": "",
        })
    return parties


def detect_risks(clauses: List[Dict[str, str]], full_text: str, custom_rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
    rules = [
        ("One-Sided Obligation", "High", "Pre-Existing Condition Exclusion or Waiting Period", [r"pre-?existing\s+condition\s+exclusion", r"\d+[-\s]+month\s+waiting\s+period"], "Non-ACA plans can deny claims for diabetes, asthma, or prior prescriptions."),
        ("Liability Limitation", "High", "Annual or Lifetime Benefit Dollar Cap", [r"lifetime\s+maximum\s+benefit[^.\n]{0,60}capped", r"exceeding\s+\$[\d,]+\s+are\s+excluded"], "ACA prohibits lifetime dollar caps on Essential Health Benefits."),
        ("Hidden Fee", "High", "Balance Billing & Out-of-Network Exposure", [r"subject\s+to\s+balance\s+billing", r"zero\s+out-?of-?network\s+coverage"], "Out-of-network providers can bill the full balance above allowed rates."),
        ("Early Termination Penalty", "High", "Prior Authorization 50% Penalty or Step Therapy", [r"50%\s+benefit\s+penalty", r"mandatory\s+step\s+therapy|fail-?first"], "Missing prior authorization or step therapy can cut benefits by 50%."),
        ("Auto-Renewal", "High", "Automatic Renewal Lock-In", [r"automatically\s+renew", r"auto-?renewal"], "Plan renews automatically unless cancelled in writing before the deadline."),
    ]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", full_text) if s.strip()]
    risks: List[Dict[str, str]] = []
    seen = set()

    for cat, sev, title, pats, why in rules:
        for s in sentences:
            if any(re.search(p, s, re.I) for p in pats) and title not in seen:
                seen.add(title)
                risks.append({
                    "category": cat,
                    "severity": sev,
                    "title": title,
                    "clauseRef": "Policy Section",
                    "excerpt": s[:500],
                    "whyItMatters": why,
                    "negotiationTip": "Compare against an ACA-compliant plan before enrolling.",
                })
    for cr in custom_rules or []:
        kws = [k.strip().lower() for k in cr.get("keywords", []) if k.strip()]
        for s in sentences:
            if any(k in s.lower() for k in kws) and cr.get("title") not in seen:
                seen.add(cr.get("title"))
                risks.append({
                    "category": cr.get("category", "Unclear Condition"),
                    "severity": cr.get("severity", "Medium"),
                    "title": f"{cr.get('title', 'Custom Rule')} (Database Rule)",
                    "clauseRef": "Policy Section",
                    "excerpt": s[:500],
                    "whyItMatters": cr.get("whyItMatters", "Flagged by custom database rule."),
                    "negotiationTip": "Review clause against custom policy rules.",
                })
    return risks[:40]


def detect_contradictions(full_text: str) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", full_text) if s.strip()]
    contradictions: List[Dict[str, str]] = []
    for s in sentences:
        nw = re.search(r"notwithstanding\s+(Section\s+[\d.]+)", s, re.I)
        if nw:
            contradictions.append({
                "category": "Clause Conflict",
                "severity": "Medium",
                "title": f"Section Override Against {nw.group(1)}",
                "sectionALabel": nw.group(1),
                "sectionAQuote": f"Base rule in {nw.group(1)}.",
                "sectionBLabel": "Override Clause",
                "sectionBQuote": s[:400],
                "explanation": f"Explicitly overrides {nw.group(1)} using 'Notwithstanding {nw.group(1)}'.",
                "recommendedResolution": f"Consolidate both exceptions directly inside {nw.group(1)}.",
            })
    return contradictions[:20]


def analyze_contract_deterministic(
    extracted_text: str,
    custom_rules: Optional[List[Dict[str, Any]]] = None,
    medical_text: str = "",
    financial_text: str = "",
    financial_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    classification = classify_document(extracted_text)
    clauses = parse_clauses(extracted_text) if classification["isContract"] else []
    parties = extract_parties(extracted_text)
    risks = detect_risks(clauses, extracted_text, custom_rules) if classification["isContract"] else []
    contradictions = detect_contradictions(extracted_text) if classification["isContract"] else []

    health_analysis = analyze_health_insurance_and_sentiment(
        policy_text=extracted_text,
        is_health_policy=classification["isHealthPolicy"],
        parsed_clauses=clauses,
        medical_text=medical_text,
        financial_text=financial_text,
        financial_overrides=financial_overrides,
    )

    return {
        "title": classification["title"],
        "isContract": classification["isContract"],
        "documentType": classification["documentType"],
        "category": classification["category"],
        "confidenceNote": classification["confidenceNote"],
        "extractedText": extracted_text,
        "parties": parties,
        "summary": {
            "overview": f"{classification['documentType']} with {len(clauses)} section(s) and {len(risks)} flagged risk(s).",
        },
        "clauses": clauses,
        "risks": risks,
        "contradictions": contradictions,
        "healthAnalysis": health_analysis,
    }
