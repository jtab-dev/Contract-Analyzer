from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from contract_engine import analyze_contract_deterministic, extract_text_from_inputs
from firebase_store import list_user_policies, query_gleif_registry, save_policy_record

load_dotenv()

app = FastAPI(
    title="ClauseVault — US Health Insurance ML & Policy Intelligence (Python)",
    version="1.0.0",
)


class FilePayload(BaseModel):
    fileName: str
    mimeType: str
    base64Data: str


class FinancialOverrides(BaseModel):
    annualIncome: Optional[float] = None
    monthlyNetIncome: Optional[float] = None
    monthlyObligations: Optional[float] = None
    liquidSavings: Optional[float] = None
    householdSize: Optional[int] = 1


class AnalyzeRequest(BaseModel):
    files: Optional[List[FilePayload]] = None
    rawText: Optional[str] = None
    medicalFiles: Optional[List[FilePayload]] = None
    medicalText: Optional[str] = ""
    financialFiles: Optional[List[FilePayload]] = None
    financialText: Optional[str] = ""
    financialOverrides: Optional[FinancialOverrides] = None
    customRules: List[Dict[str, Any]] = Field(default_factory=list)


class RegistryVerifyRequest(BaseModel):
    organizationName: str


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {"ok": True, "engine": "python-tfidf-multinomial-nb"}


@app.post("/api/contracts/analyze")
def analyze_policy(req: AnalyzeRequest) -> Dict[str, Any]:
    if not req.files and not (req.rawText and req.rawText.strip()):
        raise HTTPException(status_code=400, detail="Provide a policy file or rawText.")

    try:
        policy_text = extract_text_from_inputs(
            [f.model_dump() for f in req.files] if req.files else None,
            req.rawText,
        )
        med_text = (
            extract_text_from_inputs(
                [f.model_dump() for f in req.medicalFiles],
                req.medicalText,
            )
            if req.medicalFiles
            else (req.medicalText or "")
        )
        fin_text = (
            extract_text_from_inputs(
                [f.model_dump() for f in req.financialFiles],
                req.financialText,
            )
            if req.financialFiles
            else (req.financialText or "")
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    analysis = analyze_contract_deterministic(
        extracted_text=policy_text,
        custom_rules=req.customRules[:50],
        medical_text=med_text,
        financial_text=fin_text,
        financial_overrides=req.financialOverrides.model_dump() if req.financialOverrides else None,
    )

    org_parties = [p for p in analysis["parties"] if p.get("entityType") == "Organization"][:3]
    analysis["registryMatches"] = [query_gleif_registry(p["name"]) for p in org_parties]
    return {"analysis": analysis}


@app.post("/api/registry/verify")
def verify_organization(req: RegistryVerifyRequest) -> Dict[str, Any]:
    if not req.organizationName.strip():
        raise HTTPException(status_code=400, detail="organizationName is required.")
    return {"match": query_gleif_registry(req.organizationName)}


@app.post("/api/archive/{record_id}")
def archive_policy(record_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    saved = save_policy_record(record_id, payload)
    return {"saved": saved, "id": record_id}


@app.get("/api/archive")
def get_archive(ownerId: str) -> Dict[str, Any]:
    return {"contracts": list_user_policies(ownerId)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
