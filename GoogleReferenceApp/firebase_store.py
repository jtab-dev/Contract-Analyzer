from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import json
import urllib.request


def query_gleif_registry(entity_name: str) -> Dict[str, Any]:
    clean_name = (entity_name or "").strip()[:140]
    if not clean_name:
        return {
            "queriedName": entity_name,
            "registrySource": "GLEIF Global LEI Registry",
            "status": "No Match Found",
            "legalName": "",
            "registryId": "",
            "jurisdiction": "",
            "entityStatus": "",
            "note": "Empty entity name provided.",
        }

    url = f"https://api.gleif.org/api/v1/lei-records?filter[entity.legalName]={quote(clean_name)}&page[size]=2"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        top = (payload.get("data") or [None])[0]
        if not top:
            return {
                "queriedName": clean_name,
                "registrySource": "GLEIF Global LEI Registry",
                "status": "No Match Found",
                "legalName": "",
                "registryId": "",
                "jurisdiction": "",
                "entityStatus": "",
                "note": "No matching Legal Entity Identifier (LEI) found in GLEIF.",
            }
        attrs = top.get("attributes") or {}
        entity = attrs.get("entity") or {}
        reg = attrs.get("registration") or {}
        return {
            "queriedName": clean_name,
            "registrySource": "GLEIF Global LEI Registry",
            "status": "Matched",
            "legalName": (entity.get("legalName") or {}).get("name") or clean_name,
            "registryId": top.get("id") or "",
            "jurisdiction": entity.get("jurisdiction") or "Unknown",
            "entityStatus": entity.get("status") or reg.get("status") or "Unknown",
            "note": "Matched entity record in GLEIF LEI registry.",
        }
    except Exception:
        return {
            "queriedName": clean_name,
            "registrySource": "GLEIF Global LEI Registry",
            "status": "Lookup Unavailable",
            "legalName": "",
            "registryId": "",
            "jurisdiction": "",
            "entityStatus": "",
            "note": "Live external registry lookup unavailable from this environment.",
        }


def get_firestore_client() -> Optional[Any]:
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
    if not cred_path or not os.path.exists(cred_path):
        return None
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import credentials, firestore  # type: ignore

        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        db_id = os.getenv("FIRESTORE_DATABASE_ID")
        return firestore.client(database_id=db_id) if db_id else firestore.client()
    except Exception:
        return None


def save_policy_record(record_id: str, data: Dict[str, Any]) -> bool:
    db = get_firestore_client()
    if db is None:
        return False
    db.collection("contracts").document(record_id).set(data)
    return True


def list_user_policies(owner_id: str) -> List[Dict[str, Any]]:
    db = get_firestore_client()
    if db is None:
        return []
    docs = db.collection("contracts").where("ownerId", "==", owner_id).stream()
    return [{**d.to_dict(), "id": d.id} for d in docs]
