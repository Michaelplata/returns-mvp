import os

import pandas as pd


def _get_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        except Exception:
            pass
    return url, key


def is_configured() -> bool:
    url, key = _get_config()
    return bool(url and key)


def _client():
    from supabase import create_client
    url, key = _get_config()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(url, key)


_CASE_COLUMNS = [
    "return_id", "sku", "category", "reason_code", "days_since_purchase",
    "item_condition", "order_value", "customer_return_count", "proof_of_purchase", "decision",
]


def get_companies() -> list[dict]:
    resp = _client().table("companies").select("id, name, created_at").order("name").execute()
    return resp.data or []


def add_company(name: str) -> dict:
    resp = _client().table("companies").insert({"name": name}).execute()
    return resp.data[0]


def delete_company(company_id: str):
    _client().table("companies").delete().eq("id", company_id).execute()


def get_return_cases(company_id: str) -> pd.DataFrame:
    resp = (
        _client()
        .table("return_cases")
        .select(", ".join(_CASE_COLUMNS))
        .eq("company_id", company_id)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame(columns=_CASE_COLUMNS)
    return pd.DataFrame(resp.data)


def upload_return_cases(company_id: str, df: pd.DataFrame):
    _client().table("return_cases").delete().eq("company_id", company_id).execute()
    upload_cols = [c for c in _CASE_COLUMNS if c in df.columns]
    rows = df[upload_cols].copy().to_dict(orient="records")
    for row in rows:
        row["company_id"] = company_id
    for i in range(0, len(rows), 100):
        _client().table("return_cases").insert(rows[i : i + 100]).execute()


def log_decision(company_id: str, new_case: dict, rec: dict, human_decision: str):
    case_keys = ("sku", "category", "reason_code", "days_since_purchase",
                 "item_condition", "order_value", "customer_return_count", "proof_of_purchase")
    row = {
        "company_id": company_id,
        **{k: new_case[k] for k in case_keys if k in new_case},
        "ai_recommended_decision": rec.get("decision"),
        "ai_confidence": rec.get("confidence"),
        "ai_source": rec.get("source"),
        "human_decision": human_decision,
        "overridden": rec.get("decision") != human_decision,
    }
    _client().table("decisions").insert(row).execute()


def get_decisions(company_id: str) -> pd.DataFrame:
    resp = (
        _client()
        .table("decisions")
        .select("*")
        .eq("company_id", company_id)
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def get_all_decisions() -> pd.DataFrame:
    resp = (
        _client()
        .table("decisions")
        .select("*, companies(name)")
        .order("created_at", desc=True)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    if "companies" in df.columns:
        df["company_name"] = df["companies"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else None
        )
        df = df.drop(columns=["companies"])
    return df


def get_company_stats() -> pd.DataFrame:
    companies = get_companies()
    if not companies:
        return pd.DataFrame()
    client = _client()
    rows = []
    for c in companies:
        cases = client.table("return_cases").select("id", count="exact").eq("company_id", c["id"]).execute()
        dec = client.table("decisions").select("overridden", count="exact").eq("company_id", c["id"]).execute()
        dec_data = dec.data or []
        overrides = sum(1 for d in dec_data if d.get("overridden"))
        rows.append({
            "Firma": c["name"],
            "_id": c["id"],
            "Przypadki hist.": cases.count or 0,
            "Decyzje AI": len(dec_data),
            "Zmienione": overrides,
        })
    return pd.DataFrame(rows)
