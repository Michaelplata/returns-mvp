import os

import numpy as np
import pandas as pd
import streamlit as st

from recommender import DECISIONS, ReturnIndex, log_feedback, recommend

st.set_page_config(page_title="Asystent Decyzji Zwrotów", layout="wide")

DATA_PATH = "data/sample_returns.csv"
FEEDBACK_PATH = "data/feedback_log.csv"

DECISION_LABELS = {
    "approve_refund": "Zwrot pieniędzy",
    "approve_exchange": "Wymiana produktu",
    "approve_store_credit": "Kredyt sklepowy",
    "approve_repair": "Naprawa",
    "deny": "Odmowa",
}

CONFIDENCE_LABELS = {
    "high": "Wysoka",
    "medium": "Srednia",
    "low": "Niska",
}

SOURCE_LABELS = {
    "policy_rule": "Twarda reguła polityki",
    "llm": "Ocena AI",
}

COLUMN_LABELS = {
    "return_id": "ID zwrotu",
    "sku": "SKU",
    "category": "Kategoria",
    "reason_code": "Powód zwrotu",
    "days_since_purchase": "Dni od zakupu",
    "item_condition": "Stan produktu",
    "order_value": "Wartość (zł)",
    "customer_return_count": "Poprzednie zwroty",
    "proof_of_purchase": "Dowód zakupu",
    "decision": "Decyzja",
    "similarity": "Podobieństwo",
    "ai_recommended_decision": "Decyzja AI",
    "ai_confidence": "Pewność AI",
    "ai_source": "Źródło",
    "human_decision": "Decyzja człowieka",
    "overridden": "Zmieniona",
}


def _polish_df(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in df.columns})
    if "Decyzja" in renamed.columns:
        renamed["Decyzja"] = renamed["Decyzja"].map(lambda d: DECISION_LABELS.get(d, d))
    if "Decyzja AI" in renamed.columns:
        renamed["Decyzja AI"] = renamed["Decyzja AI"].map(lambda d: DECISION_LABELS.get(d, d))
    if "Dowód zakupu" in renamed.columns:
        renamed["Dowód zakupu"] = renamed["Dowód zakupu"].map(lambda x: "Tak" if x == "yes" else ("Nie" if x == "no" else x))
    return renamed

st.markdown("""
<style>
div[data-testid="stButton"] button[kind="primary"] {
    height: 3.2rem;
    font-size: 1.05rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Password gate ─────────────────────────────────────────────────────────────

def _check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    expected = st.secrets.get("DEMO_PASSWORD", None) or os.getenv("DEMO_PASSWORD")
    if not expected:
        st.session_state["authenticated"] = True
        return True
    st.title("Asystent Decyzji Zwrotów")
    pwd = st.text_input("Hasło demo", type="password")
    if st.button("Wejdź"):
        if pwd == expected:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Nieprawidłowe hasło.")
    return False

if not _check_password():
    st.stop()


# ── Index loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Budowanie indeksu podobieństwa...")
def _load_default_index() -> ReturnIndex:
    return ReturnIndex.from_csv(DATA_PATH)


def _load_uploaded_index(file) -> ReturnIndex:
    file_id = file.file_id
    if st.session_state.get("_upload_id") != file_id:
        file.seek(0)
        st.session_state["_upload_id"] = file_id
        st.session_state["_upload_index"] = ReturnIndex.from_csv(file)
        for key in ["form_category", "form_reason", "form_condition", "form_proof", "demo_cases", "rec", "show_override"]:
            st.session_state.pop(key, None)
    return st.session_state["_upload_index"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Dane historyczne")
    uploaded = st.file_uploader("Wgraj plik CSV ze zwrotami", type=["csv"])
    if uploaded is not None:
        st.info("Dane przetwarzane tylko w pamięci — nigdzie niezapisywane.")
    st.caption(
        "Wymagane kolumny: sku, category, reason_code, days_since_purchase, "
        "item_condition, order_value, customer_return_count, proof_of_purchase, decision"
    )

using_upload = uploaded is not None
index = _load_uploaded_index(uploaded) if using_upload else _load_default_index()
df = index.df


def _generate_sample_cases(source_df: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng()
    rows = []
    for i in range(n):
        rows.append({
            "sku": f"SKU-DEMO-{1000 + i}",
            "category": str(rng.choice(source_df["category"].values)),
            "reason_code": str(rng.choice(source_df["reason_code"].values)),
            "days_since_purchase": int(rng.choice(source_df["days_since_purchase"].values)),
            "item_condition": str(rng.choice(source_df["item_condition"].values)),
            "order_value": round(float(rng.choice(source_df["order_value"].values)), 2),
            "customer_return_count": int(rng.choice(source_df["customer_return_count"].values)),
            "proof_of_purchase": str(rng.choice(source_df["proof_of_purchase"].values)),
        })
    return pd.DataFrame(rows)


with st.sidebar:
    st.divider()
    st.header("Generator przypadków demo")
    st.caption("Generuje realistyczne przypadki na podstawie wgranych danych.")
    n_cases = st.slider("Liczba przypadków", 3, 20, 8)
    if st.button("Generuj", use_container_width=True):
        st.session_state["demo_cases"] = _generate_sample_cases(df, n_cases)

    if "demo_cases" in st.session_state:
        demo_df = st.session_state["demo_cases"]
        selected_idx = st.selectbox(
            "Wybierz przypadek do załadowania",
            range(len(demo_df)),
            format_func=lambda i: f"#{i+1}  {demo_df.iloc[i]['category']} — {demo_df.iloc[i]['reason_code']}",
        )
        if st.button("Zaladuj do formularza", use_container_width=True):
            case = demo_df.iloc[selected_idx]
            st.session_state["form_sku"] = case["sku"]
            st.session_state["form_category"] = case["category"]
            st.session_state["form_reason"] = case["reason_code"]
            st.session_state["form_days"] = int(case["days_since_purchase"])
            st.session_state["form_condition"] = case["item_condition"]
            st.session_state["form_value"] = float(case["order_value"])
            st.session_state["form_returns"] = int(case["customer_return_count"])
            st.session_state["form_proof"] = case["proof_of_purchase"]
            st.session_state.pop("rec", None)
            st.session_state.pop("show_override", None)
            st.rerun()
        st.dataframe(
            demo_df[["category", "reason_code", "days_since_purchase", "item_condition"]],
            use_container_width=True,
            hide_index=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Asystent Decyzji Zwrotów")
st.caption("Sprawdź reguły polityki, znajdź podobne przypadki i uzyskaj rekomendację decyzji.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("Klucz ANTHROPIC_API_KEY nie jest ustawiony. Przypadki wymagające oceny AI nie zadziałają.")

# ── Section 1: Form ───────────────────────────────────────────────────────────

st.subheader("1. Dane przypadku zwrotu")

category_options = sorted(df["category"].unique())
reason_options = sorted(df["reason_code"].unique())
condition_options = sorted(df["item_condition"].unique())

for key, default in [
    ("form_sku", "SKU-NEW-01"),
    ("form_days", 10),
    ("form_value", 50.0),
    ("form_returns", 0),
    ("form_proof", "yes"),
    ("form_category", category_options[0]),
    ("form_reason", reason_options[0]),
    ("form_condition", condition_options[0]),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input("SKU", key="form_sku", help="Unikalny identyfikator produktu")
    st.selectbox("Kategoria", category_options, key="form_category")
    st.selectbox(
        "Powód zwrotu", reason_options, key="form_reason",
        help="Główny powód zgłoszenia zwrotu przez klienta",
    )
with col2:
    st.number_input(
        "Dni od zakupu", min_value=0, key="form_days",
        help="Ile dni minęło od daty zakupu",
    )
    st.selectbox(
        "Stan produktu", condition_options, key="form_condition",
        help="Aktualny stan fizyczny zwracanego produktu",
    )
    st.number_input("Wartość zamówienia (zł)", min_value=0.0, step=1.0, key="form_value")
with col3:
    st.number_input(
        "Poprzednie zwroty klienta", min_value=0, key="form_returns",
        help="Liczba zwrotów tego klienta w historii",
    )
    st.selectbox(
        "Dowód zakupu", ["yes", "no"], key="form_proof",
        format_func=lambda x: "Tak" if x == "yes" else "Nie",
        help="Czy klient dostarczył paragon lub fakturę?",
    )

new_case = {
    "sku": st.session_state["form_sku"],
    "category": st.session_state["form_category"],
    "reason_code": st.session_state["form_reason"],
    "days_since_purchase": st.session_state["form_days"],
    "item_condition": st.session_state["form_condition"],
    "order_value": st.session_state["form_value"],
    "customer_return_count": st.session_state["form_returns"],
    "proof_of_purchase": st.session_state["form_proof"],
}

st.button(
    "Uzyskaj rekomendację",
    type="primary",
    key="get_rec_btn",
    use_container_width=False,
)

if st.session_state.get("get_rec_btn"):
    with st.spinner("Analizuję przypadek..."):
        try:
            rec, similar_df = recommend(new_case, index)
            st.session_state["rec"] = rec
            st.session_state["similar_df"] = similar_df
            st.session_state["new_case"] = new_case
            st.session_state.pop("show_override", None)
        except Exception as e:
            st.error(f"Nie udało się uzyskać rekomendacji: {e}")


# ── Section 2: Recommendation ─────────────────────────────────────────────────

if "rec" in st.session_state:
    rec = st.session_state["rec"]
    similar_df = st.session_state["similar_df"]
    is_approve = rec["decision"].startswith("approve")

    st.divider()
    st.subheader("2. Rekomendacja")

    m1, m2, m3 = st.columns(3)
    m1.metric("Decyzja", DECISION_LABELS.get(rec["decision"], rec["decision"]))
    m2.metric("Pewność", CONFIDENCE_LABELS.get(rec["confidence"], rec["confidence"]))
    m3.metric("Podstawa", SOURCE_LABELS.get(rec["source"], rec["source"]))

    if is_approve:
        st.success(rec["rationale"])
    else:
        st.error(rec["rationale"])

    with st.expander(f"Podobne przypadki historyczne ({len(similar_df)})"):
        st.dataframe(_polish_df(similar_df), use_container_width=True, hide_index=True)

    # ── Section 3: Confirm ────────────────────────────────────────────────────

    st.divider()
    st.subheader("3. Twoja decyzja")

    def _save_decision(decision: str):
        if using_upload:
            row = {
                **st.session_state["new_case"],
                "ai_recommended_decision": rec.get("decision"),
                "ai_confidence": rec.get("confidence"),
                "ai_source": rec.get("source"),
                "human_decision": decision,
                "overridden": rec.get("decision") != decision,
            }
            st.session_state.setdefault("feedback_rows", []).append(row)
        else:
            log_feedback(st.session_state["new_case"], rec, decision, FEEDBACK_PATH)
        st.session_state.pop("rec", None)
        st.session_state.pop("show_override", None)

    col_confirm, col_change = st.columns([3, 1])
    with col_confirm:
        label = DECISION_LABELS.get(rec["decision"], rec["decision"])
        if st.button(f"Potwierdź: {label}", type="primary", use_container_width=True):
            _save_decision(rec["decision"])
            st.success("Decyzja zapisana.")
            st.rerun()
    with col_change:
        if st.button("Zmień decyzję", use_container_width=True):
            st.session_state["show_override"] = not st.session_state.get("show_override", False)

    if st.session_state.get("show_override"):
        override = st.selectbox(
            "Wybierz inną decyzję",
            DECISIONS,
            format_func=lambda d: DECISION_LABELS.get(d, d),
            index=DECISIONS.index(rec["decision"]),
        )
        if st.button("Zapisz zmienioną decyzję", type="primary"):
            _save_decision(override)
            st.success(f"Zapisano zmienioną decyzję: {DECISION_LABELS.get(override, override)}")
            st.rerun()


# ── Footer: data & feedback ───────────────────────────────────────────────────

st.divider()
with st.expander("Dane historyczne"):
    st.dataframe(_polish_df(df), use_container_width=True, hide_index=True)

if using_upload:
    rows = st.session_state.get("feedback_rows", [])
    if rows:
        with st.expander(f"Decyzje z tej sesji ({len(rows)})"):
            fb = pd.DataFrame(rows)
            agreement = (~fb["overridden"]).mean()
            st.metric("Zgodność AI / człowiek", f"{agreement:.0%}")
            st.dataframe(_polish_df(fb), use_container_width=True, hide_index=True)
            st.download_button(
                "Pobierz log decyzji",
                fb.to_csv(index=False).encode(),
                "decyzje_sesja.csv",
                "text/csv",
            )
elif os.path.exists(FEEDBACK_PATH):
    with st.expander("Log decyzji"):
        fb = pd.read_csv(FEEDBACK_PATH)
        agreement = (~fb["overridden"]).mean() if len(fb) else 0
        st.metric("Zgodność AI / człowiek", f"{agreement:.0%}")
        st.dataframe(_polish_df(fb), use_container_width=True, hide_index=True)
