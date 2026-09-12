import os

import numpy as np
import pandas as pd
import streamlit as st

from recommender import DECISIONS, ReturnIndex, log_feedback, recommend

st.set_page_config(page_title="Asystent Decyzji Zwrotów", layout="wide")

DATA_PATH = "data/sample_returns.csv"
FEEDBACK_PATH = "data/feedback_log.csv"


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
    """Cache uploaded index in session state, keyed by file_id so it rebuilds only on new upload."""
    file_id = file.file_id
    if st.session_state.get("_upload_id") != file_id:
        file.seek(0)
        st.session_state["_upload_id"] = file_id
        st.session_state["_upload_index"] = ReturnIndex.from_csv(file)
        # Reset form and demo state when a new file is loaded
        for key in ["form_category", "form_reason", "form_condition", "form_proof", "demo_cases", "rec"]:
            st.session_state.pop(key, None)
    return st.session_state["_upload_index"]


# ── Sidebar — part 1: data upload ─────────────────────────────────────────────

with st.sidebar:
    st.header("Dane historyczne")
    uploaded = st.file_uploader("Wgraj własny plik CSV ze zwrotami", type=["csv"])
    if uploaded is not None:
        st.info("Dane są przetwarzane wyłącznie w pamięci i nie są nigdzie zapisywane.")
    st.caption(
        "Wymagane kolumny: sku, category, reason_code, days_since_purchase, "
        "item_condition, order_value, customer_return_count, proof_of_purchase, decision"
    )

using_upload = uploaded is not None
index = _load_uploaded_index(uploaded) if using_upload else _load_default_index()
df = index.df


# ── Sidebar — part 2: sample case generator ───────────────────────────────────

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
    n_cases = st.slider("Liczba przypadków", 3, 20, 8)
    if st.button("Generuj przypadki demo"):
        st.session_state["demo_cases"] = _generate_sample_cases(df, n_cases)

    if "demo_cases" in st.session_state:
        demo_df = st.session_state["demo_cases"]
        st.dataframe(
            demo_df[["category", "reason_code", "days_since_purchase", "item_condition"]],
            use_container_width=True,
        )
        selected_idx = st.selectbox(
            "Wybierz przypadek",
            range(len(demo_df)),
            format_func=lambda i: f"#{i+1}: {demo_df.iloc[i]['category']} — {demo_df.iloc[i]['reason_code']}",
        )
        if st.button("Załaduj do formularza"):
            case = demo_df.iloc[selected_idx]
            st.session_state["form_sku"] = case["sku"]
            st.session_state["form_category"] = case["category"]
            st.session_state["form_reason"] = case["reason_code"]
            st.session_state["form_days"] = int(case["days_since_purchase"])
            st.session_state["form_condition"] = case["item_condition"]
            st.session_state["form_value"] = float(case["order_value"])
            st.session_state["form_returns"] = int(case["customer_return_count"])
            st.session_state["form_proof"] = case["proof_of_purchase"]
            st.rerun()


# ── Main form ─────────────────────────────────────────────────────────────────

st.title("Asystent Decyzji Zwrotów (MVP)")
st.caption(
    "Wprowadź nowy przypadek zwrotu. Asystent sprawdza twarde reguły polityki, wyszukuje podobne "
    "poprzednie przypadki i rekomenduje decyzję do potwierdzenia lub zmiany."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "Klucz ANTHROPIC_API_KEY nie jest ustawiony. Przypadki z twardymi regułami polityki nadal będą działać, "
        "ale przypadki wymagające oceny nie zadziałają."
    )

# Initialize form session state defaults
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

st.subheader("Nowy przypadek zwrotu")
col1, col2, col3 = st.columns(3)
with col1:
    sku = st.text_input("SKU", key="form_sku")
    category = st.selectbox("Kategoria", category_options, key="form_category")
    reason_code = st.selectbox("Kod przyczyny", reason_options, key="form_reason")
with col2:
    days_since_purchase = st.number_input("Dni od zakupu", min_value=0, key="form_days")
    item_condition = st.selectbox("Stan produktu", condition_options, key="form_condition")
    order_value = st.number_input("Wartość zamówienia (zł)", min_value=0.0, step=1.0, key="form_value")
with col3:
    customer_return_count = st.number_input("Poprzednie zwroty klienta", min_value=0, key="form_returns")
    proof_of_purchase = st.selectbox("Dowód zakupu", ["yes", "no"], key="form_proof")

new_case = {
    "sku": sku,
    "category": category,
    "reason_code": reason_code,
    "days_since_purchase": days_since_purchase,
    "item_condition": item_condition,
    "order_value": order_value,
    "customer_return_count": customer_return_count,
    "proof_of_purchase": proof_of_purchase,
}

if st.button("Uzyskaj rekomendację", type="primary"):
    with st.spinner("Sprawdzanie reguł polityki i podobnych przypadków..."):
        try:
            rec, similar_df = recommend(new_case, index)
            st.session_state["rec"] = rec
            st.session_state["similar_df"] = similar_df
            st.session_state["new_case"] = new_case
        except Exception as e:
            st.error(f"Nie udało się uzyskać rekomendacji: {e}")


# ── Recommendation ────────────────────────────────────────────────────────────

if "rec" in st.session_state:
    rec = st.session_state["rec"]
    similar_df = st.session_state["similar_df"]

    st.divider()
    st.subheader("Rekomendacja")

    badge = {"policy_rule": "Twarda reguła polityki", "llm": "Ocena AI"}.get(rec.get("source"), rec.get("source"))
    st.markdown(f"**Decyzja:** `{rec['decision']}`  |  **Pewność:** {rec['confidence']}  |  **Podstawa:** {badge}")
    st.write(rec["rationale"])

    st.subheader("Podobne poprzednie przypadki")
    st.dataframe(similar_df, use_container_width=True)

    st.subheader("Twoja decyzja")
    final = st.selectbox("Potwierdź lub zmień ostateczną decyzję", DECISIONS, index=DECISIONS.index(rec["decision"]))
    if st.button("Potwierdź i zapisz decyzję"):
        if using_upload:
            row = {
                **st.session_state["new_case"],
                "ai_recommended_decision": rec.get("decision"),
                "ai_confidence": rec.get("confidence"),
                "ai_source": rec.get("source"),
                "human_decision": final,
                "overridden": rec.get("decision") != final,
            }
            st.session_state.setdefault("feedback_rows", []).append(row)
        else:
            log_feedback(st.session_state["new_case"], rec, final, FEEDBACK_PATH)
        st.success(f"Zapisano decyzję '{final}'.")
        del st.session_state["rec"]


# ── Historical data & feedback ────────────────────────────────────────────────

st.divider()
with st.expander("Zobacz dane historyczne"):
    st.dataframe(df, use_container_width=True)

if using_upload:
    rows = st.session_state.get("feedback_rows", [])
    if rows:
        with st.expander(f"Decyzje z tej sesji ({len(rows)})"):
            fb = pd.DataFrame(rows)
            agreement = (~fb["overridden"]).mean()
            st.metric("Wskaźnik zgodności AI/człowiek", f"{agreement:.0%}")
            st.dataframe(fb, use_container_width=True)
            st.download_button(
                "Pobierz log decyzji",
                fb.to_csv(index=False).encode(),
                "decyzje_sesja.csv",
                "text/csv",
            )
elif os.path.exists(FEEDBACK_PATH):
    with st.expander("Zobacz log opinii (wskaźnik zgodności)"):
        fb = pd.read_csv(FEEDBACK_PATH)
        agreement = (~fb["overridden"]).mean() if len(fb) else 0
        st.metric("Wskaźnik zgodności AI/człowiek", f"{agreement:.0%}")
        st.dataframe(fb, use_container_width=True)
