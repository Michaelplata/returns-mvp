import os

import pandas as pd
import streamlit as st

from recommender import DECISIONS, ReturnIndex, log_feedback, recommend

st.set_page_config(page_title="Asystent Decyzji Zwrotów", layout="wide")

DATA_PATH = "data/sample_returns.csv"
FEEDBACK_PATH = "data/feedback_log.csv"

st.title("Asystent Decyzji Zwrotów (MVP)")
st.caption(
    "Wprowadź nowy przypadek zwrotu. Asystent sprawdza twarde reguły polityki, wyszukuje podobne "
    "poprzednie przypadki i rekomenduje decyzję do potwierdzenia lub zmiany."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "Klucz ANTHROPIC_API_KEY nie jest ustawiony. Przypadki z twardymi regułami polityki nadal będą działać, "
        "ale przypadki wymagające oceny nie zadziałają. "
        "Ustaw zmienną środowiskową i uruchom ponownie: `export ANTHROPIC_API_KEY=sk-...`"
    )

with st.sidebar:
    st.header("Dane historyczne")
    uploaded = st.file_uploader("Wgraj własny plik CSV ze zwrotami", type=["csv"])
    data_path = uploaded if uploaded is not None else DATA_PATH
    st.caption(
        "Wymagane kolumny: sku, category, reason_code, days_since_purchase, "
        "item_condition, order_value, customer_return_count, proof_of_purchase, decision"
    )

@st.cache_resource(show_spinner="Budowanie indeksu podobieństwa...")
def load_index(path):
    return ReturnIndex.from_csv(path)

index = load_index(data_path)
df = index.df

st.subheader("Nowy przypadek zwrotu")
col1, col2, col3 = st.columns(3)
with col1:
    sku = st.text_input("SKU", value="SKU-NEW-01")
    category = st.selectbox("Kategoria", sorted(df["category"].unique()))
    reason_code = st.selectbox("Kod przyczyny", sorted(df["reason_code"].unique()))
with col2:
    days_since_purchase = st.number_input("Dni od zakupu", min_value=0, value=10)
    item_condition = st.selectbox("Stan produktu", sorted(df["item_condition"].unique()))
    order_value = st.number_input("Wartość zamówienia (zł)", min_value=0.0, value=50.0, step=1.0)
with col3:
    customer_return_count = st.number_input("Poprzednie zwroty klienta", min_value=0, value=0)
    proof_of_purchase = st.selectbox("Dowód zakupu", ["yes", "no"])

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
        log_feedback(st.session_state["new_case"], rec, final, FEEDBACK_PATH)
        st.success(f"Zapisano decyzję '{final}' do {FEEDBACK_PATH}.")
        del st.session_state["rec"]

st.divider()
with st.expander("Zobacz dane historyczne"):
    st.dataframe(df, use_container_width=True)

if os.path.exists(FEEDBACK_PATH):
    with st.expander("Zobacz log opinii (wskaźnik zgodności)"):
        fb = pd.read_csv(FEEDBACK_PATH)
        agreement = (~fb["overridden"]).mean() if len(fb) else 0
        st.metric("Wskaźnik zgodności AI/człowiek", f"{agreement:.0%}")
        st.dataframe(fb, use_container_width=True)
