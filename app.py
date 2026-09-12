import os

import pandas as pd
import streamlit as st

from recommender import DECISIONS, ReturnIndex, log_feedback, recommend

st.set_page_config(page_title="Returns Decision Assistant", layout="wide")

DATA_PATH = "data/sample_returns.csv"
FEEDBACK_PATH = "data/feedback_log.csv"

st.title("Returns Decision Assistant (MVP)")
st.caption(
    "Enter a new return case. The assistant checks hard policy rules, retrieves similar "
    "past cases, and recommends a decision for you to confirm or override."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "ANTHROPIC_API_KEY is not set. Hard-policy-rule cases will still work, "
        "but cases that need judgment will fail when calling Claude. "
        "Set the env var and restart: `export ANTHROPIC_API_KEY=sk-...`"
    )

with st.sidebar:
    st.header("Historical data")
    uploaded = st.file_uploader("Upload your own returns CSV", type=["csv"])
    data_path = uploaded if uploaded is not None else DATA_PATH
    st.caption(
        "Expected columns: sku, category, reason_code, days_since_purchase, "
        "item_condition, order_value, customer_return_count, proof_of_purchase, decision"
    )

@st.cache_resource(show_spinner="Building similarity index...")
def load_index(path):
    return ReturnIndex.from_csv(path)

index = load_index(data_path)
df = index.df

st.subheader("New return case")
col1, col2, col3 = st.columns(3)
with col1:
    sku = st.text_input("SKU", value="SKU-NEW-01")
    category = st.selectbox("Category", sorted(df["category"].unique()))
    reason_code = st.selectbox("Reason code", sorted(df["reason_code"].unique()))
with col2:
    days_since_purchase = st.number_input("Days since purchase", min_value=0, value=10)
    item_condition = st.selectbox("Item condition", sorted(df["item_condition"].unique()))
    order_value = st.number_input("Order value ($)", min_value=0.0, value=50.0, step=1.0)
with col3:
    customer_return_count = st.number_input("Customer's prior returns", min_value=0, value=0)
    proof_of_purchase = st.selectbox("Proof of purchase", ["yes", "no"])

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

if st.button("Get recommendation", type="primary"):
    with st.spinner("Checking policy rules and similar cases..."):
        try:
            rec, similar_df = recommend(new_case, index)
            st.session_state["rec"] = rec
            st.session_state["similar_df"] = similar_df
            st.session_state["new_case"] = new_case
        except Exception as e:
            st.error(f"Failed to get recommendation: {e}")

if "rec" in st.session_state:
    rec = st.session_state["rec"]
    similar_df = st.session_state["similar_df"]

    st.divider()
    st.subheader("Recommendation")

    badge = {"policy_rule": "Hard policy rule", "llm": "AI judgment call"}.get(rec.get("source"), rec.get("source"))
    st.markdown(f"**Decision:** `{rec['decision']}`  |  **Confidence:** {rec['confidence']}  |  **Basis:** {badge}")
    st.write(rec["rationale"])

    st.subheader("Similar past cases used")
    st.dataframe(similar_df, use_container_width=True)

    st.subheader("Your decision")
    final = st.selectbox("Confirm or override the final decision", DECISIONS, index=DECISIONS.index(rec["decision"]))
    if st.button("Confirm and log decision"):
        log_feedback(st.session_state["new_case"], rec, final, FEEDBACK_PATH)
        st.success(f"Logged decision '{final}' to {FEEDBACK_PATH}.")
        del st.session_state["rec"]

st.divider()
with st.expander("View historical data"):
    st.dataframe(df, use_container_width=True)

if os.path.exists(FEEDBACK_PATH):
    with st.expander("View feedback log (agreement rate)"):
        fb = pd.read_csv(FEEDBACK_PATH)
        agreement = (~fb["overridden"]).mean() if len(fb) else 0
        st.metric("AI/human agreement rate", f"{agreement:.0%}")
        st.dataframe(fb, use_container_width=True)
