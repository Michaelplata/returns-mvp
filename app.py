import os

import numpy as np
import pandas as pd
import streamlit as st

import db
from recommender import DECISIONS, ReturnIndex, recommend, log_feedback

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
CONFIDENCE_LABELS = {"high": "Wysoka", "medium": "Srednia", "low": "Niska"}
SOURCE_LABELS = {"policy_rule": "Twarda reguła polityki", "llm": "Ocena AI"}
COLUMN_LABELS = {
    "return_id": "ID zwrotu", "sku": "SKU", "category": "Kategoria",
    "reason_code": "Powód zwrotu", "days_since_purchase": "Dni od zakupu",
    "item_condition": "Stan produktu", "order_value": "Wartość (zł)",
    "customer_return_count": "Poprzednie zwroty", "proof_of_purchase": "Dowód zakupu",
    "decision": "Decyzja", "similarity": "Podobieństwo",
    "ai_recommended_decision": "Decyzja AI", "ai_confidence": "Pewność AI",
    "ai_source": "Źródło", "human_decision": "Decyzja człowieka",
    "overridden": "Zmieniona", "created_at": "Data", "company_name": "Firma",
}

st.markdown("""
<style>
div[data-testid="stButton"] button[kind="primary"] { height: 3.2rem; font-size: 1.05rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def _polish_df(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={k: v for k, v in COLUMN_LABELS.items() if k in frame.columns})
    if "Decyzja" in renamed.columns:
        renamed["Decyzja"] = renamed["Decyzja"].map(lambda d: DECISION_LABELS.get(d, d))
    if "Decyzja AI" in renamed.columns:
        renamed["Decyzja AI"] = renamed["Decyzja AI"].map(lambda d: DECISION_LABELS.get(d, d))
    if "Decyzja człowieka" in renamed.columns:
        renamed["Decyzja człowieka"] = renamed["Decyzja człowieka"].map(lambda d: DECISION_LABELS.get(d, d))
    if "Dowód zakupu" in renamed.columns:
        renamed["Dowód zakupu"] = renamed["Dowód zakupu"].map(
            lambda x: "Tak" if x == "yes" else ("Nie" if x == "no" else x)
        )
    return renamed


# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_auth() -> str | None:
    """Returns 'admin', 'agent', or None."""
    if st.session_state.get("role"):
        return st.session_state["role"]

    admin_pwd = st.secrets.get("ADMIN_PASSWORD", None) or os.getenv("ADMIN_PASSWORD")
    demo_pwd = st.secrets.get("DEMO_PASSWORD", None) or os.getenv("DEMO_PASSWORD")

    if not admin_pwd and not demo_pwd:
        st.session_state["role"] = "agent"
        return "agent"

    st.title("Asystent Decyzji Zwrotów")
    pwd = st.text_input("Hasło", type="password")
    if st.button("Wejdź", type="primary"):
        if admin_pwd and pwd == admin_pwd:
            st.session_state["role"] = "admin"
            st.rerun()
        elif demo_pwd and pwd == demo_pwd:
            st.session_state["role"] = "agent"
            st.rerun()
        else:
            st.error("Nieprawidłowe hasło.")
    return None


# ── Shared: index caching + generator ────────────────────────────────────────

def _get_index(company_id: str) -> ReturnIndex | None:
    key = f"idx_{company_id}"
    if key not in st.session_state:
        frame = db.get_return_cases(company_id)
        st.session_state[key] = ReturnIndex.from_df(frame) if not frame.empty else None
    return st.session_state[key]


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


# ── Recommendation form (shared by agent + legacy) ────────────────────────────

def _render_form_and_recommendation(index: ReturnIndex, company_id: str | None):
    """Renders the case input form, recommendation, and confirm step.
    company_id=None signals legacy mode (decisions written to local CSV).
    """
    df = index.df
    category_options = sorted(df["category"].unique())
    reason_options = sorted(df["reason_code"].unique())
    condition_options = sorted(df["item_condition"].unique())

    for key, default in [
        ("form_sku", "SKU-NEW-01"), ("form_days", 10), ("form_value", 50.0),
        ("form_returns", 0), ("form_proof", "yes"),
        ("form_category", category_options[0]),
        ("form_reason", reason_options[0]),
        ("form_condition", condition_options[0]),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.subheader("1. Dane przypadku zwrotu")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("SKU", key="form_sku", help="Unikalny identyfikator produktu")
        st.selectbox("Kategoria", category_options, key="form_category")
        st.selectbox("Powód zwrotu", reason_options, key="form_reason",
                     help="Główny powód zgłoszenia zwrotu")
    with col2:
        st.number_input("Dni od zakupu", min_value=0, key="form_days",
                        help="Ile dni minęło od daty zakupu")
        st.selectbox("Stan produktu", condition_options, key="form_condition",
                     help="Aktualny stan fizyczny zwracanego produktu")
        st.number_input("Wartość zamówienia (zł)", min_value=0.0, step=1.0, key="form_value")
    with col3:
        st.number_input("Poprzednie zwroty klienta", min_value=0, key="form_returns",
                        help="Liczba wcześniejszych zwrotów tego klienta")
        st.selectbox("Dowód zakupu", ["yes", "no"], key="form_proof",
                     format_func=lambda x: "Tak" if x == "yes" else "Nie",
                     help="Czy klient dostarczył paragon lub fakturę?")

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

    if st.button("Uzyskaj rekomendację", type="primary"):
        with st.spinner("Analizuję przypadek..."):
            try:
                rec, similar_df = recommend(new_case, index)
                st.session_state["rec"] = rec
                st.session_state["similar_df"] = similar_df
                st.session_state["new_case"] = new_case
                st.session_state.pop("show_override", None)
            except Exception as e:
                st.error(f"Nie udało się uzyskać rekomendacji: {e}")

    if "rec" not in st.session_state:
        return

    rec = st.session_state["rec"]
    similar_df = st.session_state["similar_df"]
    is_approve = rec["decision"].startswith("approve")

    st.divider()
    st.subheader("2. Rekomendacja")
    m1, m2, m3 = st.columns(3)
    m1.metric("Decyzja", DECISION_LABELS.get(rec["decision"], rec["decision"]))
    m2.metric("Pewność", CONFIDENCE_LABELS.get(rec["confidence"], rec["confidence"]))
    m3.metric("Podstawa", SOURCE_LABELS.get(rec["source"], rec["source"]))
    (st.success if is_approve else st.error)(rec["rationale"])

    with st.expander(f"Podobne przypadki historyczne ({len(similar_df)})"):
        st.dataframe(_polish_df(similar_df), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3. Twoja decyzja")

    def _save(decision: str):
        if company_id:
            db.log_decision(company_id, st.session_state["new_case"], rec, decision)
        else:
            log_feedback(st.session_state["new_case"], rec, decision, FEEDBACK_PATH)
        st.session_state.pop("rec", None)
        st.session_state.pop("show_override", None)

    col_ok, col_change = st.columns([3, 1])
    with col_ok:
        label = DECISION_LABELS.get(rec["decision"], rec["decision"])
        if st.button(f"Potwierdź: {label}", type="primary", use_container_width=True):
            _save(rec["decision"])
            st.success("Decyzja zapisana.")
            st.rerun()
    with col_change:
        if st.button("Zmień decyzję", use_container_width=True):
            st.session_state["show_override"] = not st.session_state.get("show_override", False)

    if st.session_state.get("show_override"):
        override = st.selectbox("Wybierz inną decyzję", DECISIONS,
                                format_func=lambda d: DECISION_LABELS.get(d, d),
                                index=DECISIONS.index(rec["decision"]))
        if st.button("Zapisz zmienioną decyzję", type="primary"):
            _save(override)
            st.success(f"Zapisano: {DECISION_LABELS.get(override, override)}")
            st.rerun()


# ── Admin view ────────────────────────────────────────────────────────────────

def _render_admin():
    col_title, col_logout = st.columns([6, 1])
    col_title.title("Panel administracyjny")
    if col_logout.button("Wyloguj"):
        st.session_state.pop("role", None)
        st.rerun()

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Klucz ANTHROPIC_API_KEY nie jest ustawiony.")

    tab_companies, tab_decisions = st.tabs(["Firmy", "Wszystkie decyzje"])

    with tab_companies:
        stats = db.get_company_stats()
        if not stats.empty:
            st.dataframe(stats.drop(columns=["_id"]), use_container_width=True, hide_index=True)
        else:
            st.info("Brak firm. Dodaj pierwszą poniżej.")

        st.divider()

        with st.expander("Dodaj nową firmę"):
            new_name = st.text_input("Nazwa firmy", key="new_company_name")
            if st.button("Dodaj firmę", type="primary") and new_name.strip():
                db.add_company(new_name.strip())
                st.success(f"Dodano: {new_name}")
                st.rerun()

        companies = db.get_companies()
        if not companies:
            return

        st.subheader("Zarządzaj firmą")
        selected_name = st.selectbox("Wybierz firmę", [c["name"] for c in companies])
        company = next(c for c in companies if c["name"] == selected_name)

        col_upload, col_delete = st.columns([3, 1])
        with col_upload:
            csv_file = st.file_uploader(
                f"Wgraj dane historyczne dla: {selected_name}", type=["csv"],
                help="Plik zostanie zapisany w bazie danych i zastąpi poprzednie dane tej firmy."
            )
            if csv_file and st.button("Zapisz dane w bazie", type="primary"):
                with st.spinner("Przesyłanie..."):
                    frame = pd.read_csv(csv_file)
                    db.upload_return_cases(company["id"], frame)
                    st.session_state.pop(f"idx_{company['id']}", None)
                st.success(f"Wgrano {len(frame)} przypadków dla: {selected_name}")
                st.rerun()

        with col_delete:
            st.write("")
            st.write("")
            if st.button("Usuń firmę"):
                st.session_state["confirm_delete"] = company["id"]
            if st.session_state.get("confirm_delete") == company["id"]:
                st.warning(f"Usunąć {selected_name} wraz ze wszystkimi danymi?")
                col_yes, col_no = st.columns(2)
                if col_yes.button("Tak, usuń", type="primary"):
                    db.delete_company(company["id"])
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()
                if col_no.button("Anuluj"):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

    with tab_decisions:
        all_dec = db.get_all_decisions()
        if all_dec.empty:
            st.info("Brak decyzji w systemie.")
            return
        agreement = (~all_dec["overridden"]).mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Łączne decyzje", len(all_dec))
        m2.metric("Zgodność AI / człowiek", f"{agreement:.0%}")
        m3.metric("Firm", all_dec["company_name"].nunique() if "company_name" in all_dec.columns else "—")
        st.dataframe(_polish_df(all_dec), use_container_width=True, hide_index=True)


# ── Agent view ────────────────────────────────────────────────────────────────

def _render_agent():
    col_title, col_logout = st.columns([6, 1])
    col_title.title("Asystent Decyzji Zwrotów")
    if col_logout.button("Wyloguj"):
        st.session_state.pop("role", None)
        st.rerun()
    st.caption("Sprawdź reguły polityki, znajdź podobne przypadki i uzyskaj rekomendację decyzji.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Klucz ANTHROPIC_API_KEY nie jest ustawiony.")

    companies = db.get_companies()
    if not companies:
        st.warning("Brak firm w systemie. Skontaktuj się z administratorem.")
        return

    selected_name = st.selectbox("Firma", [c["name"] for c in companies])
    company_id = next(c["id"] for c in companies if c["name"] == selected_name)

    with st.spinner("Ładowanie danych..."):
        index = _get_index(company_id)

    if index is None:
        st.warning("Brak danych historycznych dla tej firmy. Administrator musi najpierw wgrać dane.")
        return

    # Sidebar generator
    with st.sidebar:
        st.header("Generator przypadków demo")
        st.caption("Generuje realistyczne przypadki na podstawie danych firmy.")
        n_cases = st.slider("Liczba przypadków", 3, 20, 8)
        if st.button("Generuj", use_container_width=True):
            st.session_state["demo_cases"] = _generate_sample_cases(index.df, n_cases)

        if "demo_cases" in st.session_state:
            demo_df = st.session_state["demo_cases"]
            selected_idx = st.selectbox(
                "Wybierz przypadek",
                range(len(demo_df)),
                format_func=lambda i: f"#{i+1}  {demo_df.iloc[i]['category']} — {demo_df.iloc[i]['reason_code']}",
            )
            if st.button("Załaduj do formularza", use_container_width=True):
                case = demo_df.iloc[selected_idx]
                st.session_state.update({
                    "form_sku": case["sku"],
                    "form_category": case["category"],
                    "form_reason": case["reason_code"],
                    "form_days": int(case["days_since_purchase"]),
                    "form_condition": case["item_condition"],
                    "form_value": float(case["order_value"]),
                    "form_returns": int(case["customer_return_count"]),
                    "form_proof": case["proof_of_purchase"],
                })
                st.session_state.pop("rec", None)
                st.rerun()
            st.dataframe(
                demo_df[["category", "reason_code", "days_since_purchase", "item_condition"]],
                use_container_width=True, hide_index=True,
            )

        st.divider()
        if st.button("Odśwież dane firmy", use_container_width=True):
            st.session_state.pop(f"idx_{company_id}", None)
            st.rerun()

    _render_form_and_recommendation(index, company_id)

    st.divider()
    with st.expander("Dane historyczne firmy"):
        st.dataframe(_polish_df(index.df), use_container_width=True, hide_index=True)

    decisions = db.get_decisions(company_id)
    if not decisions.empty:
        with st.expander(f"Decyzje tej firmy ({len(decisions)})"):
            agreement = (~decisions["overridden"]).mean()
            st.metric("Zgodność AI / człowiek", f"{agreement:.0%}")
            st.dataframe(_polish_df(decisions), use_container_width=True, hide_index=True)


# ── Legacy view (no Supabase configured) ─────────────────────────────────────

def _render_legacy():
    col_title, col_logout = st.columns([6, 1])
    col_title.title("Asystent Decyzji Zwrotów")
    if col_logout.button("Wyloguj"):
        st.session_state.pop("role", None)
        st.rerun()
    st.caption("Sprawdź reguły polityki, znajdź podobne przypadki i uzyskaj rekomendację decyzji.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("Klucz ANTHROPIC_API_KEY nie jest ustawiony.")

    @st.cache_resource(show_spinner="Budowanie indeksu...")
    def _load_default():
        return ReturnIndex.from_csv(DATA_PATH)

    def _load_uploaded(file):
        fid = file.file_id
        if st.session_state.get("_upload_id") != fid:
            file.seek(0)
            st.session_state["_upload_id"] = fid
            st.session_state["_upload_index"] = ReturnIndex.from_csv(file)
            for k in ["form_category", "form_reason", "form_condition", "form_proof", "demo_cases", "rec"]:
                st.session_state.pop(k, None)
        return st.session_state["_upload_index"]

    with st.sidebar:
        st.header("Dane historyczne")
        uploaded = st.file_uploader("Wgraj plik CSV ze zwrotami", type=["csv"])
        if uploaded:
            st.info("Dane przetwarzane tylko w pamięci — nigdzie niezapisywane.")
        st.caption("Wymagane kolumny: sku, category, reason_code, days_since_purchase, "
                   "item_condition, order_value, customer_return_count, proof_of_purchase, decision")

    index = _load_uploaded(uploaded) if uploaded else _load_default()
    using_upload = uploaded is not None

    with st.sidebar:
        st.divider()
        st.header("Generator przypadków demo")
        n_cases = st.slider("Liczba przypadków", 3, 20, 8)
        if st.button("Generuj", use_container_width=True):
            st.session_state["demo_cases"] = _generate_sample_cases(index.df, n_cases)
        if "demo_cases" in st.session_state:
            demo_df = st.session_state["demo_cases"]
            selected_idx = st.selectbox(
                "Wybierz przypadek", range(len(demo_df)),
                format_func=lambda i: f"#{i+1}  {demo_df.iloc[i]['category']} — {demo_df.iloc[i]['reason_code']}",
            )
            if st.button("Załaduj do formularza", use_container_width=True):
                case = demo_df.iloc[selected_idx]
                st.session_state.update({
                    "form_sku": case["sku"], "form_category": case["category"],
                    "form_reason": case["reason_code"], "form_days": int(case["days_since_purchase"]),
                    "form_condition": case["item_condition"], "form_value": float(case["order_value"]),
                    "form_returns": int(case["customer_return_count"]), "form_proof": case["proof_of_purchase"],
                })
                st.session_state.pop("rec", None)
                st.rerun()
            st.dataframe(
                demo_df[["category", "reason_code", "days_since_purchase", "item_condition"]],
                use_container_width=True, hide_index=True,
            )

    _render_form_and_recommendation(index, company_id=None)

    st.divider()
    with st.expander("Dane historyczne"):
        st.dataframe(_polish_df(index.df), use_container_width=True, hide_index=True)

    if using_upload:
        rows = st.session_state.get("feedback_rows", [])
        if rows:
            with st.expander(f"Decyzje z tej sesji ({len(rows)})"):
                fb = pd.DataFrame(rows)
                st.metric("Zgodność AI / człowiek", f"{(~fb['overridden']).mean():.0%}")
                st.dataframe(_polish_df(fb), use_container_width=True, hide_index=True)
                st.download_button("Pobierz log decyzji", fb.to_csv(index=False).encode(),
                                   "decyzje_sesja.csv", "text/csv")
    elif os.path.exists(FEEDBACK_PATH):
        with st.expander("Log decyzji"):
            fb = pd.read_csv(FEEDBACK_PATH)
            agreement = (~fb["overridden"]).mean() if len(fb) else 0
            st.metric("Zgodność AI / człowiek", f"{agreement:.0%}")
            st.dataframe(_polish_df(fb), use_container_width=True, hide_index=True)


# ── Routing ───────────────────────────────────────────────────────────────────

role = _check_auth()
if role is None:
    st.stop()
elif role == "admin" and db.is_configured():
    _render_admin()
elif db.is_configured():
    _render_agent()
else:
    _render_legacy()
