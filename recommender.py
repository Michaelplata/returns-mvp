"""
Core logic for the returns-decision assistant MVP.

Pipeline:
1. Load historical return cases from CSV.
2. Turn each case into a short text description and build a TF-IDF index
   (no external embedding API needed for the MVP -- swap for real embeddings later).
3. For a new case, retrieve the k most similar historical cases.
4. Apply hard policy rules first (fast, deterministic, no LLM needed).
5. If no hard rule fires, ask Claude to recommend a decision, grounded in the
   retrieved similar cases, and return a structured JSON response.
"""

import json
import os
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DECISIONS = [
    "approve_refund",
    "approve_exchange",
    "approve_store_credit",
    "approve_repair",
    "deny",
]

# --- Hard policy rules -----------------------------------------------------
# Keep these as plain code, not something the LLM has to infer every time.
# Edit RETURN_WINDOW_DAYS / CATEGORY_WINDOWS to match your real policy.

RETURN_WINDOW_DAYS = 90
CATEGORY_WINDOWS = {
    "Electronics": 90,
    "Footwear": 30,
    "Apparel": 45,
    "Home": 100,
}


def apply_hard_rules(new_case: dict) -> dict | None:
    """Return a decision dict if a hard rule fires, else None (fall through to the LLM)."""
    category = new_case.get("category")
    days = int(new_case.get("days_since_purchase", 0))
    window = CATEGORY_WINDOWS.get(category, RETURN_WINDOW_DAYS)

    if days > window:
        return {
            "decision": "deny",
            "confidence": "high",
            "rationale": f"Poza oknem zwrotu wynoszącym {window} dni dla kategorii {category} "
            f"(produkt zwrócony {days} dni po zakupie). Twarda reguła polityki, nie ocena.",
            "source": "policy_rule",
            "similar_cases": [],
        }

    if new_case.get("proof_of_purchase", "yes") == "no" and new_case.get("customer_return_count", 0) and int(new_case["customer_return_count"]) >= 3:
        return {
            "decision": "deny",
            "confidence": "medium",
            "rationale": "Brak dowodu zakupu i historia 3+ poprzednich zwrotów. "
            "Twarda reguła polityki dla potencjalnego wzorca nadużyć.",
            "source": "policy_rule",
            "similar_cases": [],
        }

    return None


# --- Retrieval ---------------------------------------------------------------

def _case_to_text(row: dict) -> str:
    return (
        f"category {row.get('category')} reason {row.get('reason_code')} "
        f"condition {row.get('item_condition')} days_since_purchase {row.get('days_since_purchase')} "
        f"order_value {row.get('order_value')} customer_return_count {row.get('customer_return_count')} "
        f"proof_of_purchase {row.get('proof_of_purchase')}"
    )


@dataclass
class ReturnIndex:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer = field(default=None)
    matrix: object = field(default=None)

    @classmethod
    def from_csv(cls, path: str) -> "ReturnIndex":
        df = pd.read_csv(path)
        idx = cls(df=df)
        idx._build()
        return idx

    def _build(self):
        texts = self.df.apply(lambda r: _case_to_text(r.to_dict()), axis=1)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(texts)

    def similar_cases(self, new_case: dict, k: int = 5) -> pd.DataFrame:
        query_text = _case_to_text(new_case)
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        result = self.df.iloc[top_idx].copy()
        result["similarity"] = sims[top_idx]
        return result.reset_index(drop=True)


# --- LLM recommendation -------------------------------------------------------

RECOMMEND_SYSTEM_PROMPT = """You are a returns-decision assistant for an e-commerce retailer.
You help a human agent decide how to resolve a product return by grounding your recommendation
in similar past cases and their outcomes.

You must respond with ONLY a JSON object, no other text, no markdown fences, with these fields:
{
  "decision": one of ["approve_refund", "approve_exchange", "approve_store_credit", "approve_repair", "deny"],
  "confidence": one of ["low", "medium", "high"],
  "rationale": a short (2-4 sentence) explanation citing which similar past case(s) informed the recommendation,
  "cases_used": a list of return_id strings from the similar cases you relied on most
}

Be conservative: if the similar cases disagree with each other, say so in the rationale and lower your confidence.
This recommendation will always be reviewed by a human before being finalized -- your job is to make that review fast and well-informed, not to make the final call yourself.
"""


def get_llm_recommendation(new_case: dict, similar_df: pd.DataFrame) -> dict:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic") from e

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    similar_cases_payload = similar_df.drop(columns=["similarity"]).to_dict(orient="records")

    user_prompt = f"""New return case to decide:
{json.dumps(new_case, indent=2)}

Most similar historical cases (with their final decisions):
{json.dumps(similar_cases_payload, indent=2)}

Recommend a decision as the JSON object described in your instructions."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=RECOMMEND_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)
    parsed["source"] = "llm"
    return parsed


def recommend(new_case: dict, index: ReturnIndex, k: int = 5) -> tuple[dict, pd.DataFrame]:
    """Full pipeline: hard rules first, then retrieval + LLM. Returns (recommendation, similar_cases_df)."""
    similar_df = index.similar_cases(new_case, k=k)

    hard = apply_hard_rules(new_case)
    if hard is not None:
        return hard, similar_df

    rec = get_llm_recommendation(new_case, similar_df)
    return rec, similar_df


def log_feedback(new_case: dict, recommendation: dict, human_decision: str, path: str):
    """Append the final human-confirmed decision back into a feedback log for future retraining/analysis."""
    row = {
        **new_case,
        "ai_recommended_decision": recommendation.get("decision"),
        "ai_confidence": recommendation.get("confidence"),
        "ai_source": recommendation.get("source"),
        "human_decision": human_decision,
        "overridden": recommendation.get("decision") != human_decision,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)
