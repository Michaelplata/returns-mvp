# Returns Decision Assistant — MVP

A human-in-the-loop assistant for return decisions. Given historical return
cases and their outcomes, it retrieves similar past cases for a new return
and recommends a decision (refund / exchange / store credit / repair / deny)
for a human agent to confirm or override.

## How it works

1. **Hard policy rules** run first and deterministically (e.g. return window
   expired) — no LLM involved, so these are fast, free, and 100% predictable.
2. If no hard rule fires, the new case is compared against historical cases
   using TF-IDF similarity (`recommender.py`) to find the most relevant
   precedents.
3. Claude is given the new case + the similar precedents and asked to
   recommend a decision with a rationale that cites which past cases it used.
4. A human agent reviews the recommendation and similar cases side by side,
   then confirms or overrides it in the UI.
5. The final human decision is logged to `data/feedback_log.csv`, along with
   whether it matched the AI's recommendation — this is your quality metric
   and your growing training set.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Using your own data

Replace `data/sample_returns.csv`, or upload your own CSV from the sidebar
in the running app. Expected columns:

| Column | Example | Notes |
|---|---|---|
| `sku` | SKU-SHOE-01 | |
| `category` | Footwear | |
| `reason_code` | wrong_size | Normalize these — free text hurts retrieval |
| `days_since_purchase` | 12 | integer |
| `item_condition` | unworn_tags_attached | |
| `order_value` | 89.99 | |
| `customer_return_count` | 0 | prior returns by this customer |
| `proof_of_purchase` | yes / no | |
| `decision` | approve_refund | the historical outcome — required for training cases |

## Customizing policy rules

Edit `RETURN_WINDOW_DAYS` and `CATEGORY_WINDOWS` at the top of
`recommender.py`, and add more conditions to `apply_hard_rules()` as needed
(e.g. blocklisted SKUs, fraud flags, VIP customer overrides).

## What's intentionally simple (and what to upgrade later)

- **TF-IDF similarity, not embeddings.** Works fine at MVP scale (thousands
  of rows) with zero external dependency. Swap in a real embedding model +
  vector DB (e.g. pgvector, Chroma) once your dataset grows or you need
  semantic matching beyond keyword overlap.
- **CSV storage.** Move to Postgres once multiple people need concurrent
  access.
- **No auth / multi-tenant support.** Fine for an internal pilot with one
  team; needed before wider rollout.
- **No retraining loop yet.** The feedback log is being collected from day
  one so you can later analyze disagreement patterns or fine-tune, but the
  MVP doesn't do anything automatic with it yet — that's a deliberate
  scope cut.
