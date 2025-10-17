Please give me code (python) example how to access zapier API to do Find top 3 tech news from today and add to my Google Sheet with title link and summary in a python based agentic ai system deployed on hugging face space using streamlit, agents.yaml. Please create a comprehensive instructions for beginner. Ending with follow up questions


Nice — I’ll give you a complete, copy-pasteable example you can deploy now: a Streamlit-based **agentic** Python app (for Hugging Face Spaces) that:

1. Fetches the **top 3 tech news items from today** (using NewsAPI as the news source — you can swap this for any news source),
2. Optionally summarizes each item using OpenAI (if you provide an `OPENAI_API_KEY`) or falls back to using the article snippet, and
3. Sends them to your **Google Sheet** through a Zapier **Webhook** (you create a Zap that receives a webhook and writes rows into Google Sheets).

I’ll provide:

* full `app.py` (Streamlit) code,
* `agents.yaml` (an example agent definition / config),
* `requirements.txt`,
* step-by-step beginner friendly instructions (including setting up the Zapier Zap and Google Sheet),
* notes on deployment to Hugging Face Spaces,
* security & debugging tips,
* short follow-up questions at the end.

---

# 1) Files to create

**Files** you should create in your repo for Hugging Face Space:

* `app.py` — the Streamlit app (main).
* `agents.yaml` — YAML config for the agent pipeline (example).
* `requirements.txt` — Python deps.
* `README.md` — (optional) instructions.

---

# 2) `requirements.txt`

Create a `requirements.txt` with:

```
streamlit>=1.20
requests>=2.28
python-dateutil>=2.8
openai>=1.0.0    # optional (used if you want AI summarization)
```

(If you prefer other summarizers or article scrapers, add them here.)

---

# 3) `agents.yaml` (example)

This is a simple agent config describing two agents (fetch + post). You can expand with more fields in your own system.

```yaml
# agents.yaml
agents:
  - id: news_fetcher
    name: "TechNewsFetcher"
    description: "Fetch top 3 tech news for today from NewsAPI and produce title, url, publishedAt, description."
    type: http
    params:
      source: "newsapi"
      api_key_env: "NEWSAPI_KEY"
      category: "technology"
      max_items: 3

  - id: zapier_poster
    name: "ZapierPoster"
    description: "Post rows to Zapier Webhook which forwards them to a Google Sheet."
    type: webhook
    params:
      webhook_url_env: "ZAPIER_WEBHOOK_URL"
      field_mapping:
        - title
        - url
        - summary
        - publishedAt
```

> This `agents.yaml` is a configuration for your agent pipeline. Your Streamlit app will read it to know what to run. You can make it more sophisticated (credentials, retry logic, scheduling, agent prompts, rate limiting, etc.) as needed.

---

# 4) `app.py` — Streamlit app (complete)

```python
# app.py
import os
import streamlit as st
import requests
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import json

# Optional OpenAI summarization
try:
    import openai
except Exception:
    openai = None

# --- UTILITIES ---------------------------------------------------------
def load_agents_config(path="agents.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()  # your app can parse it if needed; we show a simple display
    except FileNotFoundError:
        return None

def get_today_iso():
    # returns YYYY-MM-DD
    return datetime.utcnow().date().isoformat()

def fetch_top_tech_news_from_newsapi(api_key, max_items=3):
    """Uses NewsAPI.org top-headlines with category=technology and today's date"""
    # NewsAPI top-headlines doesn't accept a from param for top-headlines, so we'll use everything endpoint with sortBy=publishedAt
    today = datetime.utcnow().date()
    from_date = today.isoformat()
    url = "https://newsapi.org/v2/everything"
    q = "technology OR tech OR startup OR 'Silicon Valley' OR 'AI' OR 'software'"

    params = {
        "q": q,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_items,
        "apiKey": api_key,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])[:max_items]
    # normalize
    results = []
    for a in articles:
        results.append({
            "title": a.get("title"),
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "description": a.get("description") or "",
            "content": a.get("content") or "",
            "source": a.get("source", {}).get("name"),
        })
    return results

def summarize_text_openai(text, openai_api_key, max_tokens=120):
    """Simple summarization using OpenAI (chat/completions). Optional - uses gpt-3.5/4 style call."""
    if not openai:
        raise RuntimeError("openai package not installed")
    openai.api_key = openai_api_key
    # Use ChatCompletion if available; fallback to Completion
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list() else "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": f"Summarize this in one short paragraph (max 50 words):\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        # The structure varies, but typically:
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # fallback: short slice
        return text.strip()[:200]

def summarize_fallback(text):
    # fallback simple heuristic summary: first 2 sentences or first 200 chars
    if not text:
        return ""
    # naive sentence split
    sentences = text.split(". ")
    if len(sentences) >= 2:
        return (sentences[0] + (". " + sentences[1] if len(sentences) > 1 else "")).strip()
    return text[:200].strip()

def post_to_zapier(webhook_url, items):
    """POST items to Zapier webhook. Zap should be created with 'Catch Hook' trigger and Google Sheets action."""
    payload = {"items": items}
    resp = requests.post(webhook_url, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json() if resp.content else {"status": "ok"}

# --- STREAMLIT UI ------------------------------------------------------
st.set_page_config(page_title="TechNews → Google Sheet via Zapier", layout="wide")

st.title("Top 3 Tech News → Google Sheet (via Zapier Webhook)")

st.markdown(
    """
    This app fetches the top 3 tech news from **today** and can send them to your Google Sheet via a Zapier webhook.
    You must configure:
    - `NEWSAPI_KEY` (NewsAPI.org) OR replace fetch function,
    - `ZAPIER_WEBHOOK_URL` (from a Zap with "Catch Hook" -> Google Sheets).
    Optional:
    - `OPENAI_API_KEY` to produce concise summaries.
    """
)

# load agents.yaml text (display only)
agents_yaml_text = load_agents_config("agents.yaml")
if agents_yaml_text:
    with st.expander("agents.yaml (loaded)"):
        st.code(agents_yaml_text, language="yaml")

# Input fields (secrets from environment recommended)
st.sidebar.header("Configuration (use Space secrets or env vars)")

newsapi_key = os.environ.get("NEWSAPI_KEY") or st.sidebar.text_input("NEWSAPI_KEY", value="", type="password")
zapier_webhook = os.environ.get("ZAPIER_WEBHOOK_URL") or st.sidebar.text_input("ZAPIER_WEBHOOK_URL", value="", type="password")
openai_key = os.environ.get("OPENAI_API_KEY") or st.sidebar.text_input("OPENAI_API_KEY (optional for summaries)", value="", type="password")

max_items = st.sidebar.number_input("Max items (top N)", min_value=1, max_value=10, value=3, step=1)

col1, col2 = st.columns([2,1])

with col1:
    st.header("Fetch and Preview")
    if st.button("Fetch top tech news (today)"):
        if not newsapi_key:
            st.error("Please provide NEWSAPI_KEY (or modify fetch function to use another source).")
        else:
            try:
                articles = fetch_top_tech_news_from_newsapi(newsapi_key, max_items=max_items)
                if not articles:
                    st.warning("No articles returned for today.")
                else:
                    # Summarize
                    summarized = []
                    for a in articles:
                        summary = ""
                        # prefer OpenAI if key provided and package available
                        text_for_summary = (a.get("description") or "") + "\n\n" + (a.get("content") or "")
                        if openai_key and openai:
                            try:
                                summary = summarize_text_openai(text_for_summary, openai_key)
                            except Exception as e:
                                summary = summarize_fallback(text_for_summary)
                        else:
                            summary = summarize_fallback(text_for_summary)
                        summarized.append({
                            "title": a["title"],
                            "url": a["url"],
                            "publishedAt": a["publishedAt"],
                            "summary": summary,
                            "source": a.get("source"),
                        })
                    st.session_state["latest_fetch"] = summarized
                    st.success(f"Fetched {len(summarized)} articles.")
                    for idx, art in enumerate(summarized, start=1):
                        st.subheader(f"{idx}. {art['title']}")
                        st.write(f"Source: {art.get('source')}, Published: {art.get('publishedAt')}")
                        st.write("URL:", art["url"])
                        st.write("Summary:", art["summary"])

with col2:
    st.header("Send to Google Sheet (via Zapier)")
    st.write("Zapier webhook URL should be the 'Catch Hook' endpoint created in your Zap.")
    if st.button("Send latest fetch to Google Sheet via Zapier"):
        items = st.session_state.get("latest_fetch")
        if not items:
            st.warning("No fetched items in session. First click 'Fetch top tech news (today)'.")
        elif not zapier_webhook:
            st.error("Please provide ZAPIER_WEBHOOK_URL (or set it as env var).")
        else:
            try:
                # Prepare rows payload for Zapier -> Zap should map each item to columns
                rows = []
                for it in items:
                    rows.append({
                        "title": it.get("title"),
                        "url": it.get("url"),
                        "summary": it.get("summary"),
                        "publishedAt": it.get("publishedAt"),
                        "source": it.get("source"),
                    })
                resp = post_to_zapier(zapier_webhook, rows)
                st.success("Sent to Zapier webhook successfully.")
                st.code(json.dumps(resp, indent=2))
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.markdown("### Notes and logs")
with st.expander("Session state (debug)"):
    st.write(st.session_state.get("latest_fetch", "no data"))
```

**Explanation (brief)**:

* `fetch_top_tech_news_from_newsapi` calls NewsAPI's `everything` endpoint filtered to today's date, returning top N articles.
* Summarization: optional OpenAI-based summarization if you provide `OPENAI_API_KEY` and install `openai`. Otherwise fallback to a naive first-two-sentences summary.
* `post_to_zapier` performs a JSON POST to the Zapier webhook URL. You'll create a Zap whose trigger is **"Catch Hook"** and action is **Google Sheets - Create Spreadsheet Row(s)**.
* The app stores fetched items in `st.session_state` so you can preview then send.

---

# 5) How to create the Zapier Zap (Beginner steps)

1. Log into Zapier.
2. Click **Create Zap**.
3. For the **Trigger**:

   * Choose **Webhooks by Zapier**.
   * Select **Catch Hook**.
   * Zapier will give you a **Custom Webhook URL** that looks like `https://hooks.zapier.com/hooks/catch/123456/abcdef/`. Copy this — it is your `ZAPIER_WEBHOOK_URL`.
4. Optionally test the trigger:

   * Click **Test trigger** while your Streamlit app sends a sample payload (you can test by doing a POST to the URL from `curl` or from the Streamlit app).
5. For the **Action**:

   * Choose **Google Sheets**.
   * Select **Create Spreadsheet Row(s)** (or **Create Spreadsheet Row** depending on your Zapier account).
   * Select your Google account and choose the Spreadsheet and Worksheet.
   * Map incoming webhook fields to columns:

     * e.g., `{{items__0__title}}` → Title column
     * `{{items__0__url}}` → URL column
     * `{{items__0__summary}}` → Summary column
     * `{{items__0__publishedAt}}` → Date column
   * If your webhook contains a list `items`, Zapier can iterate or create multiple rows depending on how you configure it; simplest: have Zapier create one row per item by using a Zapier action that can handle arrays or use **Zapier Looping** (Advanced). Another simple approach is to create a new Zap that receives a single item payload — your Streamlit app can POST each item separately in a loop. (I’ll give both code patterns below.)
6. Turn the Zap on.

**Two sending patterns with Zapier**:

* **Bulk POST (single request with `items` array):** Use Zapier's *Catch Hook* to receive `items` array. Use **Zapier Looping** or a Zapier action that can expand arrays into multiple rows. (Looping by Zapier is premium.)
* **One POST per article (recommended for free Zapier plans):** In `app.py`, instead of posting array under `"items"`, loop and send each article as its own POST (Zap will create one row per request). This is simpler and works with the basic "Catch Hook" -> "Create Spreadsheet Row" action.

**If you prefer one-request-per-row**, replace `post_to_zapier` in `app.py` with:

```python
def post_single_to_zapier(webhook_url, item):
    resp = requests.post(webhook_url, json=item, timeout=15)
    resp.raise_for_status()
    return resp.json() if resp.content else {"status":"ok"}
```

and send each item in a loop.

---

# 6) Setting secrets in Hugging Face Spaces

* Create a new Space (choose **Streamlit**).
* In the Space's **Settings → Secrets**, add these names and values:

  * `NEWSAPI_KEY` = *your NewsAPI key*
  * `ZAPIER_WEBHOOK_URL` = *your Zapier webhook URL*
  * `OPENAI_API_KEY` = *optional, for summaries*
* Push the repo files (`app.py`, `requirements.txt`, `agents.yaml`) to the Space repo.
* The Space will install packages and run `streamlit run app.py`.

---

# 7) Getting a NewsAPI key (quick)

* Go to [https://newsapi.org](https://newsapi.org) and sign up for an API key.
* Note: the free plan has rate limits and may restrict some endpoints — if you use other news providers (Bing News Search, GDELT, etc.), adapt the fetch function accordingly.

---

# 8) Optional: Use OpenAI for better summarization

* Install `openai` in `requirements.txt`.
* Set `OPENAI_API_KEY` in Environment/Secrets.
* The example will call OpenAI's Chat API to produce concise summaries. You can tune `max_tokens` and `temperature` in `summarize_text_openai`.

> If you prefer to avoid using OpenAI, the fallback summarizer will still produce a short snippet based on description/content.

---

# 9) Example: One-POST-per-article code snippet (send each item individually)

Replace the Zapier send block in `app.py` with this pattern:

```python
for it in items:
    single_payload = {
        "title": it["title"],
        "url": it["url"],
        "summary": it["summary"],
        "publishedAt": it["publishedAt"],
    }
    post_single_to_zapier(zapier_webhook, single_payload)
```

This produces one Zap run per article (simple mapping to sheet row).

---

# 10) Deployment checklist (quick)

1. Confirm `requirements.txt` includes all used libraries.
2. Add `app.py`, `agents.yaml`, `requirements.txt` to your Git repo.
3. Create a new **Space** on Hugging Face with **Streamlit** runtime.
4. Add secrets in the Space settings (`NEWSAPI_KEY`, `ZAPIER_WEBHOOK_URL`, optionally `OPENAI_API_KEY`).
5. Push repo to Space. The Space will build and present the app UI.
6. Test: click **Fetch** then **Send**. Check Zap runs and Google Sheet rows.

---

# 11) Debugging tips

* If Zapier reports no data: check that your webhook URL is the **Catch Hook** URL and that your POST payload is JSON. You can test manually:

  ```bash
  curl -X POST https://hooks.zapier.com/hooks/catch/XXX/YYY/ -H "Content-Type: application/json" -d '{"title":"Test","url":"https://example.com","summary":"hello"}'
  ```
* If NewsAPI returns 401: verify `NEWSAPI_KEY`.
* If no articles: try removing the `from` date or expanding the `q` parameter (free NewsAPI limits may affect results).
* For local development, set env vars or input keys in the sidebar.

---

# 12) Security & costs

* Never commit secrets (API keys, webhook URLs) to a public repo. Use Hugging Face **Secrets** or environment variables.
* Zapier runs can consume tasks — check your Zapier plan for quota.
* OpenAI summarization consumes tokens — watch your usage.

---

# 13) Summary: what this system does

* `news_fetcher` agent (app) queries NewsAPI for today's tech stories.
* Summaries are generated via OpenAI if available (or fallback).
* `zapier_poster` agent (app) posts each story (title + url + summary + publishedAt) to a Zapier webhook that writes rows into your Google Sheet.

---

# 14) Short follow-up questions (as requested)

1. Do you want the app to **automatically run daily** (e.g., scheduled Space or external scheduler) or will you run it manually from the UI?
2. Do you prefer **one Zap per article** (simpler) or a **single bulk POST for multiple rows** (requires Zapier Looping/premium or array handling)?
3. Do you want me to adapt the code to use a different news source (e.g., Bing News Search, RSS feeds, or your own custom sources)?

---

If you want, I can now:

* provide the **one-POST-per-row** version of `app.py` (so it maps directly to Google Sheet rows via basic Zapier action), or
* produce a version that uses **Bing News Search** instead of NewsAPI, or
* create a more advanced `agents.yaml` with retry logic, rate-limiting, and agent prompts for each step.

Which of those would you like next?
