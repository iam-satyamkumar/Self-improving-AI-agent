import requests
import json
from datetime import datetime
import re

LOG_FILE = "logs/decisions.jsonl"


def log_decision(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def call_local_llm(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": prompt, "stream": False},
            timeout=60,
        )
        print("DEBUG RESPONSE in router.py :", res.text)
        return res.json().get("response", "")
    except Exception as e:
        return f"ERROR: {str(e)}"


def safe_parse_json(text):
    """
    Extract JSON from messy LLM output (handles ```json blocks)
    """
    try:
        # remove markdown code blocks if present
        text = re.sub(r"```json|```", "", text)

        # extract JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    return None


def route(query):
    query_lower = query.lower()

    # 🔥 PRE-CHECK (fast + reliable)
    if any(x in query_lower for x in ["this project", "this code", "repo"]):
        result = {"decision": "RAG", "reason": "pre-check: project-specific query"}

        log_decision(
            {
                "query": query,
                "decision": result["decision"],
                "reason": result["reason"],
                "source": "heuristic",
                "time": str(datetime.now()),
            }
        )

        return result

    prompt = f"""
Classify the query.

RAG → if about code, project, debugging
DIRECT → if general knowledge

Respond ONLY as JSON:
{{"decision": "RAG" or "DIRECT", "reason": "..."}}

Query: {query}
"""

    raw_output = call_local_llm(prompt)

    parsed = safe_parse_json(raw_output)

    valid_decisions = ["RAG", "DIRECT"]

    # 🔥 fallback
    if (
        not parsed
        or "decision" not in parsed
        or parsed["decision"] not in valid_decisions
    ):
        parsed = {
            "decision": "DIRECT",
            "reason": "fallback due to parsing/invalid response",
        }
        source = "fallback"
    else:
        source = "llm"

    log_decision(
        {
            "query": query,
            "decision": parsed["decision"],
            "reason": parsed["reason"],
            "raw_output": raw_output,
            "source": source,
            "time": str(datetime.now()),
        }
    )

    return parsed
