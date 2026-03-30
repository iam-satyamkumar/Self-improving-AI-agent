import requests
import json
from datetime import datetime

LOG_FILE = "logs/decisions.jsonl"


def log_decision(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def call_local_llm(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": prompt, "stream": False},  # or "phi3"
            timeout=60,
        )
        print("DEBUG RESPONSE in router.py :", res.text)
        return res.json().get("response", "")
    except Exception as e:
        return f"ERROR: {str(e)}"


def safe_parse_json(text):
    """
    Extract JSON from messy LLM output
    """
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return None


def route(query):
    prompt = f"""
You are an intelligent AI system deciding how to answer a query.

Your task:
Decide whether the query needs access to a specific codebase (RAG) or not.

Choose:
- RAG → if the query refers to:
    * "this project", "this code", "this repo"
    * debugging, errors, implementation details
    * internal logic of a system
- DIRECT → if the query is general knowledge

STRICT RULE:
If query mentions ANY project-specific wording → ALWAYS choose RAG

Respond ONLY in JSON:
{{"decision": "RAG" or "DIRECT", "reason": "..."}}

Query: {query}
"""

    raw_output = call_local_llm(prompt)

    parsed = safe_parse_json(raw_output)

    # 🔥 fallback if parsing fails
    if not parsed or "decision" not in parsed:
        parsed = {"decision": "DIRECT", "reason": "fallback due to parsing error"}

    log_decision(
        {
            "query": query,
            "decision": parsed["decision"],
            "reason": parsed["reason"],
            "raw_output": raw_output,
            "time": str(datetime.now()),
        }
    )

    return parsed
