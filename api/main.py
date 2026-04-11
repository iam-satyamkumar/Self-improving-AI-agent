from fastapi import FastAPI
from pydantic import BaseModel
from agent.router import route
from rag.retriever import retrieve
from openai import OpenAI
from config import *
import requests
from memory.retriever import retrieve_memory
from memory.store import add_memory


def local_llm(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "phi3", "prompt": prompt, "stream": False},  # or phi3
    )
    print("DEBUG RESPONSE in main.py :", res.text)
    return res.json()["response"]


app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def handle_query(req: QueryRequest):
    decision = route(req.query)

    # ✅ ALWAYS retrieve memory
    memory_context = retrieve_memory(req.query)

    memory_text = "\n\n".join(
        [
            f"Past mistake: {m['bad_answer']}\nFix: {m['correct_answer']}"
            for m in memory_context
        ]
    )

    if decision["decision"] == "RAG":
        context = retrieve(req.query)

        context_text = "\n\n".join(
            [f"FILE: {c['file']}\n{c['text']}" for c in context[:3]]
        )

        prompt = f"""
You are a senior backend engineer.

Answer the question ONLY using the following-
1. Code context
2. Past learnings

STRICT RULES:
- Do NOT add generic explanations
- Do NOT assume anything not in code
- If something is unclear → say "Not found in code"
- Keep answer concise and structured
- Reference actual logic from code
- Only use given code
- No generic explanations
- Be concise

PAST LEARNINGS:
{memory_text}

CODE CONTEXT:
{context_text}

QUESTION:
{req.query}
"""
    else:
        context = None

        prompt = f"""
You are an expert assistant.

Use past learnings if relevant.

PAST LEARNINGS:
{memory_text}

QUESTION:
{req.query}
"""

    response = local_llm(prompt)

    return {
        "decision": decision,
        "context": context,
        "answer": response,
    }
