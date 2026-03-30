from fastapi import FastAPI
from pydantic import BaseModel
from agent.router import route
from rag.retriever import retrieve
from openai import OpenAI
from config import *
import requests


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

    if decision["decision"] == "RAG":
        context = retrieve(req.query)

        context_text = "\n\n".join([c["text"] for c in context])

        prompt = f"""
You are a senior backend engineer.

Answer the question ONLY using the provided code context.

STRICT RULES:
- Do NOT add generic explanations
- Do NOT assume anything not in code
- If something is unclear → say "Not found in code"
- Keep answer concise and structured
- Reference actual logic from code

CODE CONTEXT:
{context_text}

QUESTION:
{req.query}
"""

    else:
        prompt = req.query

    response = local_llm(prompt)

    return {
        "decision": decision,
        "context": context if decision["decision"] == "RAG" else None,
        "answer": response,
    }
