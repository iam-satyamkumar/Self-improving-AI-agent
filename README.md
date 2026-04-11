# 🧠 Self-Improving Coding Agent (Phase 1)

This is NOT a chatbot.

This is an early-stage AI system that demonstrates:
- Decision-making (RAG vs Direct)
- Codebase understanding
- Retrieval transparency

---

## 🚀 What This System Does

Given a developer query, the system:

1. Decides whether to:
   - Use RAG (codebase search)
   - Answer directly

2. Retrieves relevant code chunks (if needed)

3. Generates an answer using an LLM

---

## 🧠 Architecture

User Query  
↓  
Router (LLM-based decision)  
↓  
[ RAG ] OR [ Direct ]  
↓  
LLM Response  

---

## 🔍 Key Features

### ✅ Decision Layer (Not Hardcoded)

The system uses an LLM to decide:

```json
{
  "decision": "RAG",
  "reason": "Query refers to project-specific code"
}
```

---

🧪 Setup

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create required folders
mkdir -p data logs sample_codebase

# 4. Add your codebase files inside:
# sample_codebase/

# 5. Build FAISS index
rm -rf data/*                
python -m rag.indexer

# 6. Start Ollama (separate terminal)
ollama serve

# 7. Pull lightweight model (first time only)
ollama run phi3

# 8. Start API server
uvicorn api.main:app --reload