import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"

CODEBASE_PATH = "../A/AstroMagic-1"
FAISS_INDEX_PATH = "./data/index.faiss"
CHUNK_SIZE = 300
