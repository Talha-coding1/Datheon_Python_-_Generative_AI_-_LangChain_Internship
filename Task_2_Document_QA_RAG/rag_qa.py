"""
Document Q&A Assistant with RAG & Vector Embeddings
----------------------------------------------------
Pipeline:
  1. Load PDF / TXT documents          (LangChain document loaders)
  2. Chunk text recursively            (RecursiveCharacterTextSplitter)
  3. Generate vector embeddings        (HuggingFace all-MiniLM-L6-v2, runs locally, free)
  4. Store embeddings in ChromaDB      (persistent on disk)
  5. Retrieve relevant chunks          (similarity search, top-k)
  6. Answer strictly from context      (RetrievalQA chain) + source page citations

LLM backends (auto-selected, in this order):
  - OpenAI  : if OPENAI_API_KEY is set
  - Gemini  : if GOOGLE_API_KEY is set (free key from aistudio.google.com)
  - Ollama  : if USE_OLLAMA=1 (e.g. llama3.2 running locally, free)
  - None    : extractive fallback -> prints the best matching passages with citations

Usage:
  python rag_qa.py ingest                 # build the vector store from ./documents
  python rag_qa.py ask "your question"    # single question
  python rag_qa.py chat                   # interactive Q&A loop
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path(os.getenv("DOCS_DIR", "documents"))
DB_DIR = os.getenv("DB_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 4))

NOT_FOUND = "I could not find the answer in the provided documents."

PROMPT_TEMPLATE = """You are a document question-answering assistant.
Answer the question using ONLY the context below. Do not use outside knowledge.
If the answer is not contained in the context, reply exactly: "{not_found}"

Context:
{{context}}

Question: {{question}}

Answer:"""


# ---------- 1. Load ----------
def load_documents(docs_dir: Path):
    docs = []
    files = sorted(list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt")))
    if not files:
        sys.exit(f"No PDF/TXT files found in '{docs_dir}'. Add some and retry.")
    for f in files:
        loader = PyPDFLoader(str(f)) if f.suffix.lower() == ".pdf" else TextLoader(str(f), encoding="utf-8")
        loaded = loader.load()  # one Document per PDF page, metadata has 'source' and 'page'
        print(f"  loaded {f.name}: {len(loaded)} page(s)")
        docs.extend(loaded)
    return docs


# ---------- 2. Chunk ----------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)  # page metadata is preserved on every chunk
    print(f"  split into {len(chunks)} chunks")
    return chunks


# ---------- 3 & 4. Embed + store ----------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore():
    print("Loading documents...")
    docs = load_documents(DOCS_DIR)
    chunks = split_documents(docs)
    print("Embedding and storing in ChromaDB (first run downloads the model)...")
    # Rebuild cleanly so re-ingesting doesn't create duplicates
    try:
        Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings()).delete_collection()
    except Exception:
        pass
    db = Chroma.from_documents(chunks, get_embeddings(), persist_directory=DB_DIR)
    print(f"Done. Vector store saved to '{DB_DIR}/'.")
    return db


def load_vectorstore():
    if not Path(DB_DIR).exists():
        sys.exit("Vector store not found. Run:  python rag_qa.py ingest")
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


# ---------- 5 & 6. Retrieve + answer ----------
def get_llm():
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"), temperature=0)
    if os.getenv("USE_OLLAMA") == "1":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
    return None


def format_sources(docs):
    seen, out = set(), []
    for d in docs:
        name = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page")
        label = f"{name}, page {page + 1}" if isinstance(page, int) else name  # PyPDF pages are 0-indexed
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def answer(db, question: str, llm):
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    if llm is None:  # extractive fallback (no LLM needed)
        docs = retriever.invoke(question)
        text = "\n\n".join(f"[{i+1}] {d.page_content.strip()}" for i, d in enumerate(docs))
        return f"(No LLM configured - showing most relevant passages)\n\n{text}", docs

    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE.format(not_found=NOT_FOUND),
        input_variables=["context", "question"],
    )
    try:
        from langchain.chains import RetrievalQA  # classic chain (langchain 0.3.x)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        res = chain.invoke({"query": question})
        return res["result"].strip(), res["source_documents"]
    except ImportError:
        # Same "stuff" RAG logic without the legacy chain (works with newer langchain-core)
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        msg = (prompt | llm).invoke({"context": context, "question": question})
        text = msg.content if isinstance(msg.content, str) else "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in msg.content
        )
        return text.strip(), docs


def print_answer(db, question, llm):
    ans, docs = answer(db, question, llm)
    print(f"\nAnswer:\n{ans}\n")
    if NOT_FOUND not in ans:
        print("Sources:")
        for s in format_sources(docs):
            print(f"  - {s}")
    print()


def main():
    p = argparse.ArgumentParser(description="Document Q&A with RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Build vector store from ./documents")
    a = sub.add_parser("ask", help="Ask a single question")
    a.add_argument("question")
    sub.add_parser("chat", help="Interactive Q&A")
    args = p.parse_args()

    if args.cmd == "ingest":
        build_vectorstore()
        return

    db, llm = load_vectorstore(), get_llm()
    if args.cmd == "ask":
        print_answer(db, args.question, llm)
    else:
        print("Ask questions about your documents (type 'exit' to quit).")
        while True:
            q = input("You: ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            print_answer(db, q, llm)


if __name__ == "__main__":
    main()
