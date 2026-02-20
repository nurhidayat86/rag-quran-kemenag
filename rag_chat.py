"""
RAG chat over the Quran Chroma DB using LangChain, Gemini, and your local chroma_db.
Run: .\.conda\python.exe rag_chat.py
Then type questions in the terminal. Type 'quit' or 'exit' to stop.

Uses config.yaml for: gemini.api_key, chroma.persist_dir, chroma.collection_name,
and translation.model for the chat LLM. Query embeddings use chroma.embedding_model
(gemini-embedding-001) to match the existing Chroma collection.
"""
from pathlib import Path

import yaml

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. Copy config.example.yaml to config.yaml."
        )
    config = load_config(config_path) or {}
    gemini_cfg = config.get("gemini") or {}
    chroma_cfg = config.get("chroma") or {}
    translation_cfg = config.get("translation") or {}

    api_key = (gemini_cfg.get("api_key") or "").strip()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Set gemini.api_key in config.yaml. Get a key at https://aistudio.google.com/apikey"
        )

    persist_dir = chroma_cfg.get("persist_dir", "chroma_db")
    collection_name = chroma_cfg.get("collection_name", "quran")
    persist_path = (script_dir / persist_dir).resolve()
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at {persist_path}. Run build_chroma_db.py first."
        )

    chat_model_name = translation_cfg.get("model", "gemini-flash-latest")
    # Query embeddings must match the DB: use Gemini embedding (same as build_chroma_db when using Google)
    embed_model_for_queries = "gemini-embedding-001"

    # -------------------------------------------------------------------------
    # LangChain: embeddings, vector store, retriever, LLM, chain
    # -------------------------------------------------------------------------
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
    except ImportError as e:
        raise ImportError(
            "Install RAG dependencies: pip install langchain langchain-chroma langchain-google-genai"
        ) from e

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model_for_queries,
        google_api_key=api_key,
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatGoogleGenerativeAI(
        model=chat_model_name,
        google_api_key=api_key,
        temperature=0.2,
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about the Quran using only the provided context (translations, tafseer, or summaries). If the context does not contain enough information, say so. Answer in a clear, concise way. Do not make up verses or references."""),
        ("human", """Context from the Quran corpus:

{context}

Question: {question}"""),
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # -------------------------------------------------------------------------
    # Terminal chat loop
    # -------------------------------------------------------------------------
    print("RAG Chat (Quran). Using Chroma DB at:", persist_path)
    print("Model:", chat_model_name, "| Query embedding:", embed_model_for_queries)
    print('Type your question and press Enter. Type "quit" or "exit" to end.\n')

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        try:
            answer = rag_chain.invoke(question)
            print("Assistant:", answer, "\n")
        except Exception as e:
            print("Error:", e, "\n")


if __name__ == "__main__":
    main()
