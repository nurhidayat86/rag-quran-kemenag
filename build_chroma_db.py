"""
Load all vector JSON files into a ChromaDB collection.
Reads from pre_chroma_db/ if present (converted format), otherwise from pre_vector_db/.
Uses a single collection with metadata.text_type to distinguish translation, tafseer, summary.
Settings (collection_name, persist_dir, batch_size, dirs, embedding, embedding_device,
huggingface_access_token) are read from config.yaml [chroma].
Embedding: use_external_embedding=false → Chroma default; true with embedding_model=BAAI/... → sentence-transformers;
embedding_model=gemini-embedding-001 (or text-embedding-004 in config) → Google Gen AI (uses gemini.api_key).
Gemini API only supports gemini-embedding-001; text-embedding-004 is mapped to it.
GPU: set embedding_device to "cuda" (NVIDIA) or "xpu" (Intel Arc; requires intel-extension-for-pytorch).
Requires: chromadb, PyYAML; for sentence-transformers: sentence-transformers; for Google: google-genai.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import yaml

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:
    raise ImportError("Install chromadb: pip install chromadb")

# Google Gen AI embedding (optional)
def _google_embed_batch(client, model: str, texts: List[str], embed_batch_size: int = 100) -> List[List[float]]:
    """Call Google embed_content in batches; returns list of embedding vectors (one per text)."""
    all_embeddings: List[List[float]] = []
    for start in range(0, len(texts), embed_batch_size):
        chunk = texts[start : start + embed_batch_size]
        result = client.models.embed_content(model=model, contents=chunk)
        # Response: result.embeddings is list of vectors (one per content), or single vector for single content
        def to_vec(emb):
            if hasattr(emb, "values"):
                return list(emb.values)
            return list(emb)
        if hasattr(result, "embeddings") and result.embeddings:
            embs = result.embeddings
            # Single vector (list of floats) vs list of vectors
            if embs and isinstance(embs[0], (int, float)):
                all_embeddings.append(to_vec(embs))
            else:
                for emb in embs:
                    all_embeddings.append(to_vec(emb))
        else:
            vec = getattr(result, "embedding", result)
            all_embeddings.append(to_vec(vec))
        if start + embed_batch_size < len(texts):
            time.sleep(0.2)
    return all_embeddings


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_json_paths(
    script_dir: Path, pre_chroma_dir: str, pre_vector_dir: str
) -> List[Tuple[Path, Path]]:
    """Return list of (source_path, display_name) for each JSON to load. Prefer pre_chroma when file exists there."""
    pre_chroma = script_dir / pre_chroma_dir
    pre_vector = script_dir / pre_vector_dir
    if not pre_vector.exists():
        return []
    result = []
    for path in sorted(pre_vector.glob("*.json")):
        alt = pre_chroma / path.name
        source = alt if alt.exists() else path
        result.append((source, path.name))
    return result


def load_json_records(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def main():
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. Copy config.example.yaml to config.yaml."
        )
    config = load_config(config_path) or {}
    chroma_cfg = config.get("chroma") or {}
    collection_name = chroma_cfg.get("collection_name", "quran")
    persist_dir = chroma_cfg.get("persist_dir", "chroma_db")
    batch_size = int(chroma_cfg.get("batch_size", 5000))
    pre_chroma_dir = chroma_cfg.get("pre_chroma_dir", "pre_chroma_db")
    pre_vector_dir = chroma_cfg.get("pre_vector_dir", "pre_vector_db")
    use_external_embedding = bool(chroma_cfg.get("use_external_embedding", False))
    embedding_model = chroma_cfg.get("embedding_model", "BAAI/bge-large-en-v1.5")
    embedding_device = (chroma_cfg.get("embedding_device") or "cpu").strip().lower()
    hf_token = (chroma_cfg.get("huggingface_access_token") or "").strip()

    json_paths = get_json_paths(script_dir, pre_chroma_dir, pre_vector_dir)
    if not json_paths:
        raise FileNotFoundError(
            f"{pre_vector_dir} not found or empty under {script_dir}. "
            "Run convert_pre_vector_to_chroma.py first if needed."
        )

    print(f"Loading vector JSONs (from {pre_chroma_dir}/ when present, else {pre_vector_dir}/) ...", flush=True)
    all_ids = []
    all_documents = []
    all_metadatas = []

    for path, name in json_paths:
        print(f"  Loading {name} ...", end=" ", flush=True)
        records = load_json_records(path)
        n_added = 0
        for rec in records:
            if not isinstance(rec, dict) or "id" not in rec or "text" not in rec:
                continue
            all_ids.append(rec["id"])
            all_documents.append(rec["text"])
            meta = rec.get("metadata") or {}
            all_metadatas.append(meta)
            n_added += 1
        print(f"{n_added} records (source: {path.parent.name}/)", flush=True)

    print(f"Total: {len(all_ids)} documents to add to ChromaDB.", flush=True)

    if not all_ids:
        raise ValueError("No valid records found in JSON files.")

    # Store ChromaDB data inside the chroma_db folder (under project root)
    persist_path = (script_dir / persist_dir).resolve()
    persist_path.mkdir(parents=True, exist_ok=True)
    print(f"ChromaDB will be written to: {persist_path}", flush=True)

    _emb = (embedding_model or "").strip()
    use_google_embedding = _emb.lower() in (
        "text-embedding-004",
        "text-embedding-005",
        "gemini-embedding-001",
    )
    # Gemini API (Google AI Studio) only supports gemini-embedding-001; text-embedding-004 is not available
    google_embed_model = "gemini-embedding-001" if use_google_embedding else None
    embedding_fn = None
    genai_client = None
    embed_batch_size = int(chroma_cfg.get("embed_batch_size", 100))

    if use_google_embedding:
        api_key = (config.get("gemini") or {}).get("api_key") or ""
        api_key = (api_key.strip() if isinstance(api_key, str) else "") or ""
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError(
                "Google embedding (gemini-embedding-001) requires gemini.api_key in config.yaml."
            )
        from google import genai
        genai_client = genai.Client(api_key=api_key)
        print(f"Using Google Gen AI embedding model: {google_embed_model} (config: {embedding_model}; API batch size: {embed_batch_size}).", flush=True)
    elif use_external_embedding:
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            print("Using Hugging Face access token from config for model download.", flush=True)
        print(f"Loading external embedding model: {embedding_model} (device: {embedding_device}) ...", flush=True)
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device=embedding_device,
        )
        print("  Model loaded.", flush=True)
    else:
        print("Using ChromaDB default (internal) embedding model.", flush=True)

    print(f"Creating ChromaDB collection '{collection_name}' ...", flush=True)
    client = chromadb.PersistentClient(path=str(persist_path), settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    create_kw: dict = {"name": collection_name, "metadata": {"description": "Quran translation, tafseer, and summary chunks"}}
    if embedding_fn is not None:
        create_kw["embedding_function"] = embedding_fn
    collection = client.create_collection(**create_kw)
    print("  Collection created.", flush=True)

    num_batches = (len(all_ids) + batch_size - 1) // batch_size
    print(f"Writing documents (embedding + persist) in {num_batches} batch(es) of up to {batch_size} ...", flush=True)
    total_written = 0
    for i in range(0, len(all_ids), batch_size):
        batch_num = i // batch_size + 1
        ids_batch = all_ids[i : i + batch_size]
        docs_batch = all_documents[i : i + batch_size]
        metas_batch = all_metadatas[i : i + batch_size]
        print(f"  Batch {batch_num}/{num_batches}: embedding and storing {len(ids_batch)} documents ...", end=" ", flush=True)
        if genai_client is not None:
            embeddings_batch = _google_embed_batch(
                genai_client, google_embed_model, docs_batch, embed_batch_size=embed_batch_size
            )
            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                documents=docs_batch,
                metadatas=metas_batch,
            )
        else:
            collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
        total_written += len(ids_batch)
        print(f"done (total: {total_written})", flush=True)

    n = collection.count()
    print(f"Done. Collection '{collection_name}' has {n} documents. Persisted to {persist_path}", flush=True)


if __name__ == "__main__":
    main()
