# rag-quran-kemenag
RAG chatbot for kemenag quran

## Running Python scripts
Use the Python from the `.conda` environment in this repo:

```powershell
.\.conda\python.exe <script_name>.py
```

## Prerequisites
- **config.yaml** — Copy `config.example.yaml` to `config.yaml` and set `gemini.api_key`. Get a key at: https://aistudio.google.com/apikey
- **Dependencies** — `pip install -r requirements.txt` (or use the `.conda` env). Uses `google-genai`, `chromadb`, `PyYAML`; for local embeddings also `sentence-transformers`.

---

## Translate to English (Gemini)
Translates `surah_cleaned/` (Indonesian) to English and writes to `surah_en/`:
- `translations.en.text` = English verse translations  
- `tafsir.en.text` = English tafseer  

```powershell
.\.conda\python.exe translate_to_english.py
```

---

## Building the Chroma DB (step-by-step)

The Chroma DB is built from vector JSON files. Follow these steps in order.

### 1. Prepare cleaned English Quran (if you have `surah_en/`)
If you already have `surah_en/` (e.g. from `translate_to_english.py`), build the single cleaned JSON:

| Script | Input | Output |
|--------|--------|--------|
| `build_quran_english_cleaned.py` | `surah_en/*.json` (1.json … 114.json) | `surah_en_cleaned/quran_english_cleaned.json` |

```powershell
.\.conda\python.exe build_quran_english_cleaned.py
```

### 2. Build vector JSONs for translation, tafseer, and summary
All three scripts read from `surah_en_cleaned/quran_english_cleaned.json` and write into `pre_vector_db/`.

| Script | Output | Description |
|--------|--------|-------------|
| `build_translation_vector_json.py` | `pre_vector_db/translation_vector.json` | 3-ayah sliding-window chunks of verse translation |
| `build_tafseer_vector_json.py` | `pre_vector_db/tafseer_vector.json` | Tafseer chunks (~1800 chars, ~200 overlap, split on paragraphs/sentences) |
| `build_summary_vector_json.py` | `pre_vector_db/summary_vector.json` | One summary per surah via Gemini (uses `config.yaml`: `gemini.api_key`, `translation.model`) |

```powershell
.\.conda\python.exe build_translation_vector_json.py
.\.conda\python.exe build_tafseer_vector_json.py
.\.conda\python.exe build_summary_vector_json.py
```

### 3. (Optional) Convert for Chroma compatibility
Checks `pre_vector_db/*.json` and, if any have metadata types Chroma doesn’t support, writes converted files to `pre_chroma_db/`. If you skip this, `build_chroma_db.py` will still load from `pre_vector_db/` when `pre_chroma_db/` is missing or empty.

| Script | Input | Output |
|--------|--------|--------|
| `convert_pre_vector_to_chroma.py` | `pre_vector_db/*.json` | `pre_chroma_db/*.json` (only files that needed conversion, or all with `--force`) |

```powershell
.\.conda\python.exe convert_pre_vector_to_chroma.py
```

### 4. Build the Chroma DB
Loads all vector JSONs from `pre_chroma_db/` if present, otherwise from `pre_vector_db/`, embeds texts, and persists a single collection. Settings (collection name, paths, embedding model, batch size) come from `config.yaml` under `chroma`.

| Script | Input | Output |
|--------|--------|--------|
| `build_chroma_db.py` | `pre_chroma_db/*.json` or `pre_vector_db/*.json` | `chroma_db/` (path from `chroma.persist_dir`, default `chroma_db`) |

```powershell
.\.conda\python.exe build_chroma_db.py
```

**Quick recap (scripts in order):**
1. `build_quran_english_cleaned.py`  
2. `build_translation_vector_json.py`  
3. `build_tafseer_vector_json.py`  
4. `build_summary_vector_json.py`  
5. `convert_pre_vector_to_chroma.py` (optional)  
6. `build_chroma_db.py`  

---

## Chat with the Quran (RAG)

After the Chroma DB is built, run the RAG chat script to ask questions in the terminal. It uses LangChain, Gemini (from `config.yaml`), and your local `chroma_db`.

```powershell
.\.conda\python.exe rag_chat.py
```

Type your question and press Enter. Type `quit` or `exit` to end. Requires `langchain`, `langchain-chroma`, and `langchain-google-genai` (see `requirements.txt`).

---

## ChromaDB and embeddings

Embedding behavior is controlled by `config.yaml` under `chroma`:

- **Google Gen AI (`gemini-embedding-001`)** — Set `chroma.embedding_model: "gemini-embedding-001"`. Uses `gemini.api_key`; no local model. Optional `chroma.embed_batch_size` (default 100).
- **Sentence-transformers (e.g. BGE)** — Set `chroma.embedding_model` to a Hugging Face model (e.g. `BAAI/bge-large-en-v1.5`). Then optionally:
  - **NVIDIA GPU:** `chroma.embedding_device: "cuda"`.
  - **Intel Arc (XPU):** `chroma.embedding_device: "xpu"` and install Intel Extension for PyTorch. On Windows, `pip install intel-extension-for-pytorch` often fails; use `chroma.embedding_device: "cpu"` or follow [Intel’s installation guide](https://intel.github.io/intel-extension-for-pytorch/).

---

credit to:
