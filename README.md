# rag-quran-kemenag
RAG chatbot for kemenag quran

## Running Python scripts
Use the Python from the `.conda` environment in this repo:

```powershell
.\.conda\python.exe clean_tafsir_structure.py
.\.conda\python.exe json_parser.py
```

## Translate to English (Gemini)
Translates `surah_cleaned/` (Indonesian) to English and writes to `surah_en/`:
- `translations.en.text` = English verse translations
- `tafsir.id.text` = English tafsir

1. Install deps: `pip install -r requirements.txt` (uses `google-genai` SDK; or use `.conda` env).
2. Copy `config.example.yaml` to `config.yaml` and set `gemini.api_key`. Get a key at: https://aistudio.google.com/apikey
3. Run:
   ```powershell
   .\.conda\python.exe translate_to_english.py
   ```

credit to: 
