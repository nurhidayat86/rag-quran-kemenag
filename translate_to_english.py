"""
Translate translations.id.text and tafsir.id.text from Indonesian to English using Gemini (fast).
Reads from surah_cleaned/, writes to surah_en/ with:
  - translations.en.name = English translation of translations.id.name
  - translations.en.text = English verse translations
  - tafsir.id unchanged (original Indonesian)
  - tafsir.en.name = English translation of translations.id.name; tafsir.en.text = English tafsir
API key and translation settings are read from config.yaml.
Uses Google Gen AI SDK (google-genai).
"""
import json
import re
import time
from copy import deepcopy
from pathlib import Path

import yaml

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")

# GEMINI_MODEL, SLEEP_BETWEEN_CALLS, etc. are set in main() from config.yaml (translation section)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def translate_name(client: genai.Client, id_name: str) -> str:
    """Translate the Indonesian translation/tafsir name (e.g. 'Pembukaan') to English."""
    if not id_name or not id_name.strip():
        return "English"
    prompt = f"""Translate the following Indonesian phrase to English. Output only the English phrase, nothing else. It is a surah or translation title.

Indonesian: {id_name.strip()}

English:"""
    response = _generate_with_retry(client, prompt)
    return (response.text or "English").strip() or "English"


def _generate_with_retry(client: genai.Client, prompt: str):
    """Call generate_content with retry on 429 (rate limit). Waits for response, then sleeps before returning so the next request is never sent until after this delay."""
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
            # Wait after each response before allowing the next translation request
            time.sleep(SLEEP_BETWEEN_CALLS)
            return response
        except ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < RATE_LIMIT_MAX_RETRIES:
                wait = RATE_LIMIT_BASE_WAIT * (2**attempt)
                print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{RATE_LIMIT_MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for rate limit")


def translate_verse_batch(client: genai.Client, verses: dict) -> dict:
    """Translate verse translations (ayah number -> Indonesian text) to English. Returns dict ayah -> English.
    If the verse block would exceed VERSE_BATCH_MAX_CHARS, splits into multiple requests by complete ayahs (no broken ayah in a chunk)."""
    if not verses:
        return {}
    sorted_verses = sorted(verses.items(), key=lambda x: int(x[0]))
    # Chunk by complete ayahs so each chunk's block length <= VERSE_BATCH_MAX_CHARS
    chunks = []
    current = {}
    current_len = 0
    for k, v in sorted_verses:
        line = f"{k}: {v}"
        line_len = len(line) + 1  # +1 for newline between lines
        if current and current_len + line_len > VERSE_BATCH_MAX_CHARS:
            chunks.append(current)
            current = {}
            current_len = 0
        current[k] = v
        current_len += line_len
    if current:
        chunks.append(current)

    prompt_prefix = """Translate the following Indonesian Quran verse translations to English. Keep the same numbering format: one line per verse, "number: translation". Output only the translated lines, no other text. Preserve meaning and tone (formal, reverent).

Indonesian:
"""
    prompt_suffix = """

English (same format):"""

    out = {}
    for i, chunk in enumerate(chunks):
        lines = [f"{k}: {v}" for k, v in sorted(chunk.items(), key=lambda x: int(x[0]))]
        block = "\n".join(lines)
        prompt = prompt_prefix + block + prompt_suffix
        if len(chunks) > 1:
            print(f"    Verse chunk {i + 1}/{len(chunks)} (ayahs {min(chunk)}–{max(chunk)}, {len(block)} chars)...")
        response = _generate_with_retry(client, prompt)
        text = (response.text or "").strip()
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[:\-]\s*(.*)$", line, re.DOTALL)
            if m:
                num, trans = m.group(1), m.group(2).strip()
                out[num] = trans
    # Fill any missing with original key to avoid losing verses
    for k in verses:
        if k not in out:
            out[k] = verses[k]
    return out


def translate_tafsir_chunk(client: genai.Client, text: str) -> str:
    """Translate one chunk of tafsir text to English."""
    if not text or not text.strip():
        return text
    prompt = f"""Translate the following Indonesian Islamic tafsir (Quran commentary) to English. Preserve meaning, paragraph breaks, and any Arabic terms or references (e.g. hadith citations). Keep the same structure.

Indonesian:
{text}

English:"""
    response = _generate_with_retry(client, prompt)
    return (response.text or "").strip()


def translate_tafsir_batch(client: genai.Client, tafsir_texts: dict) -> dict:
    """Translate tafsir.id.text (ayah -> long Indonesian text) to English. Returns dict ayah -> English. Each chunk waits for Gemini response and delay before the next."""
    if not tafsir_texts:
        return {}
    out = {}
    for ayah_num in sorted(tafsir_texts.keys(), key=int):
        text = tafsir_texts[ayah_num]
        out[ayah_num] = translate_tafsir_chunk(client, text)
    return out


def process_file(
    client: genai.Client,
    src_path: Path,
    out_dir: Path,
) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Each file has one top-level key = surah number (e.g. "1")
    keys = [k for k in data if k.isdigit()]
    if not keys:
        raise ValueError(f"No surah key in {src_path}")
    surah_key = keys[0]
    block = data[surah_key]

    translations_id = block.get("translations", {}).get("id", {})
    verse_texts = translations_id.get("text") or {}
    id_name = translations_id.get("name") or ""
    tafsir_id = block.get("tafsir", {}).get("id", {})
    tafsir_texts = tafsir_id.get("text") or {}

    # Translate the Indonesian name (e.g. "Pembukaan") to English for use in en.name
    print(f"  Translating name '{id_name}' to English...")
    en_name = translate_name(client, id_name)

    # Translate verse translations
    print(f"  Translating {len(verse_texts)} verses...")
    en_verse_texts = translate_verse_batch(client, verse_texts)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # Translate tafsir per ayah
    print(f"  Translating tafsir for {len(tafsir_texts)} ayahs...")
    en_tafsir_texts = translate_tafsir_batch(client, tafsir_texts)

    # Build output: deep copy keeps original tafsir.id; add translations.en and tafsir.en
    out_data = deepcopy(data)
    ob = out_data[surah_key]

    if "translations" not in ob:
        ob["translations"] = {}
    ob["translations"]["en"] = {
        "name": en_name,
        "text": en_verse_texts,
    }

    # Keep tafsir.id as original; add tafsir.en with English name and translated text
    if "tafsir" not in ob:
        ob["tafsir"] = {}
    ob["tafsir"]["en"] = {
        "name": en_name,
        "source": tafsir_id.get("source", "Indonesian Ministry of Religious Affairs Quran App"),
        "text": en_tafsir_texts,
    }

    out_path = out_dir / src_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)
    print(f"  Wrote {out_path.name}")


def main():
    global GEMINI_MODEL, TAFSIR_BATCH_CHARS, VERSE_BATCH_MAX_CHARS, SLEEP_BETWEEN_CALLS, RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_BASE_WAIT

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found. Copy config.example.yaml to config.yaml and set gemini.api_key. "
            f"Get a key at https://aistudio.google.com/apikey"
        )

    config = load_config(config_path) or {}
    api_key = (config.get("gemini") or {}).get("api_key") or ""
    api_key = api_key.strip() if isinstance(api_key, str) else ""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Invalid or missing gemini.api_key in config.yaml. "
            "Copy config.example.yaml to config.yaml and replace YOUR_GEMINI_API_KEY_HERE with your key from https://aistudio.google.com/apikey"
        )

    # Load translation settings from config (with defaults)
    t = config.get("translation") or {}
    GEMINI_MODEL = t.get("model", "gemini-2.0-flash")
    TAFSIR_BATCH_CHARS = int(t.get("tafsir_batch_chars", 25000))
    VERSE_BATCH_MAX_CHARS = int(t.get("verse_batch_max_chars", 25000))
    SLEEP_BETWEEN_CALLS = float(t.get("sleep_between_calls", 1.0))
    RATE_LIMIT_MAX_RETRIES = int(t.get("rate_limit_max_retries", 5))
    RATE_LIMIT_BASE_WAIT = int(t.get("rate_limit_base_wait", 10))

    client = genai.Client(api_key=api_key)

    src_dir = script_dir / "surah_cleaned"
    out_dir = script_dir / "surah_en"
    if not src_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for i, src_path in enumerate(files[3:115]):
        print(f"[{i + 1}/{len(files)}] {src_path.name}")
        try:
            process_file(client, src_path, out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            raise
    print("Done.")


if __name__ == "__main__":
    main()
