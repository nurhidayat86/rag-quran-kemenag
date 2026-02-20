"""
Build surah summary vector DB JSON using Google Gemini.
For each surah, sends the full English tafseer to Gemini and gets a summary (max 2000 chars).
Reads from surah_en_cleaned/quran_english_cleaned.json; uses config.yaml for API key and model.
Output: pre_vector_db/summary_vector.json
"""
import json
import time
from pathlib import Path

import yaml

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ClientError
except ImportError:
    raise ImportError("Install google-genai: pip install google-genai")

# Max summary length (characters) and max tafseer sent to Gemini (to stay within context limits)
MAX_SUMMARY_CHARS = 2000
MAX_TAFSEER_CHARS = 750_000  # ~200k tokens; truncate if surah tafseer is larger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_with_retry(client: genai.Client, prompt: str) -> str:
    """Call Gemini with retry on 429. Returns response text."""
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,  # ~2000 chars
                ),
            )
            time.sleep(SLEEP_BETWEEN_CALLS)
            return (response.text or "").strip()
        except ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < RATE_LIMIT_MAX_RETRIES:
                wait = RATE_LIMIT_BASE_WAIT * (2**attempt)
                print(
                    f"    Rate limited (429). Waiting {wait}s before retry "
                    f"{attempt + 1}/{RATE_LIMIT_MAX_RETRIES}...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for rate limit")


def main():
    global GEMINI_MODEL, SLEEP_BETWEEN_CALLS, RATE_LIMIT_MAX_RETRIES, RATE_LIMIT_BASE_WAIT

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "config.yaml not found. Copy config.example.yaml to config.yaml and set gemini.api_key."
        )

    config = load_config(config_path) or {}
    api_key = (config.get("gemini") or {}).get("api_key") or ""
    api_key = api_key.strip() if isinstance(api_key, str) else ""
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "Invalid or missing gemini.api_key in config.yaml. "
            "Get a key at https://aistudio.google.com/apikey"
        )

    t = config.get("translation") or {}
    GEMINI_MODEL = t.get("model", "gemini-2.0-flash")
    SLEEP_BETWEEN_CALLS = float(t.get("sleep_between_calls", 1.0))
    RATE_LIMIT_MAX_RETRIES = int(t.get("rate_limit_max_retries", 5))
    RATE_LIMIT_BASE_WAIT = int(t.get("rate_limit_base_wait", 60))

    src_path = script_dir / "surah_en_cleaned" / "quran_english_cleaned.json"
    out_dir = script_dir / "pre_vector_db"
    out_path = out_dir / "summary_vector.json"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        quran = json.load(f)

    surahs_with_tafseer = [
        (k, v) for k, v in sorted(quran.items(), key=lambda x: int(x[0])) if v.get("tafseer")
    ]
    total = len(surahs_with_tafseer)
    print(f"Summarizing {total} surahs with Gemini ({GEMINI_MODEL})...", flush=True)

    client = genai.Client(api_key=api_key)
    records = []

    for idx, (surah_num_str, surah_data) in enumerate(surahs_with_tafseer, start=1):
        surah_number = int(surah_num_str)
        name_latin = surah_data.get("name_latin", "")
        name_en = surah_data.get("name", "")
        tafseer = surah_data.get("tafseer") or {}
        ayah_nums = sorted(int(k) for k in tafseer.keys())
        if not ayah_nums:
            continue

        # Build full tafseer text: ayah 1 text + " (1)\n\n" + ayah 2 text + ...
        parts = []
        for a in ayah_nums:
            parts.append(tafseer[str(a)])
            parts.append(f" ({a})\n\n")
        full_tafseer = "".join(parts).rstrip()
        if not full_tafseer:
            continue

        if len(full_tafseer) > MAX_TAFSEER_CHARS:
            full_tafseer = full_tafseer[:MAX_TAFSEER_CHARS] + "\n\n[Text truncated for context limit.]"
            print(
                f"  Surah {surah_number} tafseer truncated to {MAX_TAFSEER_CHARS} chars",
                flush=True,
            )

        print(
            f"  [{idx}/{total}] Surah {surah_number} {name_latin} (ayahs 1–{ayah_nums[-1]}) ...",
            end=" ",
            flush=True,
        )

        prompt = f"""Summarize the following English tafseer (commentary) of Surah {name_latin} (Surah {surah_number}, {name_en}). The text covers all verses of the surah.

Requirements:
- Write a single, coherent summary in English.
- Maximum length: {MAX_SUMMARY_CHARS} characters.
- Capture the main themes, key teachings, and important narratives. Do not list verse-by-verse; synthesize into a flowing summary.
- Output only the summary, no preamble or labels.

Tafseer text:

{full_tafseer}

Summary:"""

        try:
            summary = _generate_with_retry(client, prompt)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            summary = ""

        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"

        record_id = f"surah_{surah_number}_ayah_1_summary_chunk_1"
        records.append(
            {
                "id": record_id,
                "text": summary,
                "metadata": {
                    "surah_number": surah_number,
                    "surah_name": name_latin,
                    "surah_name_en": name_en,
                    "ayah_number": list(range(1, ayah_nums[-1] + 1)),
                    "text_type": "summary",
                    "chunk_index": 1,
                },
            }
        )
        print(f"{len(summary)} chars", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(records)} summary records)")


if __name__ == "__main__":
    main()
