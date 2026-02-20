"""
Build translation vector DB JSON from surah_en_cleaned/quran_english_cleaned.json.
Chunking: 3 ayahs per chunk, sliding window (1-3, 2-4, 3-5, ...). No overlap in chunk size; overlap by 2 ayahs between consecutive chunks.
Output: pre_vector_db/translation_vector.json
"""
import json
from pathlib import Path
from typing import List


def build_translation_chunks(translation: dict) -> List[List[int]]:
    """Return list of ayah-number lists: each inner list is one chunk (3 ayahs, or fewer for short surahs)."""
    if not translation:
        return []
    ayah_nums = sorted(int(k) for k in translation.keys())
    if len(ayah_nums) <= 3:
        return [ayah_nums]
    # Sliding window of 3: (1,2,3), (2,3,4), ...
    return [ayah_nums[i : i + 3] for i in range(len(ayah_nums) - 2)]


def format_ayah_with_number(text: str, ayah_num: int) -> str:
    """Append ayah number in parentheses at end of ayah text, preserving trailing punctuation."""
    s = (text or "").strip()
    if not s:
        return f"({ayah_num})."
    if s[-1] in ".,;:":
        return s[:-1].rstrip() + f" ({ayah_num})" + s[-1]
    return s + f" ({ayah_num})."


def main():
    script_dir = Path(__file__).resolve().parent
    src_path = script_dir / "surah_en_cleaned" / "quran_english_cleaned.json"
    out_dir = script_dir / "pre_vector_db"
    out_path = out_dir / "translation_vector.json"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        quran = json.load(f)

    records = []
    for surah_num_str, surah_data in sorted(quran.items(), key=lambda x: int(x[0])):
        surah_number = int(surah_num_str)
        name_latin = surah_data.get("name_latin", "")
        name_en = surah_data.get("name", "")
        translation = surah_data.get("translation") or {}

        chunks = build_translation_chunks(translation)
        for chunk_index, ayah_nums in enumerate(chunks, start=1):
            # Exact text of each ayah with ayah number at end, joined with double newline
            text_parts = [
                format_ayah_with_number(translation[str(a)], a) for a in ayah_nums
            ]
            text = "\n\n".join(text_parts)

            # id e.g. surah_2_ayah_1_2_3_translation_chunk_1
            ayah_str = "_".join(str(a) for a in ayah_nums)
            record_id = f"surah_{surah_number}_ayah_{ayah_str}_translation_chunk_{chunk_index}"

            records.append({
                "id": record_id,
                "text": text,
                "metadata": {
                    "surah_number": surah_number,
                    "surah_name": name_latin,
                    "surah_name_en": name_en,
                    "ayah_number": ayah_nums,
                    "text_type": "translation",
                    "chunk_index": chunk_index,
                },
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(records)} translation chunks)")


if __name__ == "__main__":
    main()
