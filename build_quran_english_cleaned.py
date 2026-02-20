"""
Build a single cleaned JSON from surah_en/*.json (1.json through 114.json).
Output: surah_en_cleaned/quran_english_cleaned.json with surah number as parent index.
"""
import json
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / "surah_en"
    out_dir = script_dir / "surah_en_cleaned"
    out_path = out_dir / "quran_english_cleaned.json"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    for n in range(1, 115):
        surah_num = str(n)
        src_path = src_dir / f"{surah_num}.json"
        if not src_path.exists():
            print(f"  Skip (missing): {surah_num}.json")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Each file has one top-level key = surah number
        if surah_num not in data:
            keys = [k for k in data if k.isdigit()]
            if not keys:
                print(f"  Skip (no surah key): {surah_num}.json")
                continue
            block = data[keys[0]]
        else:
            block = data[surah_num]

        translations_en = block.get("translations", {}).get("en", {})
        tafsir_en = block.get("tafsir", {}).get("en", {})

        result[surah_num] = {
            "surah_number": surah_num,
            "name_latin": block.get("name_latin", ""),
            "translation": translations_en.get("text") or {},
            "tafseer": tafsir_en.get("text") or {},
            "name": translations_en.get("name", ""),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(result)} surahs)")


if __name__ == "__main__":
    main()
