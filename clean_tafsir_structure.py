"""
Clean tafsir structure: move data from [x]['tafsir']['id']['kemenag'] to [x]['tafsir']['id'].
Reads from surah/ (1.json .. 114.json), writes to surah_cleaned/ with same filenames.
"""
import json
from pathlib import Path


def clean_tafsir(data: dict, surah_key: str) -> None:
    """Replace tafsir.id with tafsir.id.kemenag content (in-place)."""
    try:
        tafsir_id = data[surah_key]["tafsir"]["id"]
        if "kemenag" in tafsir_id:
            data[surah_key]["tafsir"]["id"] = tafsir_id["kemenag"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Unexpected structure for surah {surah_key}") from e


def main():
    script_dir = Path(__file__).resolve().parent
    surah_dir = script_dir / "surah"
    out_dir = script_dir / "surah_cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, 115):
        filename = f"{n}.json"
        src = surah_dir / filename
        if not src.exists():
            print(f"Skip (missing): {filename}")
            continue

        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        key = str(n)
        if key not in data:
            print(f"Skip (no key '{key}'): {filename}")
            continue

        clean_tafsir(data, key)

        dst = out_dir / filename
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"OK: {filename} -> surah_cleaned/{filename}")

    print("Done.")


if __name__ == "__main__":
    main()
