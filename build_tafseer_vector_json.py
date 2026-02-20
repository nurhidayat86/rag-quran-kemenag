"""
Build tafseer vector DB JSON from surah_en_cleaned/quran_english_cleaned.json.
Chunking: recursive character splitter, ~1800 chars per chunk, ~200 char overlap (~10–15%).
Split priority: \\n\\n, \\n, ". "
Each chunk ends with ayah number in brackets e.g. (1). Chunks spanning two ayahs include " (1) \\n\\n ... (2)."
Output: pre_vector_db/tafseer_vector.json
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

# ~400–512 tokens ≈ 1500–2000 chars; overlap ~50–100 tokens ≈ 10–15%
CHUNK_SIZE = 1800
OVERLAP_SIZE = 200
SEPARATORS = ["\n\n", "\n", ". "]


def find_last_separator(text: str, start: int, end: int) -> int:
    """Return position after the last separator in text[start:end], or end if none."""
    segment = text[start:end]
    best = -1
    for sep in SEPARATORS:
        pos = segment.rfind(sep)
        if pos != -1:
            # position in full text
            candidate = start + pos + len(sep)
            if candidate > best:
                best = candidate
    return best if best != -1 else end


def recursive_split_with_positions(
    text: str, chunk_size: int, overlap_size: int
) -> List[Tuple[str, int, int]]:
    """Split text into chunks (recursive: \\n\\n, \\n, . ). Returns list of (chunk_text, start, end)."""
    if not text or not text.strip():
        return []
    result = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                result.append((chunk, start, text_len))
            break
        split_after = find_last_separator(text, start, end)
        chunk = text[start:split_after].strip()
        if chunk:
            result.append((chunk, start, split_after))
        # Next window: overlap by overlap_size, but always advance past current chunk to avoid infinite loop
        next_start = split_after - overlap_size
        if next_start <= start:
            next_start = split_after
        start = next_start
        if start >= text_len:
            break
    return result


def ensure_ayah_marker_at_end(chunk: str, last_ayah: int) -> str:
    """Ensure chunk ends with ' (n).' for the last ayah. Don't duplicate if already present."""
    chunk = chunk.rstrip()
    # Already ends with " (digits)." or " (digits),"
    if re.search(r"\s\(\d+\)[.,;:]?$", chunk):
        return chunk
    if chunk.endswith(")"):
        # e.g. "... (1)" without period
        return chunk + "."
    return chunk + f" ({last_ayah})."


def get_ayahs_in_range(
    ayah_ranges: List[Tuple[int, int, int]], start: int, end: int
) -> List[int]:
    """Return sorted list of ayah numbers whose [range_start, range_end) overlaps [start, end)."""
    ayahs = []
    for r_start, r_end, ayah_num in ayah_ranges:
        if r_end > start and r_start < end:
            ayahs.append(ayah_num)
    return sorted(ayahs)


def main():
    script_dir = Path(__file__).resolve().parent
    src_path = script_dir / "surah_en_cleaned" / "quran_english_cleaned.json"
    out_dir = script_dir / "pre_vector_db"
    out_path = out_dir / "tafseer_vector.json"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        quran = json.load(f)

    # Count surahs that have tafseer for progress denominator
    surahs_with_tafseer = [
        k for k, v in quran.items()
        if v.get("tafseer")
    ]
    total_surahs = len(surahs_with_tafseer)
    print(f"Processing tafseer for {total_surahs} surahs...", flush=True)

    records = []
    processed = 0
    for surah_num_str, surah_data in sorted(quran.items(), key=lambda x: int(x[0])):
        surah_number = int(surah_num_str)
        name_latin = surah_data.get("name_latin", "")
        name_en = surah_data.get("name", "")
        tafseer = surah_data.get("tafseer") or {}

        if not tafseer:
            continue

        processed += 1
        ayah_nums = sorted(int(k) for k in tafseer.keys())
        ayah_range = f"{ayah_nums[0]}-{ayah_nums[-1]}" if ayah_nums else "0"
        print(
            f"  Surah {processed}/{total_surahs}: {surah_number} {name_latin} "
            f"(ayahs {ayah_range}) ...",
            end=" ",
            flush=True,
        )

        # Build full surah tafseer with ayah boundaries: "... (1)\n\n... (2)\n\n..."
        parts = []
        ayah_ranges = []  # (start, end, ayah_num) for each ayah's content span
        pos = 0
        for a in ayah_nums:
            seg = tafseer[str(a)]
            r_start = pos
            parts.append(seg)
            pos += len(seg)
            boundary = f" ({a})\n\n"
            parts.append(boundary)
            pos += len(boundary)
            ayah_ranges.append((r_start, r_start + len(seg), a))
        full_text = "".join(parts).rstrip()
        if not full_text:
            print("0 chunks (empty)", flush=True)
            continue

        chunks_with_pos = recursive_split_with_positions(
            full_text, CHUNK_SIZE, OVERLAP_SIZE
        )
        for chunk_index, (chunk_text, pos_start, pos_end) in enumerate(
            chunks_with_pos, start=1
        ):
            ayah_list = get_ayahs_in_range(ayah_ranges, pos_start, pos_end)
            if not ayah_list:
                continue
            last_ayah = ayah_list[-1]
            text_final = ensure_ayah_marker_at_end(chunk_text, last_ayah)
            ayah_suffix = "_".join(str(a) for a in ayah_list)
            record_id = f"surah_{surah_number}_ayah_{ayah_suffix}_tafseer_chunk_{chunk_index}"

            records.append(
                {
                    "id": record_id,
                    "text": text_final,
                    "metadata": {
                        "surah_number": surah_number,
                        "surah_name": name_latin,
                        "surah_name_en": name_en,
                        "ayah_number": ayah_list,
                        "text_type": "tafseer",
                        "chunk_index": chunk_index,
                    },
                }
            )
        print(f"{len(chunks_with_pos)} chunks", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(records)} tafseer chunks)")


if __name__ == "__main__":
    main()
