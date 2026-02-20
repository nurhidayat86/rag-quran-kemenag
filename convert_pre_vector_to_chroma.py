"""
Check pre_vector_db/*.json for ChromaDB compatibility and convert if needed.
ChromaDB metadata allows: str, int, float, bool, and arrays of these (same type, non-empty).
Output: pre_chroma_db/*.json (only files that needed conversion, or all if --force).
"""
import json
import sys
from pathlib import Path
from typing import Any, List, Tuple

# Chroma allows: str, int, float, bool, List[str], List[int], List[float], List[bool] (non-empty)
ALLOWED_SCALARS = (str, int, float, bool)
ALLOWED_ARRAY_ELEMENT_TYPES = (str, int, float, bool)


def is_chroma_compatible_value(v: Any) -> bool:
    if v is None:
        return False
    if type(v) in ALLOWED_SCALARS:
        return True
    if isinstance(v, list):
        if len(v) == 0:
            return False  # Chroma disallows empty arrays
        return all(type(x) in ALLOWED_SCALARS for x in v)
    return False


def convert_value_to_chroma(v: Any) -> Any:
    """Convert a value to a ChromaDB-compatible type."""
    if v is None:
        return ""
    if type(v) in ALLOWED_SCALARS:
        return v
    if isinstance(v, list):
        if len(v) == 0:
            return ""
        if all(isinstance(x, (int, float)) for x in v):
            return [int(x) if isinstance(x, float) and x == int(x) else x for x in v]
        if all(isinstance(x, str) for x in v):
            return v
        if all(isinstance(x, bool) for x in v):
            return v
        # Mixed or other: serialize to comma-separated string
        return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def check_and_convert_record(record: dict) -> Tuple[bool, dict]:
    """Return (is_compatible, record). If not compatible, return converted record."""
    if not isinstance(record, dict) or "id" not in record or "text" not in record or "metadata" not in record:
        return False, record
    if not isinstance(record["id"], str) or not isinstance(record["text"], str):
        return False, record
    meta = record.get("metadata")
    if not isinstance(meta, dict):
        return False, record

    compatible = True
    for k, v in meta.items():
        if not is_chroma_compatible_value(v):
            compatible = False
            break

    if compatible:
        return True, record

    # Build converted metadata
    new_meta = {}
    for k, v in meta.items():
        new_meta[k] = convert_value_to_chroma(v)
    return False, {
        "id": record["id"],
        "text": record["text"],
        "metadata": new_meta,
    }


def process_file(src_path: Path, out_dir: Path, force_write_all: bool) -> Tuple[int, int, bool]:
    """Process one JSON file. Returns (total_records, converted_count, written)."""
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  Skip {src_path.name}: not a list of records")
        return 0, 0, False

    all_compatible = True
    converted = []
    for rec in data:
        ok, rec_out = check_and_convert_record(rec)
        if not ok:
            all_compatible = False
        converted.append(rec_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src_path.name
    # Write to pre_chroma_db if conversion was needed or if --force (so Chroma script has single source)
    should_write = not all_compatible or force_write_all
    if should_write:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
    return len(data), 0 if all_compatible else len(data), should_write


def main():
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir / "pre_vector_db"
    out_dir = script_dir / "pre_chroma_db"
    force = "--force" in sys.argv

    if not src_dir.exists():
        print(f"pre_vector_db not found: {src_dir}")
        sys.exit(1)

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        print("No JSON files in pre_vector_db")
        sys.exit(0)

    print("Checking pre_vector_db JSON files for ChromaDB compatibility...")
    total_records = 0
    converted_count = 0
    for path in json_files:
        n, c, written = process_file(path, out_dir, force)
        total_records += n
        converted_count += c
        if c > 0:
            print(f"  Converted {path.name} -> pre_chroma_db/ ({c} records)")
        elif written:
            print(f"  Copied {path.name} -> pre_chroma_db/ ({n} records, compatible)")
        elif n > 0:
            print(f"  OK {path.name} (compatible, {n} records, not written)")

    print(
        f"Done. Total records: {total_records}. "
        "Run build_chroma_db.py to load into Chroma (uses pre_chroma_db/ if present, else pre_vector_db/)."
    )


if __name__ == "__main__":
    main()
