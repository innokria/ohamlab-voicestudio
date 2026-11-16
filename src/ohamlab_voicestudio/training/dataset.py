"""Dataset preflight for consented/appropriately licensed speech corpora.

Manifest JSONL rows: {"audio": "path.wav", "text": "हिंदी पाठ", "speaker": "id"}.
"""
import json
from pathlib import Path

def validate_manifest(manifest):
    manifest = Path(manifest)
    rows, speakers, errors = 0, set(), []
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            audio = Path(item["audio"])
            if not audio.is_absolute():
                audio = manifest.parent / audio
            if not audio.is_file():
                raise ValueError(f"missing audio: {audio}")
            if not str(item["text"]).strip():
                raise ValueError("empty text")
            speakers.add(str(item.get("speaker", "unknown")))
            rows += 1
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"line {lineno}: {exc}")
    return {"valid_rows": rows, "speakers": sorted(speakers), "errors": errors}
