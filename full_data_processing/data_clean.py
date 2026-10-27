import json
from pathlib import Path

def clean_failed_records():
    """Remove failed UIDs while preserving the remaining JSONL records."""
    data_dir = Path(__file__).resolve().parent / "full_processed_data_labeled"
    splits = ("train", "validation", "test")
    failed_uids = [json.loads(line)["uid"] for line in (data_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines() if (line.strip())]
    records = {}
    for split in splits:
        path = data_dir / f"{split}_labeled.jsonl"
        records[split] = [(json.loads(line)["uid"], line) for line in path.read_text(encoding="utf-8").splitlines() if (line.strip())]
    changed_splits = set()
    for uid in failed_uids:
        for split in splits:
            match_index = next((index for index, (row_uid, _) in enumerate(records[split]) if (row_uid == uid)), None)
            if (match_index is not None):
                del records[split][match_index]
                changed_splits.add(split)
                break
    for split in splits:
        if (split in changed_splits):
            path = data_dir / f"{split}_labeled.jsonl"
            path.write_text("\n".join(line for _, line in records[split]), encoding="utf-8")

if __name__ == "__main__":
    clean_failed_records()