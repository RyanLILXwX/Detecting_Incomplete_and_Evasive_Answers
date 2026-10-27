import json
import random
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "first_round_data"
RAW_FILE = DATA_DIR / "raw/evasionbench_17k_eva4b_labels_dedup.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "full_processed_data"
SEED = 43
SPLIT_SIZES = {"train": 12000, "validation": 2000, "test": 2126, "human_test": 600}
EXISTING_FILES = {
    "train": [DATA_DIR / "LLM_annotations/llm_train_2700_annotated/train_2700.jsonl"], 
    "validation": [DATA_DIR / "LLM_annotations/llm_dev_test_300_annotated/dev_300.jsonl"], 
    "test": [DATA_DIR / "LLM_annotations/llm_dev_test_300_annotated/test_300.jsonl"], 
    "human_test": [DATA_DIR / "human_annotated_data/first_round" / f"batch_{number:02d}.jsonl" for number in range(1, 7)]
}

def read_jsonl(path: Path):
    """Read nonempty JSONL lines into a list of records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if (line.strip())]

def split_remaining(raw_by_uid: dict[str, dict], split_uids: dict[str, list[str]]):
    """
    Fill each split to its target size using stratified sampling with a fixed seed.
    Update the UID lists in place while preserving existing memberships.
    """
    assigned_uids = {uid for uids in split_uids.values() for uid in uids}
    remaining_rows = [row for uid, row in raw_by_uid.items() if (uid not in assigned_uids)]
    rng = random.Random(SEED)
    rng.shuffle(remaining_rows)
    pools = {label: [] for label in ("direct", "intermediate", "fully_evasive")}
    for row in remaining_rows:
        pools[row["eva4b_label"]].append(row["uid"])
    for split, target in SPLIT_SIZES.items():
        needed = target - len(split_uids[split])
        if (needed == 0):
            continue
        remaining_count = sum(len(pool) for pool in pools.values())
        quotas = {label: needed * len(pool) // remaining_count for label, pool in pools.items()}
        remainder_order = sorted(pools, key=lambda label: -(needed * len(pools[label]) % remaining_count))
        for label in remainder_order[:needed - sum(quotas.values())]:
            quotas[label] += 1
        new_uids = []
        for label, pool in pools.items():
            count = quotas[label]
            new_uids.extend(pool[:count])
            del pool[:count]
        rng.shuffle(new_uids)
        split_uids[split].extend(new_uids)

def clean_record(row: dict, annotation_by: str):
    """Keep the seven agreed fields and flatten per-question labels, preserving missing values as None."""
    sub_question_labels = None
    if (row.get("annotations") is not None):
        sub_question_labels = [annotation["label"] for annotation in row["annotations"]]
    return {
        "uid": row["uid"], 
        "question": row["question"], 
        "answer": row["answer"], 
        "annotation_by": annotation_by, 
        "overall_label": row.get("overall_label"), 
        "sub_questions": row.get("sub_questions"), 
        "sub_question_labels": sub_question_labels
    }

def write_jsonl(path: Path, rows: list[dict]):
    """Write records as UTF-8 JSONL without a trailing newline."""
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

def main():
    """Preserve split memberships and fill target sizes, then write labeled records first with the assigned annotation method."""
    raw_rows = pd.read_parquet(RAW_FILE, columns=["uid", "question", "answer", "eva4b_label"]).to_dict("records")
    raw_by_uid = {row["uid"]: row for row in raw_rows}
    if (len(raw_by_uid) != len(raw_rows)):
        raise ValueError("Raw data contains duplicate UIDs.")
    if (sum(SPLIT_SIZES.values()) != len(raw_rows)):
        raise ValueError("Target split sizes must add up to the raw record count.")
    existing_rows, split_uids = {}, {}
    for split, paths in EXISTING_FILES.items():
        split_uids[split] = []
        for path in paths:
            for row in read_jsonl(path):
                uid = row["uid"]
                if (uid in existing_rows or uid not in raw_by_uid):
                    raise ValueError(f"Duplicate or unknown existing UID: {uid}")
                if (any(row[key] != raw_by_uid[uid][key] for key in ("question", "answer"))):
                    raise ValueError(f"Existing question or answer differs from raw data: {uid}")
                existing_rows[uid] = row
                split_uids[split].append(uid)
        if (len(split_uids[split]) > SPLIT_SIZES[split]):
            raise ValueError(f"Existing {split} records exceed the target size.")
    split_remaining(raw_by_uid, split_uids)
    all_uids = [uid for uids in split_uids.values() for uid in uids]
    if (len(all_uids) != len(set(all_uids)) or set(all_uids) != set(raw_by_uid)):
        raise ValueError("Final splits must cover every raw UID exactly once.")
    if (any(len(split_uids[split]) != target for split, target in SPLIT_SIZES.items())):
        raise ValueError("Final split sizes do not match the targets.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, uids in split_uids.items():
        annotation_by = "llm"
        if (split == "human_test"):
            annotation_by = "human"
        rows = []
        for uid in uids:
            row = existing_rows.get(uid, raw_by_uid[uid])
            rows.append(clean_record(row, annotation_by))
        rows.sort(key=lambda row: row["overall_label"] in (None, ""))
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} records")

if __name__ == "__main__":
    main()