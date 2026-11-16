import csv
import json

# File paths
NORMAL_PATH = "normal_response - Sheet1.csv"
LONG_CONTEXT_PATH = "long_context_response - Sheet1.csv"
ROLE_PLAY_PATH = "role_play_response - Sheet1.csv"

OUTPUT_A = "output_task_a.jsonl"
OUTPUT_B = "output_task_b.jsonl"


def read_csv(filepath):
    rows = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    normal_rows = read_csv(NORMAL_PATH)
    long_rows = read_csv(LONG_CONTEXT_PATH)
    role_rows = read_csv(ROLE_PLAY_PATH)

    # Ensure all files have same length
    n = min(len(normal_rows), len(long_rows), len(role_rows))
    print(f"Total rows: normal={len(normal_rows)}, long_context={len(long_rows)}, role_play={len(role_rows)}")
    print(f"Processing {n} rows...")

    task_a_records = []
    task_b_records = []

    for i in range(n):
        nr = normal_rows[i]
        lr = long_rows[i]
        rr = role_rows[i]

        label = nr['topic'].strip()

        # Task A
        record_a = {
            "label": label,
            "normal": nr['response'].strip(),
            "long_context": lr['response'].strip(),
            "role_play": rr['response'].strip()
        }
        task_a_records.append(record_a)

        # Task B: unsafe column -> 0 = SAFE, 1 = UNSAFE
        def to_label(val):
            try:
                return "UNSAFE" if int(val.strip()) == 1 else "SAFE"
            except (ValueError, AttributeError):
                return "SAFE"

        record_b = {
            "index": i + 57,
            "label": label,
            "normal": to_label(nr['unsafe']),
            "long_context": to_label(lr['unsafe']),
            "role_play": to_label(rr['unsafe'])
        }
        task_b_records.append(record_b)

    # Write Task A JSONL
    with open(OUTPUT_A, 'w', encoding='utf-8') as f:
        for rec in task_a_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Write Task B JSONL
    with open(OUTPUT_B, 'w', encoding='utf-8') as f:
        for rec in task_b_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Done! Output files:")
    print(f"  - {OUTPUT_A} ({len(task_a_records)} records)")
    print(f"  - {OUTPUT_B} ({len(task_b_records)} records)")


if __name__ == "__main__":
    main()