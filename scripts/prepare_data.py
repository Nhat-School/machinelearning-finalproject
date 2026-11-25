"""Download and audit two pinned public datasets using only Python's standard library."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ebf3695f01d2dec197253eb89499f4835bb7d0fa"
FILES = {"upenn": "All_Courses_REDACTED_CODED.csv", "stanford": "Stanford.csv"}
EXPECTED_SHA256 = {
    "upenn": "403301d079abcea1acc318b7a63f766658350dbc0e85eead79f7befc65aa48c3",
    "stanford": "31e8885939604a20cd0a903d1e5ebca811b850530b8e9ed5169e1936628142f2",
}
FIELDS = ["post_id", "source", "source_row", "source_id", "source_time", "source_course_type", "text_raw",
          "text_clean", "urgency", "priority", "word_count", "text_sha256"]
TAG = re.compile(r"</?(?:p|br|div|span|a|b|i|strong|em|ul|ol|li|blockquote|pre|code|hr)(?:\s[^<>]*?)?\s*/?>", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    return " ".join(TAG.sub(" ", unicodedata.normalize("NFC", html.unescape(text))).split())


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing raw files, never replace existing files")
    args = parser.parse_args()
    rejected, datasets, audit = [], {}, {"source_revision": REVISION, "label_rule": "priority = int(urgency > 4)", "datasets": {}}
    for source, filename in FILES.items():
        path = ROOT / "data/raw" / filename
        url = f"https://raw.githubusercontent.com/pcla-code/forum-posts-urgency/{REVISION}/{filename}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if not args.download:
                raise SystemExit(f"Thiếu {path.name}. Chạy bash Project/scripts/launchers/run.sh --download từ machine.")
            print(f"Đang tải {source} từ commit {REVISION[:12]}...", flush=True)
            temporary = path.with_suffix(".csv.part")
            try:
                subprocess.run(["curl", "-q", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "10", "--max-time", "120", "--retry", "1",
                                "--output", str(temporary), url], check=True)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(path) != EXPECTED_SHA256[source]:
            raise ValueError(f"SHA-256 tệp gốc không khớp phiên bản cố định: {filename}")
        groups = defaultdict(list)
        counts = Counter()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"post_text", "Urgency_1_7"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Schema không phù hợp: {path.name}")
            columns = reader.fieldnames
            for row_number, raw in enumerate(reader, start=2):
                counts["raw_rows"] += 1
                reason = None
                if None in raw or raw.get("post_text") is None or raw.get("Urgency_1_7") is None:
                    reason = "malformed_or_missing_fields"
                try:
                    urgency = float(raw.get("Urgency_1_7") or "nan")
                except (TypeError, ValueError):
                    urgency = float("nan")
                if reason is None and (not math.isfinite(urgency) or not 1 <= urgency <= 7):
                    reason = "missing_or_invalid_urgency"
                text = clean_text(raw.get("post_text") or "")
                if reason is None and not text:
                    reason = "empty_text"
                if reason:
                    counts[reason] += 1
                    rejected.append({"source": source, "source_row": row_number, "reason": reason})
                    continue
                counts["valid_rows"] += 1
                key = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
                groups[key].append({"post_id": f"{source}_{row_number:06d}", "source": source,
                                    "source_row": row_number, "source_id": raw.get("id", ""),
                                    "source_time": raw.get("post_time", ""),
                                    "source_course_type": raw.get("CourseType", ""), "text_raw": raw["post_text"],
                                    "text_clean": text, "urgency": urgency, "priority": int(urgency > 4),
                                    "word_count": len(text.split()), "text_sha256": key})
        retained = []
        for rows in groups.values():
            conflict = len({row["priority"] for row in rows}) > 1
            if conflict:
                counts["conflicting_groups"] += 1
            discarded = rows if conflict else rows[1:]
            reason = "conflicting_binary_labels" if conflict else "duplicate_within_source"
            counts[reason] += len(discarded)
            rejected.extend({"source": source, "source_row": row["source_row"], "reason": reason} for row in discarded)
            if not conflict:
                retained.append(rows[0])
        counts["after_within_source_dedup"] = len(retained)
        counts["cross_source_duplicates"] = 0
        datasets[source] = sorted(retained, key=lambda row: row["source_row"])
        audit["datasets"][source] = {"raw_file": str(path.relative_to(ROOT)), "url": url,
                                       "raw_sha256": sha256(path), "raw_bytes": path.stat().st_size,
                                       "columns": columns, "counts": dict(counts)}

    upenn_keys = {row["text_sha256"] for row in datasets["upenn"]}
    unique_stanford = []
    for row in datasets["stanford"]:
        if row["text_sha256"] in upenn_keys:
            audit["datasets"]["stanford"]["counts"]["cross_source_duplicates"] += 1
            rejected.append({"source": "stanford", "source_row": row["source_row"], "reason": "duplicate_across_sources"})
        else:
            unique_stanford.append(row)
    datasets["stanford"] = unique_stanford

    for source, rows in datasets.items():
        if not rows:
            raise ValueError(f"Không có dữ liệu hợp lệ: {source}")
        target = ROOT / "data/processed" / f"{source}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        item = audit["datasets"][source]
        item["counts"]["final_rows"] = len(rows)
        item["label_counts"] = dict(Counter(str(row["priority"]) for row in rows))
        item["urgency_counts"] = dict(Counter(str(row["urgency"]) for row in rows))
        lengths = [row["word_count"] for row in rows]
        item["word_lengths"] = {"min": min(lengths), "median": percentile(lengths, .5),
                                 "p95": percentile(lengths, .95), "max": max(lengths),
                                 "mean": round(sum(lengths) / len(lengths), 2)}
        bounds = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, None)]
        item["length_histogram"] = {f"{lower + 1}–{upper}" if upper else f">{lower}":
            sum(length > lower and (upper is None or length <= upper) for length in lengths)
            for lower, upper in bounds}
        item["processed_file"] = str(target.relative_to(ROOT))
        item["processed_sha256"] = sha256(target)
        n_rejected = sum(record["source"] == source for record in rejected)
        assert item["counts"]["raw_rows"] == len(rows) + n_rejected
        assert all(row["text_clean"].strip() for row in rows)
        assert all(math.isfinite(row["urgency"]) and 1 <= row["urgency"] <= 7 for row in rows)
        assert all(row["priority"] == int(row["urgency"] > 4) for row in rows)
        assert len({row["text_sha256"] for row in rows}) == len(rows)
        assert {row["priority"] for row in rows} == {0, 1}
        print(f"{source}: raw={item['counts']['raw_rows']}, final={len(rows)}, priority={item['label_counts'].get('1', 0)}", flush=True)
    assert not ({row["text_sha256"] for row in datasets["upenn"]} &
                {row["text_sha256"] for row in datasets["stanford"]})
    audit["checks"] = {"rows_accounted_for": True, "nonempty_text": True, "labels_in_range": True,
                        "both_binary_classes_present": True, "no_exact_duplicates_within_sources": True,
                        "no_exact_duplicates_between_sources": True}
    dump_json(ROOT / "reports/data_audit.json", audit)
    with (ROOT / "reports/rejected_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "source_row", "reason"])
        writer.writeheader()
        writer.writerows(sorted(rejected, key=lambda row: (row["source"], row["source_row"])))


if __name__ == "__main__":
    main()
