"""Check the evidence manifest using only Python's standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "reports/evidence_manifest.json").read_text(encoding="utf-8"))
failures = []
for name, expected in manifest["files"].items():
    path = ROOT / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit("FAIL: " + ", ".join(failures))
print(f"PASS: {len(manifest['files'])} tệp khớp SHA-256 trong manifest.")
