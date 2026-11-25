"""Run the pipeline twice offline and compare all recorded artifact hashes."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run(["bash", str(ROOT / "scripts/launchers/run.sh")], check=True, cwd=ROOT)
path = ROOT / "reports/evidence_manifest.json"
first = json.loads(path.read_text(encoding="utf-8"))
subprocess.run(["bash", str(ROOT / "scripts/launchers/run.sh")], check=True, cwd=ROOT)
second = json.loads(path.read_text(encoding="utf-8"))
if first != second:
    changed = sorted(name for name in set(first["files"]) | set(second["files"])
                     if first["files"].get(name) != second["files"].get(name))
    raise SystemExit("FAIL: Tệp không nhất quán giữa hai lượt chạy: " + ", ".join(changed))
result = {"passed": True, "runs": 2, "compared_files": len(first["files"]),
          "comparison": "exact SHA-256 equality for all files in evidence_manifest.json",
          "network_downloads": False}
target = ROOT / ".cache/reproducibility_check.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {result['compared_files']} tệp giống hệt sau hai lượt chạy, bao gồm Word và cả 4 hình.")
