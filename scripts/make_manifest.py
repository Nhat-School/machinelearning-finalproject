"""Hash source files and evidence, excluding caches, environments and Office lock files."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[]
for path in ROOT.rglob('*'):
    relative=path.relative_to(ROOT)
    if not path.is_file() or any(part in {'.venv','.cache','__pycache__','.git'} for part in relative.parts):continue
    if path.name.startswith('~$') or path.name in {'evidence_manifest.json','.DS_Store'} or path.suffix=='.pyc':continue
    files.append(path)
data={'algorithm':'sha256','files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(ROOT/'reports/evidence_manifest.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Manifest:',len(files),'files')
