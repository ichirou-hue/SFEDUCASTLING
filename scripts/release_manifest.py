"""Record exact built bytes and source version."""
import hashlib
import json
import os
from pathlib import Path
root = Path(__file__).resolve().parents[1]
dist = root / "frontend/dist"
manifest = {"sha": os.environ["GITHUB_SHA"], "files": {
 p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(dist.rglob("*")) if p.is_file() and p.name != "version.json"
}}
(root / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
(dist / "version.json").write_text(json.dumps(manifest), encoding="utf-8")
