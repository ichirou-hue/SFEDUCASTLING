"""Fail when public backend or static files differ from CI."""
import hashlib
import json
import sys
from urllib.request import Request, urlopen
base, sha = sys.argv[1:]
def fetch(path):
 with urlopen(Request(base.rstrip("/") + path + "?release=" + sha, headers={"Cache-Control": "no-cache"}), timeout=30) as response:
  return response.read()
manifest = json.loads(fetch("/version.json"))
assert manifest["sha"] == sha, "Frontend version mismatch"
health = json.loads(fetch("/api/health"))
assert health["status"] == "ok" and health["sha"] == sha, "Backend version mismatch"
for path, expected in manifest["files"].items():
 assert hashlib.sha256(fetch("/" + path)).hexdigest() == expected, "Asset mismatch: " + path
print("Verified backend SHA and", len(manifest["files"]), "frontend files:", sha)
