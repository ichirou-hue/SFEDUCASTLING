import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package_public(root,destination):
    root=Path(root);destination=Path(destination)
    names=['run.py','README.md','METRICS.md','PROVENANCE.md','LICENSE','requirements.txt','.gitignore']
    paths=[root/n for n in names]
    for folder,pattern in [('gigabench','*.py'),('tests','*.py'),('config','*.json'),('licenses','*.txt'),('.github','*.yml')]:
        paths.extend((root/folder).rglob(pattern))
    paths=sorted(set(paths))
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        if any(v in p.relative_to(root).parts for v in ('private','runs','__pycache__')):
            raise ValueError('Private/generated path in publication list')
    manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'x',compression=ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(root)).replace('\\','/'))
        z.writestr('PUBLIC_MANIFEST.json',json.dumps({'scope':'code/config/tests/docs only; no private comments, Word files, weights or results','sha256':manifest},ensure_ascii=False,indent=2))
    return {'path':str(destination),'files':len(paths)+1,'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
