import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package_public(root,destination):
    root=Path(root);destination=Path(destination)
    names=['run.py','METRICS.md','PROVENANCE.md','LICENSE']
    paths=[root/n for n in names]
    for folder,pattern in [('gigabench','*.py'),('config','*.json'),('licenses','*.txt'),('.github','*.yml')]:
        paths.extend((root/folder).rglob(pattern))
    paths=sorted(set(paths))
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        if any(v in p.relative_to(root).parts for v in ('private','runs','__pycache__')):
            raise ValueError('Private/generated path in publication list')
    manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    readme=root.resolve().parents[1]/'README.md'
    if not readme.is_file():raise FileNotFoundError(readme)
    manifest['README.md']=hashlib.sha256(readme.read_bytes()).hexdigest()
    ignore=root.resolve().parents[1]/'.gitignore'
    manifest['.gitignore']=hashlib.sha256(ignore.read_bytes()).hexdigest()
    requirements=root.resolve().parents[1]/'requirements.txt'
    manifest['requirements.txt']=hashlib.sha256(requirements.read_bytes()).hexdigest()
    test_files=sorted((root.resolve().parents[1]/'tests/ml/gigabench').glob('*.py'))
    for test in test_files:
        manifest['tests/'+test.name]=hashlib.sha256(test.read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'x',compression=ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(root)).replace('\\','/'))
        for test in test_files:z.write(test,'tests/'+test.name)
        z.write(readme,'README.md')
        z.write(ignore,'.gitignore')
        z.write(requirements,'requirements.txt')
        z.writestr('PUBLIC_MANIFEST.json',json.dumps({'scope':'code/config/tests/docs only; no private comments, Word files, weights or results','sha256':manifest},ensure_ascii=False,indent=2))
    return {'path':str(destination),'files':len(paths)+len(test_files)+4,'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
