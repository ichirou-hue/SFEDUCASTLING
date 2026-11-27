import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def package_public(root,destination):
    root=Path(root);destination=Path(destination)
    standalone=(root/'PUBLIC_MANIFEST.json').is_file()
    project=root if standalone else root.resolve().parents[1]
    docs_root=root if standalone else project/'docs/ml/benchmark'
    paths=[root/n for n in ('run.py','LICENSE')]
    docs={name:docs_root/name
          for name in ('METRICS.md','PROVENANCE.md','VALIDATION.md','CHANGELOG.md')}
    for folder,pattern in [('gigabench','*.py'),('config','*.json'),('licenses','*.txt'),('.github','*.yml')]:
        paths.extend((root/folder).rglob(pattern))
    paths=sorted(set(paths))
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        if any(v in p.relative_to(root).parts for v in ('private','runs','__pycache__')):
            raise ValueError('Private/generated path in publication list')
    manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name,p in docs.items():
        manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    readme=root/'README.md' if standalone else docs_root/'public-package.md'
    if not readme.is_file():raise FileNotFoundError(readme)
    manifest['README.md']=hashlib.sha256(readme.read_bytes()).hexdigest()
    ignore=project/'.gitignore'
    manifest['.gitignore']=hashlib.sha256(ignore.read_bytes()).hexdigest()
    requirements=(project/'requirements.txt').read_bytes()
    if not standalone:
        # A standalone ZIP must not require the repo's editable Maia3 or training stack.
        pins=[line for line in requirements.splitlines() if line.startswith(b'chess==')]
        if len(pins)!=1:raise ValueError('Expected exactly one chess== pin in the project requirements.txt')
        requirements=pins[0]+b'\n'
    manifest['requirements.txt']=hashlib.sha256(requirements).hexdigest()
    test_root=root/'tests' if standalone else project/'tests/ml/gigabench'
    test_files=sorted(test_root.glob('*.py'))
    for test in test_files:
        manifest['tests/'+test.name]=hashlib.sha256(test.read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(destination,'x',compression=ZIP_DEFLATED) as z:
        for p in paths:z.write(p,str(p.relative_to(root)).replace('\\','/'))
        for name,p in docs.items():z.write(p,name)
        for test in test_files:z.write(test,'tests/'+test.name)
        z.write(readme,'README.md')
        z.write(ignore,'.gitignore')
        z.writestr('requirements.txt',requirements)
        z.writestr('PUBLIC_MANIFEST.json',json.dumps({'scope':'code/config/tests/docs only; no private comments, Word files, weights or results','sha256':manifest},ensure_ascii=False,indent=2))
    return {'path':str(destination),'files':len(paths)+len(docs)+len(test_files)+4,'bytes':destination.stat().st_size,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
