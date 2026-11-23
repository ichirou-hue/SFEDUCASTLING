import importlib.util
import sys
from pathlib import Path
from .io import sha

BUILTIN_PROFILES={
    'question_only':'{prompt}',
    'contract_v1_fen':'Позиция FEN: {fen}\n{prompt}'}


def input_builder(spec):
    """Return (function, metadata) that turns a frozen query into the model's text input.

    The profile belongs to the model, not to the benchmark: cases, gold and scoring stay
    frozen, and every run records which profile produced its inputs.
    `spec` is a built-in name or 'path/to/file.py:function' (for example a commentary
    template that adds engine facts exactly as in the training data).
    """
    if spec in BUILTIN_PROFILES:
        template=BUILTIN_PROFILES[spec]
        return (lambda q:template.format(prompt=q['prompt'],fen=q['fen'])),{'input_profile':spec,'template':template}
    path,_,name=spec.rpartition(':')
    if not path or not name or not Path(path).is_file():
        raise ValueError(f'Unknown input profile {spec!r}: use {sorted(BUILTIN_PROFILES)} or file.py:function')
    module_spec=importlib.util.spec_from_file_location('gigabench_input_profile',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return getattr(module,name),{'input_profile':spec,'builder_sha256':sha(path)}


class GigaChessAdapter:
    def __init__(self,project_root,config,adapter=None,profile='question_only'):
        root=Path(project_root).resolve()
        release=root/'releases/gigachess-control-v1'
        overlay=root/'code/interface_resolution_v1_20261018'
        if not (release/'gigachess_runtime.py').is_file():raise FileNotFoundError(release/'gigachess_runtime.py')
        for p in (release,overlay):sys.path.insert(0,str(p))
        import torch
        from gigachess_runtime import Runtime
        from run_candidate_interface import generate
        torch.manual_seed(config['seed'])
        self.runtime=Runtime(root,precision=config['precision'])
        self.generate=generate;self.profile=config['profile']
        self.build_input,profile_metadata=input_builder(profile)
        adapter_manifest=None
        if adapter:
            sys.path.insert(0,str(root/'tools'))
            from pilot_adapter_train import load_adapter
            adapter_manifest=load_adapter(self.runtime,Path(adapter))
        self.metadata={'backend':'gigachess','python':sys.version,'torch':torch.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'integrity':self.runtime.integrity,'profile':self.profile,
            'release_manifest_sha256':sha(release/'manifest.json'),
            'runtime_sha256':sha(release/'gigachess_runtime.py'),
            'interface_sha256':sha(overlay/'run_candidate_interface.py'),
            'codec_sha256':sha(overlay/'candidate_cell_codec.py'),
            'additional_adapter':adapter_manifest,'training_performed':False,**profile_metadata}

    def predict(self,query):
        result=self.generate(self.runtime,query['fen'],self.build_input(query),
                             profile=self.profile,max_new_tokens=query['max_new_tokens'])
        return {'text':result['decoded']['candidate_body_text'],'raw':result,
                'truncated':result['stop_reason']=='max_new_tokens'}


class MockAdapter:
    """Fixed invalid responses, NO access to gold. Not a model result."""
    metadata={'backend':'mock','not_a_model_evaluation':True,'training_performed':False}

    def __init__(self,profile='question_only'):
        self.build_input,profile_metadata=input_builder(profile)
        self.metadata={**MockAdapter.metadata,**profile_metadata}

    def predict(self,query):
        self.build_input(query)
        return {'text':'MOCK_INVALID','raw':None,'truncated':False}
