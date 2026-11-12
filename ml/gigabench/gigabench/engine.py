from pathlib import Path
import subprocess
import json
import chess
import chess.engine
from .io import digest, sha, write_json


class EngineJudge:
    def __init__(self, executable, config, workdir):
        self.config=config
        self.workdir=Path(workdir)
        self.workdir.mkdir(parents=True,exist_ok=True)
        self.engine=chess.engine.SimpleEngine.popen_uci(str(Path(executable).resolve()))
        self.engine.configure({'Threads':config['threads'],'Hash':config['hash_mb']})
        self.metadata={'id':self.engine.id,'binary_sha256':sha(executable),'settings':config,'networks':{}}
        previous_path=self.workdir/'engine.json'
        previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else None
        if previous and (previous['binary_sha256']!=self.metadata['binary_sha256'] or previous['settings']!=config):
            self.close()
            raise ValueError('Cannot change engine/budget for an existing scored run')
        if 'EvalFileSmall' in self.engine.options:
            self.close()
            raise RuntimeError('Use single-network Stockfish 19; dual-network engines need explicit separate NNUE setup')
        if 'EvalFile' not in self.engine.options:
            self.close()
            raise RuntimeError('Expected a Stockfish NNUE engine')
        original=str(self.engine.options['EvalFile'].default)
        export=self.workdir/'evaluation.nnue'
        if not export.exists():
            saved=subprocess.run([str(Path(executable).resolve())],
                input='export_net evaluation.nnue\nquit\n',text=True,capture_output=True,
                cwd=self.workdir,timeout=60)
            if saved.returncode:
                self.close()
                raise RuntimeError('NNUE export failed: '+saved.stderr)
        if not export.exists():
            self.close()
            raise RuntimeError('Engine cannot export its NNUE')
        self.engine.configure({'EvalFile':str(export.resolve())})
        self.metadata['networks']['EvalFile']={'default_name':original,'sha256':sha(export),'bytes':export.stat().st_size}
        if previous and previous['networks']!=self.metadata['networks']:
            self.close()
            raise ValueError('NNUE changed since previous scoring')
        self.cache={}
        write_json(self.workdir/'engine.json',self.metadata)

    def analyse(self,fen,uci=None):
        key=digest({'fen':fen,'uci':uci})
        if key not in self.cache:
            board=chess.Board(fen)
            self.engine.configure({'Clear Hash':None})
            extra={'root_moves':[chess.Move.from_uci(uci)]} if uci else {}
            result=self.engine.analyse(board,chess.engine.Limit(nodes=self.config['nodes']),game=object(),**extra)
            value=result['score'].pov(board.turn)
            self.cache[key]={'cp':value.score(),'mate':value.mate(),'depth':result.get('depth'),
                             'nodes':result.get('nodes'),'pv':[m.uci() for m in result.get('pv',[])[:16]]}
        return self.cache[key]

    def grade(self,fen,predicted):
        best=self.analyse(fen)
        played=self.analyse(fen,predicted) if predicted else None
        ordinary=best['cp'] is not None and (played is None or played['cp'] is not None)
        raw_loss=best['cp']-played['cp'] if ordinary and played else None
        loss=max(0,raw_loss) if raw_loss is not None else None
        return {'best':best,'played':played,'ordinary':ordinary,'raw_loss_cp':raw_loss,
                'loss_cp':loss,'accepted':bool(loss is not None and loss<=self.config['tolerance_cp']),
                'legal':predicted is not None}

    def close(self):
        if getattr(self,'engine',None):
            self.engine.quit();self.engine=None
