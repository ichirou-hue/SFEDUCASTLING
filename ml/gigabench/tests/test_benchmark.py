import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import chess
from gigabench.cases import build, FIXTURES, legacy_cases
from gigabench.oracles import mating_moves, mate_within_two
from gigabench.parsing import parse_move, parse_piece, parse_bool, parse_set, compliant
from gigabench.scoring import score_case, piece_f1
from gigabench.coaches import validate_rubrics, score_one
from gigabench.io import digest,read_jsonl
from gigabench.pipeline import prepare,generate,score,verify
from gigabench.publishing import package_public
from gigabench.engine import EngineJudge
from gigabench.comparison import compare


CONFIG=json.loads((ROOT/'config/default.json').read_text())


@contextmanager
def temporary_workspace():
    parent=ROOT/'runs/test_tmp'
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        if parent.resolve() not in Path(tmp).resolve().parents:
            raise ValueError('Test cleanup path escaped intended directory')
        yield tmp


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build(CONFIG,'smoke')
        cls.by_id={r['id']:r for r in cls.rows}

    def test_deterministic_generation(self):
        self.assertEqual(self.rows,build(CONFIG,'smoke'))

    def test_unique_and_valid_positions(self):
        self.assertEqual(len(self.rows),len(self.by_id))
        self.assertTrue(all(chess.Board(r['fen']).is_valid() for r in self.rows))

    def test_generated_games_replay(self):
        for r in self.rows:
            if r['provenance']['source']=='independent_random_legal_play':
                b=chess.Board()
                for uci in r['provenance']['moves']:
                    m=chess.Move.from_uci(uci);self.assertIn(m,b.legal_moves);b.push(m)
                self.assertEqual(b.fen(),r['fen'])

    def test_castling_and_ep_edge_cases(self):
        expected={'castle_clear':True,'castle_through_attack':False,'castle_in_check':False,
                  'en_passant':True,'en_passant_exposes_king':False,'promotion':True,'absolute_pin':False}
        for name,value in expected.items():
            self.assertEqual(self.by_id['fixture:'+name+':special']['gold']['answer'],value,name)

    def test_pin_attacks_are_not_legal_moves(self):
        fen=next(f for n,f,_ in FIXTURES if n=='absolute_pin');b=chess.Board(fen)
        self.assertTrue(b.is_pinned(chess.WHITE,chess.E2))
        self.assertIn(chess.A2,b.attacks(chess.E2))
        self.assertNotIn(chess.Move.from_uci('e2a2'),b.legal_moves)

    def test_stalemate_is_not_mate(self):
        self.assertTrue(self.by_id['fixture:stalemate:stalemate']['gold']['answer'])
        self.assertFalse(self.by_id['fixture:stalemate:checkmate']['gold']['answer'])

    def test_every_mate_one_solution_is_accepted(self):
        for r in self.rows:
            if r['task']=='mate_in_one':
                b=chess.Board(r['fen'])
                for uci in r['gold']['moves']:
                    a=b.copy();a.push_uci(uci);self.assertTrue(a.is_checkmate())
                    self.assertTrue(score_case(r,{'text':uci})['correct'])

    def test_mate_two_has_both_positive_and_negative(self):
        rows=[r for r in self.rows if r['task']=='mate_within_two']
        self.assertEqual(sum(bool(r['gold']['moves']) for r in rows),len(rows)//2)
        for r in rows:
            self.assertEqual(mate_within_two(chess.Board(r['fen'])),r['gold']['moves'])

    def test_known_distant_king_is_not_mate_two(self):
        self.assertEqual(mate_within_two(chess.Board('7k/8/8/8/8/8/8/KQ6 w - - 0 1')),[])

    def test_tracking_targets_replayed(self):
        for r in self.rows:
            if r['task']=='state_tracking':
                b=chess.Board(r['fen'])
                for m in r['params']['sequence']:b.push_uci(m)
                p=b.piece_at(chess.parse_square(r['gold']['square']))
                self.assertEqual(p.symbol() if p else '.',r['gold']['label'])

    def test_no_answer_booleans_in_choice_text(self):
        for r in self.rows:
            if r['score_type']=='choice':
                self.assertNotIn('True',r['prompt']);self.assertNotIn('False',r['prompt'])
                self.assertIn(r['gold']['answer'],'ABCD')


class ParserTests(unittest.TestCase):
    def test_contradiction_over_whole_piece_answer(self):
        self.assertIsNone(parse_piece('На e4 белая пешка. Клетка e4 пуста.','e4'))
        self.assertIsNone(parse_piece('На e4 белая пешка, а на d4 чёрный конь.','e4'))
        self.assertEqual(parse_piece('Клетка e4 пуста.','e4'),'.')

    def test_move_wrappers_and_alternatives(self):
        f=chess.STARTING_FEN
        for s in ('e2e4','1e2e4','Можно сыграть e2e4.','e2–e4'):
            self.assertEqual(parse_move(f,s),'e2e4')
        self.assertIsNone(parse_move(f,'e2e4 или d2d4'))
        self.assertIsNone(parse_move(f,'Не играйте e2e4'))
        self.assertIsNone(parse_move(f,'e2e5'))
        self.assertEqual(parse_move(f,'Кf3'),'g1f3')

    def test_underpromotion_suffix(self):
        f=next(f for n,f,_ in FIXTURES if n=='promotion')
        self.assertEqual(parse_move(f,'a7a8n'),'a7a8n')

    def test_bool_contradiction(self):
        self.assertTrue(parse_bool('Да, король под шахом.'))
        self.assertFalse(parse_bool('Нет.'))
        self.assertIsNone(parse_bool('Да. Нет.'))

    def test_format_vs_content(self):
        r={'id':'x','suite':'x','task':'x','group_id':'g','fen':chess.STARTING_FEN,
           'score_type':'human_move','gold':{'move':'e2e4'}}
        s=score_case(r,{'text':'Можно сыграть e2e4.'})
        self.assertTrue(s['correct']);self.assertFalse(s['protocol_compliant'])

    def test_missing_move_list_is_not_correct_empty_set(self):
        r={'id':'x','suite':'x','task':'x','group_id':'g','fen':chess.STARTING_FEN,
           'score_type':'move_set','gold':{'moves':[]}}
        self.assertFalse(score_case(r,{'text':''})['correct'])
        self.assertEqual(score_case(r,{'text':''})['f1'],0)
        self.assertTrue(score_case(r,{'text':'нет'})['correct'])

    def test_list_false_positives_and_duplicates(self):
        r={'id':'x','suite':'x','task':'x','group_id':'g','fen':chess.STARTING_FEN,
           'score_type':'move_set','gold':{'moves':['e2e4']}}
        s=score_case(r,{'text':'e2e4,d2d4'})
        self.assertFalse(s['correct']);self.assertAlmostEqual(s['f1'],2/3)
        self.assertFalse(compliant('move_set','e2e4,e2e4'))

    def test_model_error_cannot_pass_negative_label(self):
        r={'id':'x','suite':'x','task':'x','group_id':'g','fen':chess.STARTING_FEN,
           'score_type':'bool','gold':{'answer':False}}
        self.assertFalse(score_case(r,{'text':'нет','error':{'type':'failure'}})['correct'])

    def test_reject_is_false_negative_in_f1(self):
        s=piece_f1([{'expected':'P','predicted':'P'},{'expected':'P','predicted':None}])
        self.assertAlmostEqual(s['by_label']['P']['f1_percent'],200/3)


class CoachTests(unittest.TestCase):
    def rubric(self):
        return {'id':'x','coach_approved':True,'approved_by':'coach','approved_at':'2026-11-01T12:00:00+00:00',
                'learner_level':'beginner','required_ideas':['a','b','c'],'allowed_facts':['a']}

    def rating(self,text='response'):
        return {'id':'x','response_sha256':digest(text),'factual_reviewer':'fact reviewer',
                'claims':[{'text':str(i),'verdict':'confirmed' if i<4 else 'incorrect'} for i in range(5)],
                'idea_scores':[1,1,0],'critical_error':True,
                'coach_ratings':[{'coach':'A','criteria':[1,1,1,1]},{'coach':'B','criteria':[1,1,1,1]}]}

    def test_document_metric_example(self):
        r=score_one('response',self.rubric(),self.rating())
        self.assertEqual(r['M4'],.8);self.assertAlmostEqual(r['M5'],2/3)
        self.assertEqual(r['M6'],1);self.assertFalse(r['accepted'])

    def test_independent_raters_and_response_binding(self):
        r=self.rating();r['coach_ratings'][1]['coach']='A'
        with self.assertRaises(ValueError):score_one('response',self.rubric(),r)
        with self.assertRaises(ValueError):score_one('different',self.rubric(),self.rating())

    def test_disagreement_requires_third(self):
        r=self.rating();r['coach_ratings'][1]['criteria']=[0,0,0,0]
        with self.assertRaises(ValueError):score_one('response',self.rubric(),r)

    def test_empty_reply_cannot_get_good_teaching_score(self):
        r=score_one('',self.rubric(),self.rating(''))
        self.assertEqual(r['M4'],0);self.assertEqual(r['M7'],0);self.assertFalse(r['accepted'])

    def test_duplicate_rubrics_rejected(self):
        case={'id':'x','score_type':'explanation'}
        with self.assertRaises(ValueError):validate_rubrics([self.rubric(),self.rubric()],[case])


class PipelineTests(unittest.TestCase):
    def args(self,out):
        return argparse.Namespace(out=Path(out),config=ROOT/'config/default.json',profile='smoke',
            without_legacy=True,legacy_source=None,backend='mock',project_root=None,adapter=None,
            stage='automatic',rubrics=None,resume=False,limit=None,stockfish=None)

    def test_complete_mock_flow_and_gold_separation(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');prepare(a);generate(a);result=score(a)
            self.assertTrue(result['is_mock']);self.assertTrue(result['coverage']['complete'])
            for q in read_jsonl(a.out/'queries.jsonl'):
                self.assertEqual(set(q),{'id','fen','prompt','max_new_tokens'})
            self.assertIsNone(result['document_metrics']['M4'])

    def test_modified_dataset_is_rejected(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');prepare(a)
            with (a.out/'cases.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaises(ValueError):verify(a.out)

    def test_partial_run_never_has_total_score(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');a.limit=3
            prepare(a);generate(a);result=score(a)
            self.assertFalse(result['coverage']['complete'])
            self.assertIsNone(result['automatic_task_macro_percent'])

    def test_resume_empty_response_file(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');prepare(a)
            (a.out/'responses.jsonl').touch();a.resume=True
            generate(a)
            self.assertEqual(len(read_jsonl(a.out/'responses.jsonl')),len(read_jsonl(a.out/'queries.jsonl')))

    def test_explanations_refuse_inference_without_approved_rubrics(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');prepare(a);a.stage='explanations'
            with self.assertRaises(ValueError):generate(a)

    def test_compare_rejects_mock_as_model_results(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'a');prepare(a);generate(a);score(a)
            b=self.args(Path(tmp)/'b');prepare(b);generate(b);score(b)
            with self.assertRaises(ValueError):compare(a.out,b.out,Path(tmp)/'comparison.json')

    def test_public_package_excludes_private_sources(self):
        with temporary_workspace() as tmp:
            z=Path(tmp)/'public.zip';package_public(ROOT,z)
            with ZipFile(z) as archive:
                self.assertFalse(any('/private/' in n or n.endswith(('.docx','.nnue','.safetensors')) or n.startswith('runs/') for n in archive.namelist()))
                self.assertIn('PUBLIC_MANIFEST.json',archive.namelist())

    @unittest.skipUnless((ROOT/'data/private/legacy_cases.jsonl').exists(),'Local corpus intentionally absent in public repository')
    def test_legacy_size_and_reference_types(self):
        rows=legacy_cases(ROOT/'data/private/legacy_cases.jsonl')
        self.assertEqual(len(rows),435)
        self.assertEqual(sum(r['score_type']=='human_move' for r in rows),79)
        self.assertEqual(sum(r['score_type']=='explanation' for r in rows),89)
        self.assertTrue(all(r['private'] for r in rows))

    @unittest.skipUnless(os.environ.get('STOCKFISH_EXECUTABLE'),'Optional actual-engine integration')
    def test_actual_stockfish_quality_and_mate_separation(self):
        with temporary_workspace() as tmp:
            e=EngineJudge(os.environ['STOCKFISH_EXECUTABLE'],CONFIG['engine'],tmp)
            try:
                r=e.grade(chess.STARTING_FEN,None)
                self.assertTrue(r['ordinary']);self.assertFalse(r['accepted'])
                f=next(f for n,f,_ in FIXTURES if n=='queen_mate_white')
                r=e.grade(f,mating_moves(chess.Board(f))[0])
                self.assertFalse(r['ordinary']);self.assertIsNone(r['loss_cp'])
            finally:e.close()


if __name__=='__main__':unittest.main()
