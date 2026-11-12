import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import chess
from gigabench.cases import build
from gigabench.generated import move_statements, battery, skewer_line, fork
from gigabench.external import bucket, puzzle_cases, game_cases
from gigabench.scoring import score_case, summarize, balanced_accuracy
from gigabench.adapters import input_builder
from gigabench.io import read_jsonl
from gigabench.pipeline import prepare, generate, score, rescore, export_heldout
from test_benchmark import temporary_workspace

CONFIG=json.loads((ROOT/'config/default.json').read_text())


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build(CONFIG,'quick')

    def test_every_generated_yes_no_task_is_balanced(self):
        counts=defaultdict(Counter)
        for r in self.rows:
            if r['score_type']=='bool' and r['suite']!='fixtures':
                counts[r['suite']+'/'+r['task']][r['gold']['answer']]+=1
        self.assertTrue(counts)
        for task,c in counts.items():self.assertEqual(c[True],c[False],task)

    def test_mates_balanced(self):
        for task in ('mate_in_one','mate_within_two'):
            rows=[r for r in self.rows if r['task']==task and r['suite']=='tactics']
            self.assertEqual(sum(bool(r['gold']['moves']) for r in rows)*2,len(rows),task)

    def test_rule_gold_recomputed(self):
        for r in self.rows:
            if r['suite']!='rules':continue
            b=chess.Board(r['fen'])
            if r['task']=='checkmate':self.assertEqual(b.is_checkmate(),r['gold']['answer'])
            if r['task']=='stalemate':self.assertEqual(b.is_stalemate(),r['gold']['answer'])
            if r['task']=='castling':
                self.assertEqual(chess.Move.from_uci(r['prompt'].split()[2+1]) in b.legal_moves,r['gold']['answer'])

    def test_check_and_motif_gold_recomputed(self):
        for r in self.rows:
            b=chess.Board(r['fen'])
            if r['suite']=='structural' and r['task']=='check':self.assertEqual(b.is_check(),r['gold']['answer'])
            if r['task'] in ('double_check','discovered_check') and r['suite']=='motifs':
                m=chess.Move.from_uci(r['params']['move']);a=b.copy();a.push(m);checkers=list(a.checkers())
                expected=len(checkers)==2 if r['task']=='double_check' else any(s!=m.to_square for s in checkers)
                self.assertEqual(expected,r['gold']['answer'],r['id'])

    def test_comment_choice_has_exactly_one_true_option(self):
        rows=[r for r in self.rows if r['task']=='comment_fact_choice']
        self.assertTrue(rows)
        for r in rows:
            b=chess.Board(r['fen']);m=chess.Move.from_uci(r['params']['move'])
            truth={t:v for t,f,v in move_statements(b,m)};truth.update({f:not v for t,f,v in move_statements(b,m)})
            body=r['prompt'].split(': ',1)[1].rsplit(' Ответь',1)[0]
            options=dict(re.findall(r'([A-D])\) (.*?)(?= [A-D]\) |$)',body))
            self.assertEqual(len(options),4,r['id'])
            true=[k for k,text in options.items() if truth[text]]
            self.assertEqual(true,[r['gold']['answer']],r['id'])
        self.assertGreater(len(Counter(r['gold']['answer'] for r in rows)),1)

    def test_paraphrases_share_gold_and_differ_in_wording(self):
        groups=defaultdict(list)
        for r in self.rows:
            if r['params'].get('paraphrase_of'):groups[r['params']['paraphrase_of']].append(r)
        self.assertTrue(groups)
        for base,rows in groups.items():
            self.assertEqual(len({json.dumps(r['gold'],sort_keys=True) for r in rows}),1,base)
            self.assertEqual(len({r['prompt'] for r in rows}),len(rows),base)

    def test_state_tracking_lengths_and_empty_targets(self):
        rows=[r for r in self.rows if r['task']=='state_tracking']
        self.assertTrue(any(r['gold']['label']=='.' for r in rows))
        self.assertTrue(all(1<=len(r['params']['sequence'])<=15 for r in rows))


class MotifDefinitionTests(unittest.TestCase):
    def test_fork_skewer_battery_examples(self):
        b=chess.Board('4k3/8/8/3N4/8/8/8/4K2R w - - 0 1')
        b.push_uci('d5c7');self.assertFalse(fork(b,chess.C7))
        f=chess.Board('r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1');f.push_uci('b5c7');self.assertTrue(fork(f,chess.C7))
        s=chess.Board('8/8/8/8/k7/8/q7/R3K3 w - - 0 1');self.assertTrue(skewer_line(s,chess.A1))
        bt=chess.Board('4k3/8/8/8/8/8/Q7/R3K3 w - - 0 1');self.assertTrue(battery(bt,chess.A1,chess.A2))
        self.assertFalse(battery(chess.Board('4k3/8/8/8/8/8/B7/R3K3 w - - 0 1'),chess.A1,chess.A2))


class ScoringTests(unittest.TestCase):
    def test_constant_answer_gets_fifty_percent_balanced(self):
        rows=[{'correct':not a,'answer_class':'да' if a else 'нет'} for a in [True]*10+[False]*90]
        self.assertEqual(balanced_accuracy(rows),50.0)

    def test_cells_combined_and_every_task_in_one_block(self):
        rows=build(CONFIG,'smoke')
        scored=[score_case(c,{'text':'нет'},None,CONFIG['piece_bins']) for c in rows if c['score_type']!='explanation']
        s=summarize(scored,CONFIG)
        self.assertIn('structural/cells',s['by_block']['board_reading']['tasks'])
        self.assertNotIn('structural/piece_empty',s['by_block']['board_reading']['tasks'])
        self.assertIn('fixtures/mate_in_one',s['by_block']['diagnostics']['tasks'])
        self.assertIsNotNone(s['by_task']['structural/cells']['balanced_accuracy_percent'])

    def test_judgement_buckets_exclude_boundaries(self):
        self.assertEqual(bucket(0,15),'D');self.assertIsNone(bucket(25,15));self.assertEqual(bucket(60,15),'C')
        self.assertEqual(bucket(-1000,15),'G');self.assertIsNone(bucket(305,15))
        case={'id':'j','suite':'commentary','task':'position_judgement','group_id':'g','fen':chess.STARTING_FEN,
              'score_type':'judgement','gold':{'answer':'B'},'params':{}}
        r=score_case(case,{'text':'C'});self.assertFalse(r['correct']);self.assertTrue(r['side_correct'])


class SourceTests(unittest.TestCase):
    def test_puzzle_csv_applies_opponent_move_first(self):
        with temporary_workspace() as tmp:
            path=Path(tmp)/'p.csv'
            path.write_text('PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n'
                            'x1,6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1,g8h8 d1d8,900,70,90,5000,mateIn1 short,,\n',encoding='utf-8')
            rows=puzzle_cases(path,CONFIG,'smoke',random.Random(1))
            self.assertEqual(rows[0]['gold']['moves'],['d1d8']);self.assertEqual(chess.Board(rows[0]['fen']).turn,chess.WHITE)
            self.assertTrue(score_case(rows[0],{'text':'d1d8'})['correct'])

    def test_game_positions_use_played_move_for_m2(self):
        with temporary_workspace() as tmp:
            path=Path(tmp)/'g.pgn'
            path.write_text('[Event "t"]\n[WhiteElo "1500"]\n[BlackElo "1700"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 *\n',encoding='utf-8')
            rows=game_cases(path,CONFIG,'smoke',random.Random(3))
            m2=[r for r in rows if r['task']=='M2']
            self.assertTrue(m2)
            for r in m2:self.assertIn(chess.Move.from_uci(r['gold']['move']),chess.Board(r['fen']).legal_moves)
            self.assertTrue(all(r['group_id'].startswith('game:') and r['group_id'].count(':')==1 for r in rows))


class PipelineV11Tests(unittest.TestCase):
    def args(self,out):
        return argparse.Namespace(out=Path(out),config=ROOT/'config/default.json',profile='smoke',without_legacy=True,
            legacy_source=None,backend='mock',project_root=None,adapter=None,stage='automatic',rubrics=None,resume=False,
            limit=None,stockfish=None,games=None,puzzles=None,with_judgement=False,input_profile='question_only')

    def test_input_profiles(self):
        q={'prompt':'Вопрос?','fen':chess.STARTING_FEN}
        self.assertEqual(input_builder('question_only')[0](q),'Вопрос?')
        self.assertTrue(input_builder('contract_v1_fen')[0](q).startswith('Позиция FEN: '))
        with temporary_workspace() as tmp:
            f=Path(tmp)/'builder.py';f.write_text("def build(q):\n    return 'X '+q['prompt']\n",encoding='utf-8')
            fn,meta=input_builder(f'{f}:build');self.assertEqual(fn(q),'X Вопрос?');self.assertIn('builder_sha256',meta)
        with self.assertRaises(ValueError):input_builder('nope')

    def test_rescore_and_heldout(self):
        with temporary_workspace() as tmp:
            a=self.args(Path(tmp)/'run');prepare(a);generate(a);score(a)
            r=argparse.Namespace(source=a.out,out=Path(tmp)/'rescored',config=ROOT/'config/default.json',stockfish=None)
            s=rescore(r)
            self.assertIn('rescored',s);self.assertTrue((Path(tmp)/'rescored/report.html').exists())
            held=export_heldout(a.out,Path(tmp)/'heldout.json')
            self.assertEqual(held['positions'],len({' '.join(c['fen'].split()[:4]) for c in read_jsonl(a.out/'cases.jsonl')}))
            run=json.loads((a.out/'run.json').read_text())
            self.assertEqual(run['backend']['input_profile'],'question_only')


if __name__=='__main__':unittest.main()
