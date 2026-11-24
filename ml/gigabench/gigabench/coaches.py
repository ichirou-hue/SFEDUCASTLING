"""Explicit human annotations for M4-M7; never replace them with text similarity."""
from datetime import datetime
from .io import unique,digest


def pending_rubric(case):
    return {'id':case['id'],'coach_approved':False,'approved_by':None,'approved_at':None,
            'learner_level':None,'complex_terms':[],'required_ideas':[],
            'allowed_facts':[],'critical_error_examples':[]}


def validate_rubrics(rows,cases):
    data=unique(rows)
    expected={c['id'] for c in cases if c['score_type']=='explanation'}
    if set(data)!=expected:raise ValueError('Rubrics must cover exactly the explanation IDs')
    for r in rows:
        if not (r.get('coach_approved') is True and r.get('approved_by') and r.get('approved_at')
                and r.get('learner_level') and 2<=len(r.get('required_ideas',[]))<=5 and r.get('allowed_facts')):
            raise ValueError('Unapproved or incomplete rubric: '+r['id'])
        if any(not isinstance(s,str) or not s.strip() for s in r['required_ideas']):raise ValueError('Empty idea')
        stamp=datetime.fromisoformat(r['approved_at'])
        if stamp.tzinfo is None:raise ValueError('Approval must contain timezone')
    return data


def score_one(text,rubric,rating):
    if rating.get('response_sha256')!=digest(text):raise ValueError('Rating refers to different response')
    if not rating.get('factual_reviewer'):raise ValueError('Factual reviewer required')
    claims=rating['claims']
    if any(not c.get('text') or c.get('verdict') not in ('confirmed','incorrect','unconfirmed') for c in claims):
        raise ValueError('Invalid factual claims')
    ideas=rating['idea_scores']
    if len(ideas)!=len(rubric['required_ideas']) or any(type(v) is not int or v not in (0,1) for v in ideas):
        raise ValueError('One binary score per approved idea required')
    if type(rating['critical_error']) is not bool:raise ValueError('Critical error must be explicit')
    judges=rating['coach_ratings']
    if len(judges)!=2 or any(not j.get('coach') for j in judges) or judges[0]['coach']==judges[1]['coach']:
        raise ValueError('Two distinct coaches required')
    def total(j):
        if len(j['criteria'])!=4 or any(type(v) is not int or v not in (0,1) for v in j['criteria']):
            raise ValueError('Four binary teaching criteria required')
        return sum(j['criteria'])
    a,b=map(total,judges)
    if a!=b:
        j=rating.get('adjudication')
        if not j or not j.get('coach') or j['coach'] in {x['coach'] for x in judges}:raise ValueError('Independent third coach required')
        teaching=total(j)
    else:teaching=a
    empty=not text.strip()
    m4=sum(c['verdict']=='confirmed' for c in claims)/len(claims) if claims and not empty else 0
    m5=sum(ideas)/len(ideas) if not empty else 0
    m6=int(rating['critical_error']) if not empty else 0
    m7=teaching/4 if not empty else 0
    return {'id':rubric['id'],'M4':m4,'M5':m5,'M6':m6,'M7':m7,'empty':empty,
            'unconfirmed_claims':sum(c['verdict']=='unconfirmed' for c in claims),
            'accepted':bool(not empty and m4==1 and m5>=.8 and m6==0 and teaching>=3)}
