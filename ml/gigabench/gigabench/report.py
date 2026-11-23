"""Self-contained HTML report: block summary, per-task charts and a browsable list of every answer."""
import html
import json
from pathlib import Path
from .io import read_jsonl

ROLE={'trained':'обучаемый блок','control':'контроль','diagnostic':'диагностика, мало примеров'}
METRIC={'balanced_accuracy':'сбалансированная','accuracy':'точность'}
EXPLANATIONS={'pending_coach_review':'ожидают утверждённых рубрик','not_included':'не входят в прогон',
              'coach_review_complete':'оценены по рубрикам'}

CSS='''
:root{--bg:#f7f6f2;--surface:#ffffff;--ink:#1b2420;--muted:#5d6b65;--line:#dcd8cc;--bar:#2f6f9f;--base:#9a9486;--good:#1f7a4d;--bad:#b4481a;--warn:#8a6d00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141917;--surface:#1c2320;--ink:#e9ece8;--muted:#a3aea8;--line:#323c37;--bar:#6fa8d6;--base:#8f8a7e;--good:#5fc28e;--bad:#ef8a5a;--warn:#e0c24a}}
:root[data-theme="dark"]{--bg:#141917;--surface:#1c2320;--ink:#e9ece8;--muted:#a3aea8;--line:#323c37;--bar:#6fa8d6;--base:#8f8a7e;--good:#5fc28e;--bad:#ef8a5a;--warn:#e0c24a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 12px}h3{font-size:16px;margin:0 0 8px}
.muted{color:var(--muted)}.note{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:12px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px}
.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}.tag{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-variant-numeric:tabular-nums}
th{font-size:13px;color:var(--muted);font-weight:600}td.num,th.num{text-align:right}
.flag{font-size:12px;color:var(--warn)}.ok{color:var(--good)}.no{color:var(--bad)}
.scroll{overflow-x:auto}.chart{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;margin:8px 0 16px}
.chart svg{width:100%;height:auto;display:block}.legend{font-size:13px;color:var(--muted);margin:4px 0 0}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}select,input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink)}
.answer{white-space:pre-wrap;max-width:420px}.prompt{max-width:360px;color:var(--muted)}
'''


def pct(value):
    return '—' if value is None else f'{value:.1f}%'


def chart(tasks,summary):
    """Horizontal bars of the primary metric; grey tick = constant-answer baseline or chance."""
    rows=[(name,summary['by_task'][name]) for name in tasks if summary['by_task'][name]['n']]
    if not rows:return ''
    height=28*len(rows)+30;label_w=300;plot_w=560;out=[f'<svg viewBox="0 0 {label_w+plot_w+70} {height}" role="img" aria-label="Основная метрика по задачам">']
    for i,v in enumerate((0,25,50,75,100)):
        x=label_w+plot_w*v/100
        out.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height-22}" stroke="var(--line)" stroke-width="1"/>')
        out.append(f'<text x="{x}" y="{height-6}" font-size="11" fill="var(--muted)" text-anchor="middle">{v}%</text>')
    for i,(name,m) in enumerate(rows):
        y=i*28+4;value=m['primary_percent'] or 0
        out.append(f'<text x="{label_w-8}" y="{y+15}" font-size="12" fill="var(--ink)" text-anchor="end">{html.escape(name)}</text>')
        out.append(f'<rect x="{label_w}" y="{y+4}" width="{max(plot_w*value/100,2)}" height="16" rx="4" fill="var(--bar)"><title>{html.escape(name)}: {pct(m["primary_percent"])}</title></rect>')
        out.append(f'<text x="{label_w+plot_w*value/100+6}" y="{y+16}" font-size="12" fill="var(--ink)">{pct(m["primary_percent"])}</text>')
        base=50 if m['primary_metric']=='balanced_accuracy' else m.get('majority_target_baseline_percent')
        if base is not None:
            x=label_w+plot_w*base/100
            out.append(f'<line x1="{x}" y1="{y+1}" x2="{x}" y2="{y+23}" stroke="var(--base)" stroke-width="2"><title>уровень постоянного ответа: {base:.1f}%</title></line>')
    out.append('</svg>')
    return '<div class="chart">'+''.join(out)+'<p class="legend">Полоса — основная метрика задачи. Серая черта — результат постоянного одинакового ответа: 50% для сбалансированной точности, доля самого частого эталона для обычной.</p></div>'


def task_table(tasks,summary):
    head='<tr><th>Задача</th><th class="num">Примеров</th><th>Метрика</th><th class="num">Значение</th><th class="num">95% ДИ</th><th class="num">Точность</th><th class="num">Постоянный ответ</th><th class="num">Формат</th><th class="num">Обрезано</th><th>Пометки</th></tr>'
    body=[]
    for name in tasks:
        m=summary['by_task'][name]
        ci=m['balanced_95ci_percent'] if m['primary_metric']=='balanced_accuracy' else m['cluster_95ci_percent']
        flags=[]
        if m['n']<30:flags.append('меньше 30 примеров')
        if m.get('single_answer_class'):flags.append('эталон одного класса')
        if m['truncated']:flags.append(f"обрезано: {m['truncated']}")
        classes=m.get('by_answer_class')
        if classes:flags.append('; '.join(f"{k}: {pct(v['accuracy_percent'])} из {v['n']}" for k,v in classes.items()))
        body.append(f"<tr><td>{html.escape(name)}</td><td class='num'>{m['n']}</td><td>{METRIC[m['primary_metric']]}</td>"
                    f"<td class='num'><b>{pct(m['primary_percent'])}</b></td><td class='num'>{'—' if not ci else f'{ci[0]:.0f}–{ci[1]:.0f}'}</td>"
                    f"<td class='num'>{pct(m['accuracy_percent'])}</td><td class='num'>{pct(m.get('majority_target_baseline_percent'))}</td>"
                    f"<td class='num'>{pct(m['protocol_compliance_percent'])}</td><td class='num'>{m['truncated']}</td><td class='flag'>{html.escape(' · '.join(flags))}</td></tr>")
    return '<div class="scroll"><table>'+head+''.join(body)+'</table></div>'


def render_html(out,summary,config):
    out=Path(out);names=config.get('block_names',{})
    cases={c['id']:c for c in read_jsonl(out/'cases.jsonl')}
    responses={r['id']:r for r in read_jsonl(out/'responses.jsonl')} if (out/'responses.jsonl').exists() else {}
    scores=read_jsonl(out/'scores.jsonl') if (out/'scores.jsonl').exists() else []
    task_block={t:b for b,v in summary['by_block'].items() for t in v['tasks']}
    for combined,parts in config.get('combined_tasks',{}).items():
        for p in parts:task_block.setdefault(p,task_block.get(combined))
    records=[]
    for s in scores:
        c=cases[s['id']];r=responses.get(s['id'],{})
        records.append({'id':s['id'],'task':s['suite']+'/'+s['task'],'block':names.get(task_block.get(s['suite']+'/'+s['task']),''),
                        'prompt':c['prompt'],'fen':c['fen'],'answer':r.get('text',''),'expected':json.dumps(c['gold'],ensure_ascii=False),
                        'predicted':json.dumps(s.get('predicted'),ensure_ascii=False),'correct':s['correct'],
                        'truncated':s['truncated'],'format':s['protocol_compliant'],'pieces':s.get('piece_bin','')})
    parts=['<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Отчёт бенчмарка GigaChess</title><style>'+CSS+'</style></head><body><main>',
           '<h1>Отчёт бенчмарка GigaChess</h1>']
    if summary.get('is_mock'):parts.append('<p class="note no"><b>MOCK:</b> технический прогон, не результат модели.</p>')
    if summary.get('rescored'):
        rs=summary['rescored']
        parts.append(f"<p class='note'>Пересчёт метрик кодом {html.escape(rs['scoring_version'])} для прогона на задачах версии {html.escape(rs['cases_version'])}. Ответы модели те же, изменились только метрики. Оценки движка: {html.escape(rs['engine_verdicts'])}.</p>")
    cov=summary['coverage']
    status=EXPLANATIONS.get(summary['explanations_status'],summary['explanations_status'])
    parts.append(f"<p class='muted'>Ответов: {cov['completed']} из {cov['expected']}. Объяснения: {html.escape(status)}.</p>")
    parts.append('<h2>Блоки</h2><div class="cards">')
    for block,v in summary['by_block'].items():
        bins=' · '.join(f"{html.escape(k)}: {pct(x)}" for k,x in v['by_piece_bin'].items())
        parts.append(f"<div class='card'><div class='tag'>{ROLE.get(v['role'],'')}</div><h3>{html.escape(names.get(block,block))}</h3>"
                     f"<div class='big'>{pct(v['mean_primary_percent'])}</div><div class='muted'>среднее по {v['scored_tasks']} задачам</div>"
                     f"<div class='muted' style='font-size:13px;margin-top:6px'>{bins}</div></div>")
    parts.append('</div><p class="legend">Значение блока — среднее основной метрики его задач с равными весами. Ниже под каждым блоком: результат по числу фигур на доске.</p>')
    for block,v in summary['by_block'].items():
        if not v['tasks']:continue
        parts.append(f"<h2>{html.escape(names.get(block,block))}</h2>")
        parts.append(chart(v['tasks'],summary));parts.append(task_table(v['tasks'],summary))
    parts.append('<h2>Все ответы</h2><div class="filters"><select id="fb"><option value="">Все блоки</option></select>'
                 '<select id="ft"><option value="">Все задачи</option></select><select id="fr"><option value="">Любой результат</option>'
                 '<option value="1">Верно</option><option value="0">Неверно</option><option value="t">Обрезано</option><option value="f">Нарушен формат</option></select>'
                 '<input id="fq" type="search" placeholder="Поиск по тексту"></div><p class="muted" id="count"></p>'
                 '<div class="scroll"><table><thead><tr><th>Задача</th><th>Вопрос</th><th>Ответ модели</th><th>Эталон</th><th>Распознано</th><th>Итог</th></tr></thead><tbody id="rows"></tbody></table></div>')
    data=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
    parts.append('<script>const D='+data+''';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fb=document.getElementById('fb'),ft=document.getElementById('ft'),fr=document.getElementById('fr'),fq=document.getElementById('fq');
[...new Set(D.map(d=>d.block))].filter(Boolean).sort().forEach(b=>fb.add(new Option(b,b)));
[...new Set(D.map(d=>d.task))].sort().forEach(t=>ft.add(new Option(t,t)));
function draw(){const q=fq.value.toLowerCase();const rows=D.filter(d=>(!fb.value||d.block==fb.value)&&(!ft.value||d.task==ft.value)&&
 (!fr.value||(fr.value=='1'&&d.correct===true)||(fr.value=='0'&&d.correct===false)||(fr.value=='t'&&d.truncated)||(fr.value=='f'&&!d.format))&&
 (!q||(d.prompt+' '+d.answer+' '+d.id).toLowerCase().includes(q)));
 document.getElementById('count').textContent='Показано '+rows.length+' из '+D.length+(rows.length>500?'; выведены первые 500':'');
 document.getElementById('rows').innerHTML=rows.slice(0,500).map(d=>'<tr><td>'+esc(d.task)+'<div class="muted" style="font-size:12px">'+esc(d.id)+(d.pieces?' · '+esc(d.pieces):'')+'</div></td><td class="prompt">'+esc(d.prompt)+'<div style="font-size:12px">'+esc(d.fen)+'</div></td><td class="answer">'+esc(d.answer)+(d.truncated?' <span class="flag">[обрезано]</span>':'')+(d.format?'':' <span class="flag">[формат]</span>')+'</td><td>'+esc(d.expected)+'</td><td>'+esc(d.predicted)+'</td><td class="'+(d.correct===true?'ok':d.correct===false?'no':'muted')+'">'+(d.correct===true?'верно':d.correct===false?'неверно':'не оценено')+'</td></tr>').join('');}
[fb,ft,fr].forEach(e=>e.onchange=draw);fq.oninput=draw;draw();
</script></main></body></html>''')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')
