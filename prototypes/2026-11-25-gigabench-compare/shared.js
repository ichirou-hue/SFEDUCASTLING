const variant=document.body.dataset.variant;
const blocks=[
 {name:'Чтение доски',old:1128,new:1502,total:5538},
 {name:'Выбор хода',old:16,new:3,total:568},
 {name:'Правила',old:200,new:200,total:400},
 {name:'Тактика',old:300,new:360,total:900},
 {name:'Комментарии',old:200,new:226,total:700},
 {name:'Диагностика',old:86,new:85,total:251},
 {name:'Дополнительная диагностика',old:null,new:12,total:40}
];
const pct=(n,d)=>(100*n/d).toLocaleString('ru-RU',{minimumFractionDigits:2,maximumFractionDigits:2})+' %';
const change=b=>b.old===null||b.new===null?null:100*(b.new-b.old)/b.total;
function delta(b){let d=change(b);return d===null?'<span class="neutral">Недоступно</span>':`<span class="delta ${d>0?'positive':d<0?'negative':'neutral'}">${d>0?'↑ +':d<0?'↓ −':'= '}${Math.abs(d).toLocaleString('ru-RU',{minimumFractionDigits:2,maximumFractionDigits:2})} п. п.</span>`}
const value=(b,k)=>b[k]===null?'Нет данных':`${b[k]} / ${b.total} · ${pct(b[k],b.total)}`;
document.querySelector('main').innerHTML=`
<nav aria-label="Варианты"><a href="../v1-table/index.html">01 · Таблица</a><a href="../v2-priorities/index.html">02 · Изменения</a><a href="../v3-explorer/index.html">03 · Обозреватель</a></nav>
<div class="eyebrow">GIGACHESS / GIGABENCH · ПРОТОТИП ${variant}</div>
<h1>Что изменилось в модели</h1>
<p>Старое: <b>GigaChat · baseline</b> → Новое: <b>GigaChat · candidate</b></p>
<p class="muted">Демонстрационные агрегаты · одинаковый набор и знаменатель · без задач и ответов</p>
<div class="tools"><label>Состояние отчёта<select id="state"><option value="ready">Готов</option><option value="empty">Нет отчётов</option><option value="loading">Загрузка</option><option value="error">Ошибка сравнения</option><option value="many">Много данных</option></select></label><label>Показать<select id="filter"><option value="all">Все блоки</option><option value="positive">Улучшения</option><option value="negative">Ухудшения</option><option value="neutral">Без изменений</option><option value="unavailable">Недоступные</option></select></label></div>
<div id="content" aria-live="polite"></div>
<footer>Дельта = новая доля − старая доля, в процентных пунктах, не в процентах прироста. ↑ улучшение · ↓ ухудшение · = без изменений. Недоступность не равна нулю. Все данные условные; это не результат реального запуска.</footer>`;
const content=document.querySelector('#content');
let selected=blocks[0].name;
function detail(b){return `<section class="panel"><div class="eyebrow">БЛОК</div><h2>${b.name}</h2><div class="metric">${delta(b)}</div><div class="pair">${['old','new'].map(k=>`<div><h3>${k==='old'?'Старое · baseline':'Новое · candidate'}</h3><p>${value(b,k)}</p>${b[k]!==null?`<div class="bar" aria-hidden="true"><span style="width:${b[k]/b.total*100}%"></span></div>`:''}</div>`).join('')}</div><p class="muted">${change(b)===null?'Сравнение невозможно: в старом отчёте нет блока. Не включён в общий итог.':'Знаменатель совпадает. Чем выше доля верных ответов, тем лучше.'}</p></section>`}
function render(){
 const state=document.querySelector('#state').value,f=document.querySelector('#filter').value;
 document.querySelector('#filter').disabled=['empty','loading','error'].includes(state);
 if(state==='empty'){content.innerHTML='<section class="panel state"><h2>Пока нечего сравнивать</h2><p>Нужны старый и новый отчёты GigaBench.</p><p class="muted">Этот макет не читает файлы. Выберите «Готов», чтобы открыть пример.</p></section>';return}
 if(state==='loading'){content.innerHTML='<section class="panel state" role="status"><h2>Сопоставляем блоки…</h2><div class="skeleton"></div><div class="skeleton"></div><p>Демонстрация загрузки. Готовый отчёт пока не показан.</p></section>';return}
 if(state==='error'){content.innerHTML='<section class="panel state" role="alert"><h2>Нельзя сравнить наборы</h2><p>Различаются версии набора или знаменатели блоков.</p><p>Проверьте входные отчёты. Частичные дельты не рассчитаны.</p><button id="retry">Вернуться к примеру</button></section>';document.querySelector('#retry').onclick=()=>{document.querySelector('#state').value='ready';render()};return}
 let data=state==='many'?[...blocks,...Array.from({length:24},(_,i)=>({name:`Демо-подблок ${String(i+1).padStart(2,'0')}`,old:10+i,new:12+i,total:100}))]:blocks;
 data=data.filter(b=>f==='all'||(f==='unavailable'?change(b)===null:f==='positive'?change(b)>0:f==='negative'?change(b)<0:change(b)===0));
 content.innerHTML=`<section class="hero"><div><div class="eyebrow">ОБЩИЙ РЕЗУЛЬТАТ</div><strong>23,09 % → 28,43 %</strong><p>1930 / 8357 → 2376 / 8357</p></div><div>${delta({old:1930,new:2376,total:8357})}<p class="muted">6 сопоставимых блоков</p></div></section><p class="notice">1 дополнительный блок недоступен для сравнения и не включён в общий итог.${state==='many'?' Ещё 24 синтетических подблока показывают плотность интерфейса; они не входят в итог.':''}</p><div id="results"></div>`;
 const target=document.querySelector('#results');
 if(!data.length){target.innerHTML='<p class="panel">По этому фильтру блоков нет.</p>';return}
 if(variant==='1')target.innerHTML=`<div class="panel table-wrap" tabindex="0" role="region" aria-label="Сравнение блоков, на узком экране прокручивается"><table><caption>Все блоки · ${data.length}</caption><thead><tr><th scope="col">Блок</th><th scope="col">Старое · baseline</th><th scope="col">Новое · candidate</th><th scope="col">Дельта, п. п.</th></tr></thead><tbody>${data.map(b=>`<tr><th scope="row">${b.name}</th><td>${value(b,'old')}</td><td>${value(b,'new')}</td><td>${delta(b)}</td></tr>`).join('')}</tbody></table></div>`;
 if(variant==='2'){
  const groups=[['Ухудшения','negative'],['Улучшения','positive'],['Без изменений','neutral'],['Недоступно','unavailable']];
  target.innerHTML=groups.map(([name,type])=>{const subset=data.filter(b=>type==='unavailable'?change(b)===null:type==='positive'?change(b)>0:type==='negative'?change(b)<0:change(b)===0);return subset.length?`<section><h2>${name} · ${subset.length}</h2><div class="cards">${subset.map(b=>`<article class="card"><h3>${b.name}</h3>${delta(b)}<p><span class="tag">Старое</span><br>${value(b,'old')}</p><p><span class="tag">Новое</span><br>${value(b,'new')}</p></article>`).join('')}</div></section>`:''}).join('');
 }
 if(variant==='3'){
  if(!data.some(b=>b.name===selected))selected=data[0].name;
  target.innerHTML=`<div class="split"><div class="block-nav" role="group" aria-label="Выберите блок">${data.map((b,i)=>`<button data-index="${i}" aria-pressed="${b.name===selected}"><span>${b.name}</span>${delta(b)}</button>`).join('')}</div><div id="detail">${detail(data.find(b=>b.name===selected))}</div></div>`;
  target.querySelectorAll('[data-index]').forEach(btn=>btn.onclick=()=>{selected=data[Number(btn.dataset.index)].name;target.querySelectorAll('[data-index]').forEach(x=>x.setAttribute('aria-pressed',String(x===btn)));document.querySelector('#detail').innerHTML=detail(data[Number(btn.dataset.index)])});
 }
}
document.querySelector('#state').onchange=render;
document.querySelector('#filter').onchange=render;
render();
