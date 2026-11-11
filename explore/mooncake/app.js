'use strict';
const $ = (s) => document.querySelector(s);
const sourceMap = Object.fromEntries(sources.map(s=>[s.id,s]));
function refs(ids){return `<div class="source-chips">${ids.map(id=>{const s=sourceMap[id];return `<a href="${s.url}" target="_blank" rel="noopener" title="${s.path}:${s.line}">[${s.n}] ${s.title} ↗</a>`}).join('')}</div>`}
for(const id of ['commit-side','commit-hero','full-sha']){const a=document.getElementById(id);a.href=`https://github.com/kvcache-ai/Mooncake/commit/${SHA}`;if(id!=='commit-hero')a.textContent=id==='full-sha'?SHA:SHA.slice(0,12)}
$('#short-sha').textContent=SHA.slice(0,12);
document.querySelectorAll('[data-source]').forEach(a=>{a.href=sourceMap[a.dataset.source].url;a.target='_blank';a.rel='noopener'});
let selectedComponent='master', scenario='all', currentFlow='put', step=0;
$('#component-list').innerHTML=components.map(c=>`<button data-component="${c.id}" aria-pressed="false">${c.name}<small>${c.en.split(' · ')[0]}</small></button>`).join('');
function selectComponent(id,scroll=false){const c=components.find(c=>c.id===id);if(!c)return;selectedComponent=id;document.querySelectorAll('[data-component]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.component===id));$('#component-detail').innerHTML=`<div class="detail-kicker">${c.en}</div><h3>${c.name}</h3><p class="summary">${c.summary}</p><div class="contracts"><div><label>输入 / 它接收什么</label><p>${c.input}</p></div><div><label>输出 / 它交付什么</label><p>${c.output}</p></div></div><ul class="detail-points">${c.points.map(p=>`<li>${p}</li>`).join('')}</ul><div class="detail-boundary"><strong>责任边界</strong>${c.boundary}</div>${refs(c.refs)}`;document.querySelectorAll('.node').forEach(n=>n.classList.toggle('selected',n.dataset.node===id));if(scroll)$('#explorer').scrollIntoView({behavior:'smooth'});}
$('#component-list').addEventListener('click',e=>{const b=e.target.closest('button');if(b)selectComponent(b.dataset.component)});
const nodes = [
['conductor',40,25,220,68,'Conductor','缓存前缀索引','control'],
['framework',350,25,340,68,'推理框架 / Connector','KV · 权重 · 语义与调度','dark'],
['pg',775,25,220,68,'PG · Process Group','集合通信 / 成员状态','normal'],
['reshard',40,160,190,68,'Reshard','逻辑布局与传输计划','normal'],
['master',265,160,220,68,'Store Master','对象目录 / 分配 / 租约','control'],
['client',510,160,220,68,'Store Client','对象操作 / 存储资源','normal'],
['ep',775,160,220,68,'EP · Expert Parallel','token dispatch / combine','normal'],
['p2p',40,280,190,68,'P2P Store','Go 分片传播服务','normal'],
['rs',265,280,220,68,'Rust Store','authority mesh / owners','normal'],
['tent',510,280,220,68,'TENT · 可选实现','策略 / 调度 / failover','control'],
['metadata',775,280,220,68,'Metadata / Storage','发现与控制状态 / 存储层','control'],
['te',265,410,465,75,'Transfer Engine','注册内存 · 寻址 · 批量异步传输','dark']
];
const edges = [
['framework','te','M690 77 H745 V110 H1030 V523 H600 V485','data','直接传输（PD 等）',800,519],
['reshard','te','M40 194 H20 V470 H265','data','布局绑定后执行',46,462],
['conductor','framework','M260 59 H350','control','索引',302,49],
['framework','pg','M690 59 H775','data','通信',730,49],
['pg','ep','M885 93 V160','control','成员',896,132],
['framework','client','M620 93 V160','data','对象请求',635,130],
['master','client','M485 194 H510','control','',0,0],
['framework','reshard','M430 93 V124 H135 V160','data','布局规划',190,116],
['framework','rs','M520 93 V139 H248 V259 H375 V280','data','可选 Store',290,251],
['client','te','M720 228 H747 V447 H730','data','payload',754,382],
['rs','te','M375 348 V410','data','payload',388,382],
['p2p','te','M135 348 V447 H265','data','payload',170,437],
['tent','te','M620 348 V410','control','可选 runtime',630,382],
['ep','te','M995 194 H1015 V505 H620 V485','data','可复用 transport / 设备快速路径',785,496],
['metadata','te','M775 314 H763 V468 H730','control','',0,0]
];
const svg=$('#arch-svg');
svg.innerHTML=`<defs><marker id="arrow-data" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#478877"/></marker><marker id="arrow-control" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#9482a7"/></marker></defs><text x="40" y="394" font-size="10" fill="#9aa394" font-family="monospace">DATA MOVEMENT LAYER</text>${edges.map(([a,b,d,type,label,x,y])=>`<g class="arch-edge" data-from="${a}" data-to="${b}"><path d="${d}" fill="none" stroke="${type==='control'?'#9482a7':'#478877'}" stroke-width="1.5" ${type==='control'?'stroke-dasharray="4 4"':''} marker-end="url(#arrow-${type})"/>${label?`<text x="${x}" y="${y}" font-size="10" fill="#7c8a7b" text-anchor="${a==='conductor'||b==='pg'?'middle':'start'}">${label}</text>`:''}</g>`).join('')}${nodes.map(([id,x,y,w,h,title,sub,type])=>`<g class="node" data-node="${id}" role="button" tabindex="0" aria-label="查看 ${title} 的组件解析"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${type==='dark'?'#1b4841':type==='control'?'#eeebf1':'#eaf0e5'}" stroke="${type==='dark'?'#1b4841':type==='control'?'#d3cadc':'#cbd8c3'}"/><text x="${x+w/2}" y="${y+28}" text-anchor="middle" fill="${type==='dark'?'#fafbf4':'#24483d'}" font-size="15" font-weight="600">${title}</text><text x="${x+w/2}" y="${y+49}" text-anchor="middle" fill="${type==='dark'?'#b4c9b8':'#7a8878'}" font-size="10">${sub}</text></g>`).join('')}`;
svg.addEventListener('click',e=>{const n=e.target.closest('[data-node]');if(n)selectComponent(n.dataset.node,true)});
svg.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.dataset.node){e.preventDefault();selectComponent(e.target.dataset.node,true)}});
$('#scenario-tabs').innerHTML=scenarios.map(s=>`<button data-scenario="${s.id}" aria-pressed="false">${s.name}</button>`).join('');
function selectScenario(id){scenario=id;const s=scenarios.find(s=>s.id===id);document.querySelectorAll('[data-scenario]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.scenario===id));document.querySelectorAll('[data-node]').forEach(n=>n.classList.toggle('dim',!s.nodes.includes(n.dataset.node)));document.querySelectorAll('.arch-edge').forEach(n=>n.classList.toggle('dim',!s.nodes.includes(n.dataset.from)||!s.nodes.includes(n.dataset.to)));$('#scenario-description').innerHTML=`<b>${s.title}</b><p>${s.text}</p>`;}
$('#scenario-tabs').addEventListener('click',e=>{const b=e.target.closest('button');if(b)selectScenario(b.dataset.scenario)});
$('#flow-tabs').innerHTML=flows.map(f=>`<button data-flow="${f.id}" aria-pressed="false">${f.name}</button>`).join('');
function renderFlow(){const f=flows.find(f=>f.id===currentFlow);document.querySelectorAll('[data-flow]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.flow===f.id));$('#flow-subtitle').textContent=f.subtitle;$('#step-count').textContent=`${String(step+1).padStart(2,'0')} / ${String(f.steps.length).padStart(2,'0')}`;$('#step-track').innerHTML=f.steps.map((s,i)=>`<button data-step="${i}" class="${i===step?'active':i<step?'done':''}" aria-current="${i===step?'step':'false'}" aria-label="第 ${i+1} 步：${s[0]}"><span>${String(i+1).padStart(2,'0')}</span>${s[0]}</button>`).join('');const [name,actor,lane,type,description,code,ids]=f.steps[step];$('#step-detail').innerHTML=`<div class="step-lane"><span class="lane-tag ${type}">${lane}</span><b>${actor}</b><span>${type==='data'?'关注字节与完成状态':'关注位置、状态与契约'}</span></div><div><h3>${name}</h3><p>${description}</p><code>${code}</code>${refs(ids)}</div>`;$('#flow-caveat').textContent=f.caveat;$('#prev-step').disabled=step===0;$('#next-step').disabled=step===f.steps.length-1;}
$('#flow-tabs').addEventListener('click',e=>{const b=e.target.closest('button');if(b){currentFlow=b.dataset.flow;step=0;renderFlow()}});
$('#step-track').addEventListener('click',e=>{const b=e.target.closest('button');if(b){step=Number(b.dataset.step);renderFlow()}});
$('#prev-step').addEventListener('click',()=>{if(step>0){step--;renderFlow()}});$('#next-step').addEventListener('click',()=>{const f=flows.find(f=>f.id===currentFlow);if(step<f.steps.length-1){step++;renderFlow()}});
$('#compare-rows').innerHTML=comparisons.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('');
$('#boundary-grid').innerHTML=boundaries.map(([n,t,p,id])=>`<article class="boundary-card"><span>${n} / ARCHITECTURE BOUNDARY</span><h3>${t}</h3><p>${p}</p><a href="${sourceMap[id].url}" target="_blank" rel="noopener">源码依据 [${sourceMap[id].n}] ↗</a></article>`).join('');
$('#choices').innerHTML=choices.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('');
$('#failure-grid').innerHTML=failures.map(([t,p])=>`<article><h4>${t}</h4><p>${p}</p></article>`).join('');
$('#reading-route').innerHTML=[['01','先建立对象与地址的区别','整体架构 → TE 接口 → Store 接口','overview'],['02','追一条完整读写路径','Client::Put / Get → Master → transfer','put'],['03','理解运行时决策','selector → queue → completion / failover','selector'],['04','扩展到分布式控制','Store-RS / Conductor / PG / Reshard','rs-arch']].map(([n,t,p,id])=>`<a href="${sourceMap[id].url}" target="_blank" rel="noopener"><span>READING ORDER / ${n}</span><b>${t} ↗</b><small>${p}</small></a>`).join('');
function renderSources(query=''){const q=query.trim().toLowerCase();const found=sources.filter(s=>`${s.title} ${s.path}`.toLowerCase().includes(q));$('#source-count').textContent=`${found.length} / ${sources.length} 处固定版本证据 · 链接直接定位源码行`;$('#source-list').innerHTML=found.length?found.map(s=>`<a class="source-row" href="${s.url}" target="_blank" rel="noopener"><span>${String(s.n).padStart(2,'0')}</span><div><b>${s.title}</b><small>${s.path} : ${s.line}</small></div><em>↗</em></a>`).join(''):'<p class="empty">没有匹配结果。试试 Store、TENT、路由或 transfer。</p>';}
$('#source-search').addEventListener('input',e=>renderSources(e.target.value));
$('#glossary').innerHTML=glossary.map(([t,p])=>`<details><summary>${t}</summary><p>${p}</p></details>`).join('');
const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash===`#${e.target.id}`));}})},{rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('main>section').forEach(s=>observer.observe(s));
selectScenario('all');selectComponent('master');renderFlow();renderSources();
