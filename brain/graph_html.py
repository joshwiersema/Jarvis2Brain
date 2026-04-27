"""Inline HTML for the live brain visualization at /graph.

Streams updates over /graph/ws — new nodes appear in real time, training
epochs animate node positions toward learned xy, and a side panel shows the
live training-loss sparkline plus pending edge proposals from the NN.
"""

GRAPH_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jarvis Brain</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root {
    --bg: #07090f;
    --bg2: #0d1322;
    --border: #1a2540;
    --text: #e0e6f0;
    --muted: #6a7590;
    --accent: #5a8cff;
    --accent2: #b48aff;
    --warn: #ffba5a;
    --good: #5affb4;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: radial-gradient(ellipse at top, #101830 0%, var(--bg) 70%);
    color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    overflow: hidden; height: 100vh;
  }
  #graph { position: absolute; inset: 0; }
  #header {
    position: fixed; top: 0; left: 0; right: 0; padding: 12px 20px;
    background: rgba(7, 9, 15, 0.75); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border); z-index: 10;
    display: flex; align-items: center; gap: 16px;
  }
  #header h1 {
    margin: 0; font-size: 15px; font-weight: 600; letter-spacing: 1px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  #header .live {
    display: inline-flex; align-items: center; gap: 6px; font-size: 11px;
    color: var(--muted); padding: 3px 9px; border: 1px solid var(--border);
    border-radius: 12px; background: var(--bg2);
  }
  #header .live .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--muted);
    transition: background 0.2s;
  }
  #header.connected .live .dot { background: var(--good); box-shadow: 0 0 6px var(--good); }
  #search {
    background: var(--bg2); border: 1px solid var(--border); color: var(--text);
    padding: 7px 12px; border-radius: 6px; outline: none; width: 280px; font-size: 13px;
    transition: border-color 0.15s;
  }
  #search:focus { border-color: var(--accent); }
  button.toolbtn {
    background: var(--bg2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  button.toolbtn:hover { border-color: var(--accent); background: rgba(90,140,255,0.08); }
  #stats { color: var(--muted); font-size: 12px; margin-left: auto; }
  #side {
    position: fixed; top: 56px; right: 0; bottom: 0; width: 380px;
    background: rgba(13, 19, 34, 0.96); backdrop-filter: blur(12px);
    border-left: 1px solid var(--border); padding: 24px; overflow-y: auto;
    transform: translateX(100%); transition: transform 0.25s ease; z-index: 5;
  }
  #side.open { transform: translateX(0); }
  #side .close {
    float: right; cursor: pointer; color: var(--muted); font-size: 18px;
    line-height: 1; user-select: none;
  }
  #side .close:hover { color: var(--text); }
  #side h2 { margin: 0 0 4px; font-size: 19px; font-weight: 600; }
  #side .slug { color: var(--muted); font-size: 12px; font-family: ui-monospace, monospace; }
  #side .kind { color: var(--accent2); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }
  #side .tags { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 6px; }
  #side .tag {
    background: rgba(90, 140, 255, 0.12); color: var(--accent);
    padding: 3px 9px; border-radius: 12px; font-size: 11px;
    border: 1px solid rgba(90, 140, 255, 0.25);
  }
  #side .links { margin: 16px 0 8px; font-size: 12px; color: var(--muted); }
  #side .links a {
    color: var(--accent); text-decoration: none; cursor: pointer; margin-right: 10px;
  }
  #side .links a:hover { text-decoration: underline; }
  #side pre {
    white-space: pre-wrap; color: #b6c2d4; font-size: 13px; line-height: 1.55;
    font-family: inherit; margin: 12px 0 0;
  }
  #brain-panel {
    position: fixed; bottom: 14px; right: 14px; width: 280px;
    background: rgba(13, 19, 34, 0.92); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; z-index: 6; font-size: 12px;
  }
  #brain-panel h3 {
    margin: 0 0 8px; font-size: 11px; font-weight: 600;
    letter-spacing: 1.5px; color: var(--muted); text-transform: uppercase;
  }
  #loss-spark { display: block; width: 100%; height: 60px; }
  #loss-meta { color: var(--muted); display: flex; justify-content: space-between; margin-top: 4px; }
  #loss-meta span.val { color: var(--text); font-family: ui-monospace, monospace; }
  #proposals { margin-top: 12px; }
  #proposals .row {
    font-size: 11px; padding: 4px 0; border-top: 1px solid var(--border);
    display: flex; justify-content: space-between;
  }
  #proposals .row:first-child { border-top: 0; }
  #proposals .pair { font-family: ui-monospace, monospace; color: var(--text); }
  #proposals .prob { color: var(--accent2); }
  .legend {
    position: fixed; bottom: 14px; left: 14px; font-size: 11px; color: var(--muted);
    background: rgba(7, 9, 15, 0.75); padding: 8px 12px; border-radius: 8px; z-index: 5;
    border: 1px solid var(--border); backdrop-filter: blur(8px);
  }
  .legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  .legend .dash { display: inline-block; width: 16px; height: 0; border-top: 1.5px dashed var(--accent2); margin-right: 5px; vertical-align: middle; }
  #empty {
    position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
    text-align: center; color: var(--muted); pointer-events: none; z-index: 1;
  }
  #empty.hidden { display: none; }
  #empty h3 { font-weight: 400; font-size: 16px; margin: 0 0 6px; }
  #empty code { background: var(--bg2); padding: 2px 8px; border-radius: 4px; color: var(--text); font-size: 12px; }
</style>
</head>
<body>
<div id="header">
  <h1>JARVIS · BRAIN</h1>
  <span class="live"><span class="dot"></span><span id="live-status">connecting...</span></span>
  <input id="search" placeholder="Search notes, tags, content..." autocomplete="off"/>
  <button class="toolbtn" onclick="trainNow(1)">Train ×1</button>
  <button class="toolbtn" onclick="trainNow(5)">Train ×5</button>
  <div id="stats"></div>
</div>
<div id="graph"></div>
<div id="empty">
  <div>
    <h3>Vault is empty</h3>
    <div>Try <code>brain create my-first-note --title "Hello"</code></div>
  </div>
</div>
<div id="side">
  <span class="close" onclick="closeSide()">&times;</span>
  <h2 id="s-title"></h2>
  <div class="slug" id="s-slug"></div>
  <div class="kind" id="s-kind"></div>
  <div class="tags" id="s-tags"></div>
  <div class="links" id="s-links"></div>
  <pre id="s-body"></pre>
</div>
<div id="brain-panel">
  <h3>Brain training</h3>
  <canvas id="loss-spark" width="280" height="60"></canvas>
  <div id="loss-meta">
    <span>recon <span class="val" id="loss-val">—</span></span>
    <span>steps <span class="val" id="loss-n">0</span></span>
  </div>
  <h3 style="margin-top:14px">Proposed links</h3>
  <div id="proposals"><div style="color:var(--muted);font-size:11px">none yet</div></div>
</div>
<div class="legend">
  <span class="dot" style="background:#5a8cff"></span>note
  <span class="dot" style="background:#3a4560; opacity:0.7; margin-left:12px"></span>ghost
  <span class="dash" style="margin-left:12px"></span>NN proposal
</div>
<script>
const SCALE = 360;  // Map xy in [-1, 1] to pixel coords.
let network, allNodes, allEdges, dataByid = {};
let lossHistory = [];
const MAX_LOSS_POINTS = 120;

function nodeStyle(n) {
  const isProposal = n.kind === 'ghost';
  return {
    id: n.id,
    label: n.title || n.id,
    title: n.preview ? (n.title + '\n\n' + n.preview) : (n.title || n.id),
    shape: 'dot',
    size: 12 + Math.max(0, Math.min(8, (n.importance || 0.5) * 12)),
    borderWidth: 2,
    color: n.ghost
      ? { background: '#252d44', border: '#3a4560' }
      : kindColor(n.kind),
    font: { color: n.ghost ? '#6a7590' : '#e0e6f0', size: 13, face: 'system-ui', strokeWidth: 0 },
    shadow: n.ghost ? false : { enabled: true, color: 'rgba(90,140,255,0.40)', size: 12, x: 0, y: 0 },
    physics: false,
    x: n.xy ? n.xy[0] * SCALE : (Math.random() - 0.5) * SCALE,
    y: n.xy ? n.xy[1] * SCALE : (Math.random() - 0.5) * SCALE,
  };
}

function kindColor(kind) {
  if (kind === 'unsorted') return { background: '#9c8cff', border: '#c8baff' };
  if (kind === 'note') return { background: '#5a8cff', border: '#a0b8ff' };
  if (kind === 'fact') return { background: '#5affb4', border: '#a0ffd4' };
  if (kind === 'skill') return { background: '#ffba5a', border: '#ffd99e' };
  if (kind === 'reflection') return { background: '#ff7a9e', border: '#ffb1c4' };
  return { background: '#5a8cff', border: '#a0b8ff' };
}

function edgeStyle(e) {
  const isProp = e.kind === 'proposal';
  return {
    id: e.id || (e.from + '->' + e.to),
    from: e.from, to: e.to,
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    dashes: isProp ? [4, 4] : false,
    color: isProp
      ? { color: '#b48aff', highlight: '#d6c2ff', opacity: 0.55 }
      : { color: '#2a3550', highlight: '#7da0ff', opacity: 0.7 },
    width: isProp ? 1.2 : 1,
    smooth: { type: 'continuous', roundness: 0.4 },
  };
}

function bootstrapGraph(snapshot) {
  document.getElementById('empty').classList.toggle('hidden', snapshot.nodes.length > 0);
  dataByid = {};
  for (const n of snapshot.nodes) dataByid[n.id] = n;
  const nodes = snapshot.nodes.map(nodeStyle);
  const edges = snapshot.edges.map(edgeStyle);

  if (network) {
    allNodes.clear(); allNodes.add(nodes);
    allEdges.clear(); allEdges.add(edges);
  } else {
    allNodes = new vis.DataSet(nodes);
    allEdges = new vis.DataSet(edges);
    network = new vis.Network(
      document.getElementById('graph'),
      { nodes: allNodes, edges: allEdges },
      {
        physics: { enabled: false },
        interaction: { hover: true, tooltipDelay: 180, zoomSpeed: 0.7 },
      }
    );
    network.on('click', (params) => {
      if (params.nodes.length === 0) { closeSide(); return; }
      show(params.nodes[0]);
    });
    network.on('doubleClick', (params) => {
      if (params.nodes.length === 0) return;
      network.focus(params.nodes[0], { scale: 1.5, animation: { duration: 400 } });
    });
  }
  updateStats();
}

function updateStats() {
  if (!allNodes) return;
  const n = allNodes.length;
  const e = allEdges.length;
  document.getElementById('stats').textContent = n + ' notes · ' + e + ' links';
}

function applyEvent(ev) {
  if (ev.type === 'graph.snapshot') {
    bootstrapGraph(ev.payload);
    return;
  }
  if (ev.type === 'node.created' || ev.type === 'node.updated') {
    dataByid[ev.payload.id] = ev.payload;
    const styled = nodeStyle(ev.payload);
    if (allNodes.get(ev.payload.id)) allNodes.update(styled);
    else allNodes.add(styled);
    document.getElementById('empty').classList.add('hidden');
    updateStats();
    return;
  }
  if (ev.type === 'node.deleted') {
    delete dataByid[ev.payload.id];
    if (allNodes.get(ev.payload.id)) allNodes.remove(ev.payload.id);
    updateStats();
    return;
  }
  if (ev.type === 'training.epoch') {
    for (const s of ev.payload.summaries || []) {
      lossHistory.push(s.avg_recon);
      if (lossHistory.length > MAX_LOSS_POINTS) lossHistory.shift();
    }
    drawSpark();
    if (ev.payload.summaries && ev.payload.summaries.length) {
      const last = ev.payload.summaries[ev.payload.summaries.length - 1];
      document.getElementById('loss-val').textContent = last.avg_recon.toFixed(4);
      document.getElementById('loss-n').textContent = lossHistory.length;
    }
    // Animate xy.
    for (const m of ev.payload.positions || []) {
      const cur = allNodes.get(m.id);
      if (!cur) continue;
      animateMove(m.id, m.xy[0] * SCALE, m.xy[1] * SCALE);
    }
    return;
  }
  if (ev.type === 'training.step') {
    lossHistory.push(ev.payload.recon);
    if (lossHistory.length > MAX_LOSS_POINTS) lossHistory.shift();
    drawSpark();
    document.getElementById('loss-val').textContent = ev.payload.recon.toFixed(4);
    document.getElementById('loss-n').textContent = lossHistory.length;
    return;
  }
  if (ev.type === 'proposals.updated') {
    renderProposals(ev.payload.proposals || []);
    return;
  }
}

function animateMove(id, x, y) {
  const cur = allNodes.get(id);
  if (!cur) return;
  const x0 = cur.x ?? 0, y0 = cur.y ?? 0;
  const start = performance.now();
  const dur = 700;
  function tick(now) {
    const t = Math.min(1, (now - start) / dur);
    const ease = 1 - Math.pow(1 - t, 3);
    allNodes.update({ id, x: x0 + (x - x0) * ease, y: y0 + (y - y0) * ease });
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function drawSpark() {
  const c = document.getElementById('loss-spark');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (lossHistory.length < 2) return;
  const max = Math.max(...lossHistory);
  const min = Math.min(...lossHistory);
  const span = max - min || 1;
  ctx.strokeStyle = 'rgba(180,138,255,0.85)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (lossHistory.length - 1)) * c.width;
    const y = c.height - ((v - min) / span) * (c.height - 4) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = 'rgba(180,138,255,0.18)';
  ctx.lineTo(c.width, c.height); ctx.lineTo(0, c.height); ctx.closePath();
  ctx.fill();
}

function renderProposals(proposals) {
  const el = document.getElementById('proposals');
  if (!proposals.length) {
    el.innerHTML = '<div style="color:var(--muted);font-size:11px">none yet</div>';
    return;
  }
  el.innerHTML = '';
  for (const p of proposals.slice(0, 8)) {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="pair">' + p.from + ' → ' + p.to + '</span>'
                  + '<span class="prob">' + p.prob.toFixed(2) + '</span>';
    el.appendChild(row);
    // Add dashed edge to the graph so it's visually visible.
    const edgeId = 'prop-' + p.from + '->' + p.to;
    if (!allEdges.get(edgeId)) {
      try {
        allEdges.add(edgeStyle({ id: edgeId, from: p.from, to: p.to, kind: 'proposal' }));
      } catch (e) { /* may fail if endpoint nodes missing */ }
    }
  }
}

function show(id) {
  const n = dataByid[id];
  if (!n) return;
  document.getElementById('s-title').textContent = n.title || id;
  document.getElementById('s-slug').textContent = (n.ghost ? '[ghost] ' : '') + id;
  document.getElementById('s-kind').textContent = n.kind || 'note';
  const tagsEl = document.getElementById('s-tags');
  tagsEl.innerHTML = '';
  for (const t of n.tags || []) {
    const el = document.createElement('span');
    el.className = 'tag'; el.textContent = t;
    tagsEl.appendChild(el);
  }
  const linksEl = document.getElementById('s-links');
  linksEl.innerHTML = '';
  if (!n.ghost) {
    const out = (allEdges.get().filter(e => e.from === id) || []).map(e => e.to);
    const inc = (allEdges.get().filter(e => e.to === id) || []).map(e => e.from);
    if (out.length) {
      linksEl.innerHTML += '<div>→ ' + out.map(s => '<a onclick="show(\'' + s + '\')">' + s + '</a>').join('') + '</div>';
    }
    if (inc.length) {
      linksEl.innerHTML += '<div>← ' + inc.map(s => '<a onclick="show(\'' + s + '\')">' + s + '</a>').join('') + '</div>';
    }
  }
  document.getElementById('s-body').textContent = n.preview || (n.ghost ? '(ghost — note not yet written)' : '(empty)');
  document.getElementById('side').classList.add('open');
}

function closeSide() {
  document.getElementById('side').classList.remove('open');
}

async function trainNow(epochs) {
  await fetch('/train?epochs=' + epochs, { method: 'POST' });
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(proto + '://' + location.host + '/graph/ws');
  ws.onopen = () => {
    document.getElementById('header').classList.add('connected');
    document.getElementById('live-status').textContent = 'live';
  };
  ws.onmessage = (msg) => {
    try { applyEvent(JSON.parse(msg.data)); } catch (e) { console.warn(e); }
  };
  ws.onclose = () => {
    document.getElementById('header').classList.remove('connected');
    document.getElementById('live-status').textContent = 'reconnecting';
    setTimeout(connectWS, 2000);
  };
}

async function bootstrap() {
  // Pull initial proposals (server will also re-push on next training).
  try {
    const r = await fetch('/edge_proposals');
    const data = await r.json();
    renderProposals(data.proposals || []);
  } catch (e) { /* ignore */ }
  try {
    const r = await fetch('/training_history?tail=' + MAX_LOSS_POINTS);
    const data = await r.json();
    lossHistory = (data.history || []).map(h => h.recon);
    drawSpark();
    if (lossHistory.length) {
      document.getElementById('loss-val').textContent = lossHistory[lossHistory.length - 1].toFixed(4);
      document.getElementById('loss-n').textContent = lossHistory.length;
    }
  } catch (e) { /* ignore */ }
  connectWS();
}

document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!network) return;
  if (!q) {
    allNodes.forEach(n => allNodes.update({ id: n.id, hidden: false }));
    return;
  }
  allNodes.forEach(n => {
    const d = dataByid[n.id] || {};
    const match = (d.title || '').toLowerCase().includes(q) ||
                  (d.preview || '').toLowerCase().includes(q) ||
                  (d.tags || []).some(t => t.toLowerCase().includes(q)) ||
                  n.id.toLowerCase().includes(q);
    allNodes.update({ id: n.id, hidden: !match });
  });
});

bootstrap();
</script>
</body>
</html>
"""
