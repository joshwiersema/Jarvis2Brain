"""Inline HTML for the interactive brain visualization at /graph."""

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
  #search {
    background: var(--bg2); border: 1px solid var(--border); color: var(--text);
    padding: 7px 12px; border-radius: 6px; outline: none; width: 320px; font-size: 13px;
    transition: border-color 0.15s;
  }
  #search:focus { border-color: var(--accent); }
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
  .legend {
    position: fixed; bottom: 14px; left: 14px; font-size: 11px; color: var(--muted);
    background: rgba(7, 9, 15, 0.75); padding: 8px 12px; border-radius: 8px; z-index: 5;
    border: 1px solid var(--border); backdrop-filter: blur(8px);
  }
  .legend .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
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
  <input id="search" placeholder="Search notes, tags, content..." autocomplete="off"/>
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
  <div class="tags" id="s-tags"></div>
  <div class="links" id="s-links"></div>
  <pre id="s-body"></pre>
</div>
<div class="legend">
  <span class="dot" style="background:#5a8cff"></span>note
  <span class="dot" style="background:#3a4560; opacity:0.7; margin-left:12px"></span>ghost (broken link)
</div>
<script>
let network, allNodes, allEdges, dataByid = {};

async function load() {
  const r = await fetch('/graph.json');
  const data = await r.json();

  document.getElementById('empty').classList.toggle('hidden', data.nodes.length > 0);
  document.getElementById('stats').textContent =
    data.nodes.length + ' notes · ' + data.edges.length + ' links';

  const nodes = data.nodes.map(n => {
    dataByid[n.id] = n;
    const inDeg = data.edges.filter(e => e.to === n.id).length;
    return {
      id: n.id,
      label: n.title,
      title: n.preview ? (n.title + '\n\n' + n.preview) : n.title,
      shape: 'dot',
      size: 12 + Math.min(inDeg, 8) * 2.5,
      borderWidth: 2,
      color: n.ghost
        ? { background: '#252d44', border: '#3a4560', highlight: { background: '#3a4560', border: '#5a6580' } }
        : { background: '#5a8cff', border: '#a0b8ff', highlight: { background: '#7da0ff', border: '#cfdcff' } },
      font: { color: n.ghost ? '#6a7590' : '#e0e6f0', size: 13, face: 'system-ui', strokeWidth: 0 },
      shadow: n.ghost ? false : { enabled: true, color: 'rgba(90,140,255,0.45)', size: 14, x: 0, y: 0 },
    };
  });
  const edges = data.edges.map(e => ({
    from: e.from, to: e.to,
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    color: { color: '#2a3550', highlight: '#7da0ff', opacity: 0.7 },
    width: 1,
    smooth: { type: 'continuous', roundness: 0.4 },
  }));

  allNodes = new vis.DataSet(nodes);
  allEdges = new vis.DataSet(edges);

  network = new vis.Network(
    document.getElementById('graph'),
    { nodes: allNodes, edges: allEdges },
    {
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -55, centralGravity: 0.01,
          springLength: 110, springConstant: 0.08, damping: 0.6,
        },
        stabilization: { iterations: 250 },
      },
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

function show(id) {
  const n = dataByid[id];
  if (!n) return;
  document.getElementById('s-title').textContent = n.title;
  document.getElementById('s-slug').textContent = (n.ghost ? '[ghost] ' : '') + id;
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
  document.getElementById('s-body').textContent = n.preview || (n.ghost ? '(this note does not exist yet)' : '(empty)');
  document.getElementById('side').classList.add('open');
}

function closeSide() {
  document.getElementById('side').classList.remove('open');
}

document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!network) return;
  if (!q) {
    allNodes.forEach(n => allNodes.update({ id: n.id, hidden: false, opacity: 1 }));
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

load();
</script>
</body>
</html>
"""
