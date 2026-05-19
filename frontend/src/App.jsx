import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = window.location.port === '5173' ? 'http://localhost:8080' : '';
const DIMS = 16;
const COL = { cs: '#00d9ff', math: '#b388ff', food: '#ffb74d', sports: '#69f0ae', doc: '#a6e3a1', default: '#90a4ae' };
const DIM_COL = ['#00d9ff','#00d9ff','#00d9ff','#00d9ff','#b388ff','#b388ff','#b388ff','#b388ff','#ffb74d','#ffb74d','#ffb74d','#ffb74d','#69f0ae','#69f0ae','#69f0ae','#69f0ae'];
const KW = {
  cs: ['list','tree','graph','hash','code','algorithm','program','binary','search','dynamic','pointer','queue','stack','sort','dfs','bfs'],
  math: ['math','calculus','matrix','algebra','probability','prime','number','theorem','integral','derivative','bayes','combinator'],
  food: ['pizza','sushi','ramen','taco','food','recipe','rice','fish','dough','tomato','pork','croissant','butter'],
  sports: ['basketball','football','tennis','chess','swim','sport','game','racket','olympic','touchdown','serve','strategy']
};

function textToEmbedding(text) {
  const words = text.toLowerCase().split(/\s+/);
  const scores = { cs: 0, math: 0, food: 0, sports: 0 };
  for (const word of words) {
    for (const [cat, keywords] of Object.entries(KW)) {
      for (const kw of keywords) {
        if (word.includes(kw) || kw.startsWith(word)) {
          scores[cat] += 0.35;
          break;
        }
      }
    }
  }
  const mx = Math.max(...Object.values(scores), 0.01);
  const norm = value => Math.min(value / mx * 0.88, 0.94);
  const jitter = () => (Math.random() - 0.5) * 0.04;
  const emb = Array(DIMS).fill(0.08);
  const fill = (i, score) => {
    if (score < 0.01) return;
    const b = norm(score);
    emb[i] = Math.max(0.05, b + jitter());
    emb[i + 1] = Math.max(0.05, b + jitter());
    emb[i + 2] = Math.max(0.05, b * 0.92 + jitter());
    emb[i + 3] = Math.max(0.05, b * 0.87 + jitter());
  };
  fill(0, scores.cs);
  fill(4, scores.math);
  fill(8, scores.food);
  fill(12, scores.sports);
  return emb;
}

function pca2D(vectors) {
  if (!vectors.length) return [];
  const n = vectors.length;
  const dims = vectors[0].length;
  const mean = Array(dims).fill(0);
  vectors.forEach(v => v.forEach((x, i) => mean[i] += x / n));
  const centered = vectors.map(v => v.map((x, i) => x - mean[i]));
  const component = (avoid = null) => {
    let w = Array.from({ length: dims }, (_, i) => Math.sin(i + 1));
    for (let iter = 0; iter < 50; iter++) {
      const nw = Array(dims).fill(0);
      centered.forEach(v => {
        const dot = v.reduce((s, x, i) => s + x * w[i], 0);
        v.forEach((x, i) => nw[i] += dot * x);
      });
      if (avoid) {
        const proj = nw.reduce((s, x, i) => s + x * avoid[i], 0);
        avoid.forEach((x, i) => nw[i] -= proj * x);
      }
      const norm = Math.sqrt(nw.reduce((s, x) => s + x * x, 0)) || 1;
      w = nw.map(x => x / norm);
    }
    return w;
  };
  const pc1 = component();
  const pc2 = component(pc1);
  return centered.map(v => [
    v.reduce((s, x, i) => s + x * pc1[i], 0),
    v.reduce((s, x, i) => s + x * pc2[i], 0),
  ]);
}

function App() {
  const [items, setItems] = useState([]);
  const [algo, setAlgo] = useState('hnsw');
  const [metric, setMetric] = useState('cosine');
  const [k, setK] = useState(5);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [latency, setLatency] = useState(null);
  const [bench, setBench] = useState(null);
  const [layers, setLayers] = useState(null);
  const [tab, setTab] = useState('search');
  const [status, setStatus] = useState(null);
  const [docs, setDocs] = useState([]);
  const [docTitle, setDocTitle] = useState('');
  const [docText, setDocText] = useState('');
  const [insertStatus, setInsertStatus] = useState('');
  const [question, setQuestion] = useState('');
  const [ragK, setRagK] = useState(3);
  const [chat, setChat] = useState(null);
  const [typedAnswer, setTypedAnswer] = useState('');
  const [expandedCtx, setExpandedCtx] = useState(new Set());
  const [hitIds, setHitIds] = useState(new Set());
  const [queryPt, setQueryPt] = useState(null);
  const [lastEmbedding, setLastEmbedding] = useState(Array(DIMS).fill(0));
  const [hover, setHover] = useState(null);
  const canvasRef = useRef(null);
  const vecRef = useRef(null);

  const points = useMemo(() => {
    if (items.length < 2) return [];
    const coords = pca2D(items.map(item => item.embedding));
    return items.map((item, i) => ({ x: coords[i][0], y: coords[i][1], item }));
  }, [items]);

  const bounds = useMemo(() => {
    if (!points.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    points.forEach(p => {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    });
    const px = (maxX - minX) * 0.18 || 0.1;
    const py = (maxY - minY) * 0.18 || 0.1;
    return { minX: minX - px, maxX: maxX + px, minY: minY - py, maxY: maxY + py };
  }, [points]);

  async function refresh() {
    const t = Date.now();
    const [itemRes, layerRes, statusRes, docRes] = await Promise.all([
      fetch(`${API}/items?t=${t}`),
      fetch(`${API}/hnsw-info?t=${t}`),
      fetch(`${API}/status?t=${t}`),
      fetch(`${API}/doc/list?t=${t}`)
    ]);
    setItems(await itemRes.json());
    setLayers(await layerRes.json());
    setStatus(await statusRes.json());
    setDocs(await docRes.json());
  }

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let frame;
    let pulse = 0;
    const draw = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const w = canvas.width, h = canvas.height;
      ctx.fillStyle = '#07070f';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#111125';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 46) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 46) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      const w2c = (x, y) => [
        70 + ((x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (w - 140),
        h - 70 - ((y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (h - 140)
      ];
      ctx.fillStyle = '#1a1a38';
      ctx.font = '11px Fira Code, monospace';
      ctx.fillText('PC1 ->', w / 2 - 40, h - 18);
      ctx.save();
      ctx.translate(18, h / 2 + 50);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('PC2 ->', 0, 0);
      ctx.restore();
      ctx.fillStyle = '#151530';
      ctx.font = '12px Fira Code, monospace';
      ctx.fillText('2D PCA Projection - Semantic Space', 80, 28);
      if (queryPt && hitIds.size > 0) {
        const [qx, qy] = w2c(queryPt.x, queryPt.y);
        points.forEach(p => {
          if (!hitIds.has(p.item.id)) return;
          const [px, py] = w2c(p.x, p.y);
          ctx.strokeStyle = 'rgba(108,99,255,0.18)';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(px, py);
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }
      points.forEach(p => {
        const [x, y] = w2c(p.x, p.y);
        const col = COL[p.item.category] || COL.default;
        const hot = hitIds.has(p.item.id);
        const r = hot ? 10 : 7;
        if (hot) {
          ctx.beginPath();
          ctx.arc(x, y, r + 7 + Math.sin(pulse) * 3.5, 0, Math.PI * 2);
          ctx.strokeStyle = `${col}55`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
        gradient.addColorStop(0, `${col}${hot ? 'bb' : '88'}`);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, r * 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        if (hover?.item?.id === p.item.id) {
          ctx.beginPath();
          ctx.arc(x, y, r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      });
      if (queryPt) {
        const [x, y] = w2c(queryPt.x, queryPt.y);
        ctx.shadowColor = '#fff';
        ctx.shadowBlur = 20;
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {
          const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
          const r = i ? 8 : 12;
          ctx.lineTo(x + Math.cos(a) * r, y + Math.sin(a) * r);
        }
        ctx.closePath();
        ctx.fill();
      }
      if (!points.length) {
        ctx.fillStyle = '#4a4a6a';
        ctx.font = '13px Fira Code, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Connecting to Nexus...', w / 2, h / 2);
      }
      pulse += 0.05;
      frame = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [points, bounds, hitIds, queryPt, hover]);

  useEffect(() => {
    if (!chat || chat.loading || chat.error || !chat.answer) {
      setTypedAnswer('');
      return;
    }
    setTypedAnswer('');
    let i = 0;
    const timer = setInterval(() => {
      i += 3;
      setTypedAnswer(chat.answer.slice(0, i));
      if (i >= chat.answer.length) clearInterval(timer);
    }, 18);
    return () => clearInterval(timer);
  }, [chat]);

  useEffect(() => {
    const canvas = vecRef.current;
    if (!canvas) return;
    const width = canvas.parentElement.clientWidth;
    canvas.width = width;
    canvas.height = 76;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#07070f';
    ctx.fillRect(0, 0, width, 76);
    const bw = (width - 4) / DIMS;
    lastEmbedding.forEach((v, i) => {
      const h = v * 58;
      ctx.shadowColor = DIM_COL[i];
      ctx.shadowBlur = 5;
      ctx.fillStyle = `${DIM_COL[i]}aa`;
      ctx.fillRect(2 + i * bw + 1, 63 - h, bw - 2, h);
    });
  }, [lastEmbedding]);

  useEffect(() => {
    if (query.trim()) {
      runSearch();
    }
  }, [algo, metric, k]);

  async function runSearch() {
    if (!query.trim()) return;
    const emb = textToEmbedding(query);
    setLastEmbedding(emb);
    const res = await fetch(`${API}/search?v=${emb.join(',')}&k=${k}&metric=${metric}&algo=${algo}`);
    const data = await res.json();
    const found = data.results || [];
    setResults(found);
    setLatency(data.latencyUs || 0);
    setHitIds(new Set(found.map(r => r.id)));
    let sx = 0, sy = 0, sw = 0;
    found.slice(0, 3).forEach((r, i) => {
      const pt = points.find(p => p.item.id === r.id);
      if (pt) { const weight = 1 / (i + 1); sx += pt.x * weight; sy += pt.y * weight; sw += weight; }
    });
    if (sw > 0) setQueryPt({ x: sx / sw + (Math.random() - 0.5) * 0.015, y: sy / sw + (Math.random() - 0.5) * 0.015 });
  }

  async function runBenchmark() {
    const emb = textToEmbedding(query.trim() || 'binary tree algorithm');
    const res = await fetch(`${API}/benchmark?v=${emb.join(',')}&k=5&metric=${metric}`);
    setBench(await res.json());
  }

  async function addVector(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const metadata = String(form.get('metadata') || '').trim();
    const category = String(form.get('category') || 'cs');
    if (!metadata) return;
    await fetch(`${API}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata, category, embedding: textToEmbedding(`${metadata} ${category}`) })
    });
    event.currentTarget.reset();
    await refresh();
  }

  async function deleteItem(id) {
    await fetch(`${API}/delete/${id}`, { method: 'DELETE' });
    setResults(results.filter(r => r.id !== id));
    await refresh();
  }

  async function insertDocument() {
    if (!docTitle.trim() || !docText.trim()) {
      setInsertStatus('Need both a title and text.');
      return;
    }
    setInsertStatus('Calling Ollama nomic-embed-text...');
    const res = await fetch(`${API}/doc/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: docTitle, text: docText })
    });
    const data = await res.json();
    if (data.error) {
      setInsertStatus(data.error);
      return;
    }
    await fetch(`${API}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata: docTitle, category: 'doc', embedding: textToEmbedding(`${docTitle} ${docText}`) })
    });
    setInsertStatus(`Inserted ${data.chunks} chunk(s) - ${data.dims}D embeddings`);
    setDocTitle('');
    setDocText('');
    await refresh();
  }

  async function deleteDoc(id) {
    await fetch(`${API}/doc/delete/${id}`, { method: 'DELETE' });
    await refresh();
  }

  async function askAI() {
    if (!question.trim()) return;
    const asked = question;
    setQuestion('');
    setExpandedCtx(new Set());
    setChat({ question: asked, loading: true });
    fetch(`${API}/doc/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: asked, k: ragK })
    }).then(r => r.json()).then(data => {
      const ids = new Set();
      let sx = 0, sy = 0, sw = 0;
      (data.contexts || []).forEach((ctx, i) => {
        const pt = points.find(p => p.item.category === 'doc' && ctx.title.startsWith(p.item.metadata));
        if (pt) { ids.add(pt.item.id); const weight = 1 / (i + 1); sx += pt.x * weight; sy += pt.y * weight; sw += weight; }
      });
      setHitIds(ids);
      if (sw > 0) setQueryPt({ x: sx / sw, y: sy / sw });
      if (!data.contexts?.length) {
        const emb16 = textToEmbedding(asked);
        fetch(`${API}/search?v=${emb16.join(',')}&k=3&metric=cosine&algo=hnsw`)
          .then(r => r.json())
          .then(fallback => {
            const fallbackIds = new Set();
            let fx = 0, fy = 0, fw = 0;
            (fallback.results || []).slice(0, 3).forEach((r, i) => {
              const pt = points.find(p => p.item.id === r.id);
              if (pt) {
                fallbackIds.add(pt.item.id);
                const weight = 1 / (i + 1);
                fx += pt.x * weight;
                fy += pt.y * weight;
                fw += weight;
              }
            });
            setHitIds(fallbackIds);
            if (fw > 0) setQueryPt({ x: fx / fw + (Math.random() - 0.5) * 0.015, y: fy / fw + (Math.random() - 0.5) * 0.015 });
          }).catch(() => {});
      }
    }).catch(() => {});
    const res = await fetch(`${API}/doc/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: asked, k: ragK })
    });
    const data = await res.json();
    setChat({ question: asked, loading: false, ...data });
  }

  const maxBench = bench ? Math.max(bench.bruteforceUs, bench.kdtreeUs, bench.hnswUs, 1) : 1;
  const w2c = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    return [
      70 + ((x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (canvas.width - 140),
      canvas.height - 70 - ((y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (canvas.height - 140)
    ];
  };
  const handleCanvasMove = event => {
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    let best = null;
    let bestDist = 18;
    points.forEach(p => {
      const [x, y] = w2c(p.x, p.y);
      const d = Math.hypot(mx - x, my - y);
      if (d < bestDist) {
        bestDist = d;
        best = p.item;
      }
    });
    setHover(best ? { item: best, x: event.clientX + 14, y: event.clientY - 8 } : null);
  };
  const toggleCtx = index => {
    setExpandedCtx(prev => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  return (
    <>
      <header>
        <span className="badge hl">HNSW</span>
        <span className="badge">KD-TREE</span>
        <span className="badge">BRUTE FORCE</span>
        <span className={`badge ${status?.ollamaAvailable ? 'ok' : 'err'}`}>OLLAMA {status?.ollamaAvailable ? 'OK' : 'OFF'}</span>
        <span id="statsLabel">{items.length} vectors - {DIMS} dims</span>
      </header>

      <div className="layout">
        <aside className="left-panel">
          <section>
            <div className="sec">Query (Demo Vectors)</div>
            <div className="stack">
              <input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && runSearch()} placeholder="binary tree, sushi, basketball..." />
              <button className="btn-p" onClick={runSearch}>SEARCH</button>
            </div>
          </section>
          <section>
            <div className="sec">Algorithm</div>
            <div className="algo-row">
              {['hnsw','kdtree','bruteforce'].map(name => <button key={name} className={`algo-btn ${algo === name ? 'on' : ''}`} onClick={() => setAlgo(name)}>{name === 'bruteforce' ? 'BRUTE' : name.toUpperCase()}</button>)}
            </div>
          </section>
          <section>
            <div className="sec">Distance Metric</div>
            <select value={metric} onChange={e => setMetric(e.target.value)}>
              <option value="cosine">Cosine Similarity</option>
              <option value="euclidean">Euclidean Distance</option>
              <option value="manhattan">Manhattan Distance</option>
            </select>
          </section>
          <section>
            <div className="sec">Top-K: {k}</div>
            <input type="range" min="1" max="10" value={k} onChange={e => setK(Number(e.target.value))} />
          </section>
          <section>
            <div className="sec">Category Legend</div>
            {Object.entries({ cs: 'CS / Algorithms', math: 'Mathematics', food: 'Food & Cooking', sports: 'Sports & Games', doc: 'Documents (RAG)' }).map(([key, label]) => (
              <div className="leg-row" key={key}><span className="dot" style={{ background: COL[key], boxShadow: `0 0 5px ${COL[key]}` }} />{label}</div>
            ))}
          </section>
          <form onSubmit={addVector} className="stack">
            <div className="sec">Insert Demo Vector</div>
            <input name="metadata" placeholder="Description..." />
            <select name="category" defaultValue="cs">
              <option value="cs">CS / Algorithms</option>
              <option value="math">Mathematics</option>
              <option value="food">Food & Cooking</option>
              <option value="sports">Sports & Games</option>
            </select>
            <button className="btn-s">+ INSERT</button>
          </form>
          <section>
            <div className="sec">Benchmark</div>
            <button className="btn-s" onClick={runBenchmark}>COMPARE ALL ALGOS</button>
          </section>
        </aside>

        <main className="center-panel">
          <canvas ref={canvasRef} onMouseMove={handleCanvasMove} onMouseLeave={() => setHover(null)} />
          {hover && <div id="tip" style={{ display: 'block', left: hover.x, top: hover.y }}>
            <span style={{ color: COL[hover.item.category] || COL.default }}>[{hover.item.category}]</span><br />{hover.item.metadata}
          </div>}
        </main>

        <aside className="right-panel">
          <nav className="tabs">
            {['search','docs','rag'].map(name => <button key={name} onClick={() => setTab(name)} className={`tab ${tab === name ? 'on' : ''}`}>{name === 'rag' ? 'ASK AI' : name.toUpperCase()}</button>)}
          </nav>

          {tab === 'search' && <div className="tab-content on">
            <section>
              <div className="sec">Search Latency</div>
              <div className="lat-big">{latency == null ? '-' : latency < 1000 ? `${latency} us` : `${(latency / 1000).toFixed(2)} ms`}</div>
              <div className="lat-sub">{latency == null ? 'No query yet' : `${algo.toUpperCase()} - ${metric} - k=${k}`}</div>
            </section>
            <section>
              <div className="sec">Top Matches</div>
              <div className="results">
                {results.length ? results.map((r, i) => <article className="rcard" key={r.id}>
                  <div className="rrank">#{i + 1} NEAREST</div>
                  <div className="rmeta">{r.metadata}</div>
                  <div className="rfoot">
                    <span className="rcat" style={{ color: COL[r.category] || COL.default }}>{r.category.toUpperCase()}</span>
                    <span className="rdist">dist: {r.distance.toFixed(5)}</span>
                    <button className="del" onClick={() => deleteItem(r.id)}>x</button>
                  </div>
                </article>) : <div className="empty">Run a search to see results...</div>}
              </div>
            </section>
            <section>
              <div className="sec">Query Embedding (16D)</div>
              <canvas ref={vecRef} id="vecCvs" />
            </section>
            {bench && <section>
              <div className="sec">Algorithm Comparison</div>
              {[
                ['Brute Force', bench.bruteforceUs, '#f38ba8'],
                ['KD-Tree', bench.kdtreeUs, '#89dceb'],
                ['HNSW', bench.hnswUs, '#b388ff'],
              ].map(([label, us, color]) => <div className="brow" key={label}>
                <div className="blabel"><span style={{ color }}>{label}</span><span>{us < 1000 ? `${us} us` : `${(us / 1000).toFixed(2)} ms`}</span></div>
                <div className="btrack"><div className="bfill" style={{ width: `${Math.max((us / maxBench) * 100, 2)}%`, background: color }} /></div>
              </div>)}
            </section>}
            <section>
              <div className="sec">HNSW Graph Layers</div>
              {layers?.nodesPerLayer?.map((cnt, lyr) => <div className="lrow" key={lyr}>
                <div className="lnum">L{lyr}</div>
                <div className="ltrack"><div className="lfill" style={{ width: `${Math.max((cnt / (layers.nodesPerLayer[0] || 1)) * 100, 2)}%` }} /></div>
                <div className="lcount">{cnt}n - {layers.edgesPerLayer?.[lyr] || 0}e</div>
              </div>)}
            </section>
          </div>}

          {tab === 'docs' && <div className="tab-content on">
            <section>
              <div className="sec">Ollama Status</div>
              <div className={`ollama-status ${status?.ollamaAvailable ? 'ok' : 'err'}`}>
                {status?.ollamaAvailable
                  ? <><span className="online">Online</span><br />Embed: <span>{status.embedModel}</span><br />Generate: <span>{status.genModel}</span><br />Dims: <span>{status.docDims || '(first insert sets this)'}</span><br />Documents: <span>{status.docCount}</span></>
                  : <><span className="offline">Offline</span><br /><br />To enable RAG features:<br /><span className="muted">1. Install from ollama.com<br />2. ollama pull nomic-embed-text<br />3. ollama pull llama3.2</span></>}
              </div>
            </section>
            <section className="stack">
              <div className="sec">Insert Document</div>
              <input value={docTitle} onChange={e => setDocTitle(e.target.value)} placeholder="Document title / topic..." />
              <textarea value={docText} onChange={e => setDocText(e.target.value)} placeholder="Paste your notes, textbook excerpt, lecture content..." />
              <button className="btn-g" onClick={insertDocument}>EMBED & INSERT</button>
              <div className="muted">{insertStatus}</div>
            </section>
            <section>
              <div className="sec">Stored Documents ({docs.length})</div>
              <div className="doc-list">
                {docs.length ? docs.map(doc => <article className="dcard" key={doc.id}>
                  <div className="dcard-title">{doc.title}</div>
                  <div className="dcard-preview">{doc.preview}</div>
                  <div className="dcard-foot"><span>{doc.words} words</span><button className="del" onClick={() => deleteDoc(doc.id)}>x</button></div>
                </article>) : <div className="empty">No documents yet.</div>}
              </div>
            </section>
          </div>}

          {tab === 'rag' && <div className="tab-content on">
            <section className="stack">
              <div className="sec">Ask a Question</div>
              <textarea value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => e.key === 'Enter' && e.ctrlKey && askAI()} rows="3" placeholder={'What is dynamic programming?\nExplain the main idea of HNSW.\nHow does the recipe differ from...'} />
              <div className="ask-row">
                <select value={ragK} onChange={e => setRagK(Number(e.target.value))}>
                  <option value="2">Top 2</option>
                  <option value="3">Top 3</option>
                  <option value="5">Top 5</option>
                </select>
                <button className="btn-g" onClick={askAI} disabled={chat?.loading}>ASK AI</button>
              </div>
              <div className="hint">Uses your inserted documents as context. Answers come from the local LLM.</div>
            </section>
            <section>
              <div className="sec">Conversation</div>
              <div className="chat-history">
                {!chat && <div className="empty">Ask a question about your inserted documents...</div>}
                {chat && <><div className="chat-q">{chat.question}</div>
                  {chat.loading && <div className="thinking"><div className="spinner" />Retrieving context & generating answer...</div>}
                  {!chat.loading && <div className="chat-a">
                    <div className="chat-a-label">{chat.error ? 'ERROR' : chat.model}</div>
                    <div className={`chat-a-text ${typedAnswer !== chat.answer && !chat.error ? 'typing' : ''}`}>{chat.error || typedAnswer}</div>
                    {!!chat.contexts?.length && <div className="chat-ctx">
                      <div className="chat-ctx-label">RETRIEVED CONTEXT ({chat.contexts.length} chunks)</div>
                      {chat.contexts.map((c, i) => <React.Fragment key={c.id}>
                        <button className="ctx-chip" onClick={() => toggleCtx(i)}>#{i + 1} {c.title} - {c.distance.toFixed(3)}</button>
                        <div className="ctx-expand" style={{ display: expandedCtx.has(i) ? 'block' : 'none' }}>{c.text}</div>
                      </React.Fragment>)}
                    </div>}
                  </div>}
                </>}
              </div>
            </section>
          </div>}
        </aside>
      </div>
    </>
  );
}

createRoot(document.getElementById('root')).render(<App />);
