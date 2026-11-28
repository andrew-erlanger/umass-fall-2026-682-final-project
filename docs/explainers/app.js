(function () {
  'use strict';
  const D = window.DATA;
  const METHODS = [
    { id: 'cosine', name: 'Cosine', c: 'var(--m-cos)' },
    { id: 'cross_encoder', name: 'Cross-encoder', c: 'var(--m-ce)' },
    { id: 'gnn', name: 'GNN', c: 'var(--m-gnn)' },
    { id: 'hybrid', name: 'Hybrid', c: 'var(--m-hyb)' },
  ];
  const L = i => String.fromCharCode(65 + i);
  const pct = v => (v * 100).toFixed(1) + '%';

  /* ---------- pure logic (mirrors the Python code) ---------- */
  // eval.ranking: best first, ties keep passage order
  // scores are compared at 4 decimals so exact ties in the model break by passage order in the browser too
  const ranking = s0 => { const s = s0.map(v => Math.round(v * 1e4)); return s.map((v, i) => i).sort((a, b) => s[b] - s[a] || a - b); };
  function topK(ex, scores, k) {
    const order = ranking(scores), gold = ex.passages.map(p => p.gold);
    const nGold = gold.filter(Boolean).length, found = order.slice(0, k).filter(j => gold[j]).length;
    return { order, nGold, found, all: found === nGold, top: new Set(order.slice(0, k)) };
  }
  // graph.knn_edges: each passage picks its k most similar; union of both directions
  function picks(ex, i, k) {
    const n = ex.passages.length;
    return Array.from({ length: n }, (_, j) => j).filter(j => j !== i)
      .sort((a, b) => ex.sim[i][b] - ex.sim[i][a] || a - b).slice(0, Math.min(k, n - 1));
  }
  function knn(ex, k) {
    const set = new Set();
    for (let i = 0; i < ex.passages.length; i++) for (const j of picks(ex, i, k)) set.add(Math.min(i, j) + '-' + Math.max(i, j));
    return [...set].map(s => s.split('-').map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }
  // two GATConv layers, evaluated from the trained weights. h = first layer's linear map of the node features.
  function gat(name, ex, pairs) {
    const M = D.models[name], H = ex.h[name], n = H.length, heads = M.heads, C = M.C;
    const nb = Array.from({ length: n }, (_, i) => [i]);
    for (const [a, b] of pairs) { nb[a].push(b); nb[b].push(a); }
    const lrelu = v => (v > 0 ? v : 0.2 * v);
    const soft = e => { const m = Math.max(...e), x = e.map(v => Math.exp(v - m)), t = x.reduce((a, b) => a + b, 0); return x.map(v => v / t); };
    const s = [], d = [];
    for (let i = 0; i < n; i++) {
      s.push([]); d.push([]);
      for (let h = 0; h < heads; h++) {
        let a = 0, b = 0;
        for (let c = 0; c < C; c++) { const v = H[i][h * C + c]; a += v * M.as1[h][c]; b += v * M.ad1[h][c]; }
        s[i].push(a); d[i].push(b);
      }
    }
    const z = [], att = [];
    for (let i = 0; i < n; i++) {
      const mean = nb[i].map(() => 0); let zi = 0;
      for (let h = 0; h < heads; h++) {
        const a = soft(nb[i].map(j => lrelu(s[j][h] + d[i][h])));
        a.forEach((w, t) => { mean[t] += w / heads; });
        for (let c = 0; c < C; c++) {
          let o = M.b1[h * C + c];
          for (let t = 0; t < a.length; t++) o += a[t] * H[nb[i][t]][h * C + c];
          if (o > 0) zi += o * M.w2[h * C + c];
        }
      }
      z.push(zi); att.push(nb[i].map((j, t) => ({ j, w: mean[t] })));
    }
    const scores = z.map((_, i) => {
      const a = soft(nb[i].map(j => lrelu(z[j] * M.as2 + z[i] * M.ad2)));
      return a.reduce((acc, w, t) => acc + w * z[nb[i][t]], M.b2);
    });
    return { scores, att };
  }
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function sample(arr, m, rand) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, m); }
  function corrupt(pairs, n, mode, rate, seed) {
    const rand = rng(seed), m = Math.round(rate * pairs.length), key = p => p[0] + '-' + p[1];
    if (mode === 'delete') {
      const del = sample(pairs, m, rand), gone = new Set(del.map(key));
      return { pairs: pairs.filter(p => !gone.has(key(p))), del, add: [] };
    }
    const have = new Set(pairs.map(key)), non = [];
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (!have.has(a + '-' + b)) non.push([a, b]);
    const add = sample(non, Math.min(m, non.length), rand);
    return { pairs: pairs.concat(add), del: [], add };
  }
  function connClass(ex, pairs) {
    const gold = ex.passages.map((p, i) => (p.gold ? i : -1)).filter(i => i >= 0), par = {};
    gold.forEach(g => { par[g] = g; });
    const find = a => { while (par[a] !== a) a = par[a]; return a; };
    let gg = 0;
    for (const [a, b] of pairs) if (a in par && b in par) { gg++; par[find(a)] = find(b); }
    const c = new Set(gold.map(find)).size;
    return { gg, cls: c === 1 ? 'connected' : c === gold.length ? 'disconnected' : 'partial' };
  }
  const sameGraph = (e, pairs) => pairs.length === e.k3.length && pairs.every((p, i) => p[0] === e.k3[i][0] && p[1] === e.k3[i][1]);
  const scoresOn = (m, e, pairs) => (sameGraph(e, pairs) ? e.scores[m] : gat(m, e, pairs).scores);
  window.RR = { ranking, knn, gat, corrupt, connClass, topK };
  if (typeof document === 'undefined') return;

  /* ---------- page ---------- */
  const $ = (s, r) => (r || document).querySelector(s);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const state = { ex: 0, k: 5, showGold: true, sel: null, knnK: 3, buildFocus: null, attNode: null, noiseMode: 'delete', noiseRate: 30, noiseSeed: 1 };
  const ex = () => D.examples[state.ex];
  const chip = (e, j, small) => `<span class="chip${e.passages[j].gold ? ' gold' : ''}"${small ? ' style="width:1.3rem;height:1.3rem;font-size:.7rem"' : ''}>${L(j)}</span>`;
  const verdict = t => `<span class="verdict${t.all ? ' yes' : ''}">${t.all ? '✓ all gold in top 5' : `✗ ${t.found} of ${t.nGold} gold in top 5`}</span>`;
  const tipFor = (e, j) => `<b>${L(j)} · ${esc(e.passages[j].title)}</b>${e.passages[j].gold ? ' · gold' : ''}`;

  /* tooltip */
  const tip = $('#tip');
  document.addEventListener('mouseover', ev => {
    const t = ev.target.closest && ev.target.closest('[data-tip]');
    if (!t) { tip.hidden = true; return; }
    tip.innerHTML = t.getAttribute('data-tip'); tip.hidden = false;
  });
  document.addEventListener('mousemove', ev => {
    if (tip.hidden) return;
    const w = tip.offsetWidth, x = Math.min(ev.clientX + 14, window.innerWidth - w - 8);
    tip.style.left = Math.max(8, x) + 'px'; tip.style.top = (ev.clientY + 16) + 'px';
  });

  /* shared graph drawing */
  function drawGraph(el, e, o) {
    const W = 420, Hh = 330, pad = o.mini ? 16 : 26, r = o.mini ? 7.5 : 13;
    const X = i => pad + e.xy[i][0] * (W - 2 * pad), Y = i => pad + e.xy[i][1] * (Hh - 2 * pad);
    const line = (p, cls, w) => `<line class="edge ${cls}" data-a="${p[0]}" data-b="${p[1]}" x1="${X(p[0]).toFixed(1)}" y1="${Y(p[0]).toFixed(1)}" x2="${X(p[1]).toFixed(1)}" y2="${Y(p[1]).toFixed(1)}"${w ? ` style="stroke-width:${w.toFixed(1)}"` : ''}/>`;
    let s = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${esc(o.label || 'Passage graph')}">`;
    for (const p of o.del || []) s += line(p, 'del');
    for (const p of o.pairs) {
      const isAdd = (o.add || []).some(q => q[0] === p[0] && q[1] === p[1]);
      const gg = e.passages[p[0]].gold && e.passages[p[1]].gold;
      let cls = isAdd ? 'add' : gg ? 'gg' : '', w = 0;
      if (o.focus != null) {
        const other = p[0] === o.focus ? p[1] : p[1] === o.focus ? p[0] : -1;
        if (other < 0) cls += ' dim'; else { cls += ' hot'; if (o.weights) w = 1 + 10 * (o.weights[other] || 0); }
      }
      s += line(p, cls, w);
    }
    e.passages.forEach((p, i) => {
      const cls = 'node' + (p.gold ? ' gold' : '');
      s += `<g class="${cls}" data-i="${i}" data-tip="${esc(tipFor(e, i))}"${o.mini ? '' : ` tabindex="0" role="button" aria-label="Passage ${L(i)}: ${esc(p.title)}${p.gold ? ', gold' : ''}"`}>`;
      if ((o.ring && o.ring.has(i)) || o.focus === i) s += `<circle class="ring" cx="${X(i).toFixed(1)}" cy="${Y(i).toFixed(1)}" r="${r + 3.5}"/>`;
      s += `<circle cx="${X(i).toFixed(1)}" cy="${Y(i).toFixed(1)}" r="${r}"/>`;
      if (!o.mini) s += `<text x="${X(i).toFixed(1)}" y="${(Y(i) + 4).toFixed(1)}">${L(i)}</text>`;
      s += '</g>';
    });
    el.innerHTML = s + '</svg>';
    if (o.onPick) {
      const pick = ev => { const g = ev.target.closest('.node'); if (g) o.onPick(+g.dataset.i); };
      el.onclick = pick;
      el.onkeydown = ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(ev); } };
    }
  }

  /* ---------- the worked example bar ---------- */
  function renderExampleBar() {
    const e = ex(), n = e.passages.length, g = e.passages.filter(p => p.gold).length;
    $('#ex-q').textContent = e.question;
    $('#ex-meta').innerHTML = `Answer: <strong>${esc(e.answer)}</strong> · ${e.hops} hops · ${n} candidates, ${g} gold · gold passages ${e.cls} in the graph`;
  }

  /* ---------- view: the problem ---------- */
  function renderRanking() {
    const e = ex(), s = e.scores.cosine, t = topK(e, s, state.k), order = t.order;
    const lo = Math.min(...s), hi = Math.max(...s), span = hi - lo || 1;
    if (state.sel == null) state.sel = order.slice().reverse().find(j => e.passages[j].gold);
    let h = '';
    order.forEach((j, r) => {
      const p = e.passages[j], gold = state.showGold && p.gold;
      h += `<button type="button" class="row${gold ? ' gold' : ''}${r >= state.k ? ' out' : ''}${state.sel === j ? ' sel' : ''}" data-j="${j}">
        <span class="rk">${r + 1}</span><span class="chip${gold ? ' gold' : ''}">${L(j)}</span><span class="ttl">${esc(p.title)}</span>
        <span class="bar"><i style="width:${(12 + 88 * (s[j] - lo) / span).toFixed(0)}%"></i>${s[j].toFixed(2)}</span></button>`;
      if (r === state.k - 1) h += `<div class="cut">top ${state.k} goes to the reader</div>`;
    });
    $('#rank-list').innerHTML = h;
    $('#rank-readout').innerHTML = state.showGold
      ? `<span>recall@${state.k} <b>${t.found}/${t.nGold}</b></span><span>all_gold@${state.k} <b>${t.all ? 1 : 0}</b></span><span class="verdict${t.all ? ' yes' : ''}">${t.all ? '✓ the reader sees every hop' : '✗ the reader is missing a hop'}</span>`
      : '<span>Gold labels hidden. This is all a reranker sees at test time.</span>';
    const j = state.sel, p = e.passages[j];
    $('#reader').innerHTML = `<h4><span class="chip${state.showGold && p.gold ? ' gold' : ''}">${L(j)}</span>${esc(p.title)}<span class="cap">· rank ${order.indexOf(j) + 1} of ${order.length}</span></h4><p>${esc(p.text)}</p>`;
  }
  function renderChain() {
    const e = ex(), order = ranking(e.scores.cosine);
    let h = '', worst = null;
    e.steps.forEach((st, i) => {
      const rk = st.p == null ? null : order.indexOf(st.p) + 1;
      if (rk && (!worst || rk > worst.rk)) worst = { rk, i, p: st.p };
      h += `<div class="hop">${st.p == null ? '<span class="chip">?</span>' : chip(e, st.p)}
        <div><div class="sq">Step ${i + 1}. ${esc(st.q.includes(" >> ") ? st.q.replace(" >> ", ": ") + "?" : st.q)}</div><div class="an">→ <b>${esc(st.a)}</b></div></div>
        ${rk ? `<div class="where">Answered by passage <b>${L(st.p)}</b> (${esc(e.passages[st.p].title)}) · cosine rank <b>${rk}</b> of ${order.length}${rk > 5 ? ' · outside the top 5' : ''}</div>` : ''}</div>`;
    });
    $('#chain').innerHTML = h;
    $('#chain-cap').textContent = worst && worst.rk > 5
      ? `Cosine ranks the passage for step ${worst.i + 1} at ${worst.rk}. With a cutoff of 5 the reader never sees it.`
      : 'On this question cosine keeps every step inside the top 5.';
  }

  /* ---------- view: four rerankers ---------- */
  function renderBump() {
    const e = ex(), n = e.passages.length, W = 660, y0 = 46, rh = 19, colX = [118, 276, 434, 592];
    const Hh = y0 + n * rh + 40, y = r => y0 + (r + 0.5) * rh;
    const orders = METHODS.map(m => ranking(e.scores[m.id])), rankOf = orders.map(o => { const a = []; o.forEach((j, r) => { a[j] = r; }); return a; });
    let s = `<svg class="bump" viewBox="0 0 ${W} ${Hh}" role="img" aria-label="Rank of every candidate passage under the four rerankers">`;
    METHODS.forEach((m, c) => { s += `<rect x="${colX[c] - 46}" y="10" width="10" height="10" rx="2" fill="${m.c}"/><text x="${colX[c] - 31}" y="19.5" class="t-strong">${m.name}</text>`; });
    for (let r = 0; r < n; r++) s += `<text x="26" y="${y(r) + 4}" text-anchor="end" class="t-mono t-small">${r + 1}</text>`;
    s += `<line x1="38" x2="${W - 8}" y1="${y0 + 5 * rh}" y2="${y0 + 5 * rh}" class="s-ink" stroke-width="2"/><text x="40" y="${y0 + 5 * rh - 5}" class="t-mono t-small">top 5 ↑</text>`;
    const idx = e.passages.map((p, j) => j).sort((a, b) => e.passages[a].gold - e.passages[b].gold);
    for (const j of idx) {
      let d = '';
      for (let c = 0; c < 3; c++) { const x1 = colX[c] + 9, x2 = colX[c + 1] - 9, y1 = y(rankOf[c][j]), y2 = y(rankOf[c + 1][j]); d += `M${x1} ${y1}C${x1 + 55} ${y1} ${x2 - 55} ${y2} ${x2} ${y2}`; }
      s += `<path class="bl${e.passages[j].gold ? ' gold' : ''}" data-p="${j}" d="${d}"/>`;
    }
    for (const j of idx) METHODS.forEach((m, c) => {
      const ranks = METHODS.map((mm, cc) => `${mm.name} ${rankOf[cc][j] + 1}`).join(' · ');
      s += `<g class="bc${e.passages[j].gold ? ' gold' : ''}" data-p="${j}" data-tip="${esc(tipFor(e, j) + '<br>' + ranks)}"><circle cx="${colX[c]}" cy="${y(rankOf[c][j])}" r="8.5"/><text x="${colX[c]}" y="${y(rankOf[c][j]) + 3.5}">${L(j)}</text></g>`;
    });
    const res = METHODS.map(m => topK(e, e.scores[m.id], 5));
    res.forEach((t, c) => { s += `<text x="${colX[c]}" y="${Hh - 12}" text-anchor="middle" class="t-mono${t.all ? ' t-strong' : ''}">${t.all ? '✓ all gold' : `✗ ${t.found} of ${t.nGold} gold`}</text>`; });
    const el = $('#bump'); el.innerHTML = s + '</svg>';
    const svg = $('svg', el);
    svg.onmouseover = ev => { const g = ev.target.closest('[data-p]'); svg.classList.toggle('focus', !!g); svg.querySelectorAll('.on').forEach(x => x.classList.remove('on')); if (g) svg.querySelectorAll(`[data-p="${g.dataset.p}"]`).forEach(x => x.classList.add('on')); };
    svg.onmouseleave = () => { svg.classList.remove('focus'); };
    const join = a => (a.length > 1 ? a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1] : a[0]);
    const win = METHODS.filter((m, c) => res[c].all).map(m => m.name), lose = METHODS.filter((m, c) => !res[c].all).map(m => m.name);
    $('#bump-cap').textContent = (win.length === 4 ? 'All four rerankers get every gold passage into the top 5 here.'
      : win.length === 0 ? 'None of the four gets every gold passage into the top 5 here.'
      : `On this question ${join(win)} ${win.length > 1 ? 'get' : 'gets'} every gold passage into the top 5. ${join(lose)} ${lose.length > 1 ? 'do' : 'does'} not.`) + ' Hover a passage to trace it.';
  }

  /* ---------- view: graph and GNN ---------- */
  function renderBuild() {
    const e = ex(), k = state.knnK, pairs = knn(e, k), t = topK(e, scoresOn('gnn', e, pairs), 5), cc = connClass(e, pairs);
    const f = state.buildFocus, mine = f == null ? null : new Set(picks(e, f, k));
    drawGraph($('#g-build'), e, { pairs, ring: t.top, focus: f, label: `Similarity graph with k = ${k}`, onPick: i => { state.buildFocus = state.buildFocus === i ? null : i; renderBuild(); } });
    let h = `<h4>GNN top 5 on this graph ${verdict(t)}</h4>`;
    t.order.slice(0, 5).forEach(j => { h += `<div class="it">${chip(e, j, true)}<span>${esc(e.passages[j].title)}</span></div>`; });
    if (f != null) h += `<h4 style="margin-top:.7rem">Passage ${L(f)} picked</h4>` + ([...mine].map(j => `<div class="it">${chip(e, j, true)}<span>${esc(e.passages[j].title)} <span class="cap">· ${e.sim[f][j].toFixed(2)}</span></span></div>`).join('') || '<div class="cap">nothing at k = 0</div>');
    else h += '<div class="cap" style="margin-top:.5rem">Select a passage to list the neighbours it picked.</div>';
    $('#g-build-list').innerHTML = h;
    $('#g-build-readout').innerHTML = `<span>edges <b>${pairs.length}</b></span><span>edges joining two gold passages <b>${cc.gg}</b></span><span>gold passages are <b>${k === 0 ? 'isolated' : cc.cls}</b></span>`;
  }
  function renderAtt() {
    const e = ex(), pairs = e.k3, out = gat('gnn', e, pairs);
    if (state.attNode == null) { const o = ranking(e.scores.cosine); state.attNode = o.slice().reverse().find(j => e.passages[j].gold); }
    const i = state.attNode, w = {}; out.att[i].forEach(a => { w[a.j] = a.w; });
    drawGraph($('#g-att'), e, { pairs, focus: i, weights: w, label: `Attention paid by passage ${L(i)}`, onPick: j => { state.attNode = j; renderAtt(); } });
    const rows = out.att[i].slice().sort((a, b) => b.w - a.w);
    let h = `<h4>Passage ${L(i)} listens to</h4>`;
    rows.forEach(a => {
      h += `<div class="it">${chip(e, a.j, true)}<span>${a.j === i ? '<em>itself</em>' : esc(e.passages[a.j].title)}</span></div>
        <div class="row" style="grid-template-columns:minmax(0,1fr);cursor:default;padding:0 0 .25rem 1.75rem"><span class="bar"><i style="width:${(a.w * 100).toFixed(0)}%;background:var(--m-gnn)"></i>${(a.w * 100).toFixed(0)}%</span></div>`;
    });
    $('#g-att-list').innerHTML = h + '<div class="cap">Select another passage in the graph.</div>';
  }

  /* ---------- view: experiments ---------- */
  function renderMetrics() {
    const rows = [['random', 'Random', 'var(--ink3)'], ['cosine', 'Cosine', 'var(--m-cos)'], ['ce_pretrained', 'Cross-encoder, before fine-tuning', 'var(--ink3)'], ['cross_encoder', 'Cross-encoder', 'var(--m-ce)'], ['gnn_k0', 'GNN with no edges', 'var(--ink3)'], ['gnn', 'GNN', 'var(--m-gnn)'], ['hybrid', 'Hybrid', 'var(--m-hyb)']];
    let h = '<thead><tr><th>Reranker</th><th class="num">recall@5</th><th class="num">recall@10</th><th>all_gold@5</th></tr></thead><tbody>';
    for (const [id, name, c] of rows) {
      const m = D.metrics[id]; if (!m) continue;
      h += `<tr><td>${c.includes('ink3') ? name : `<strong>${name}</strong>`}</td><td class="num">${pct(m['recall@5'])}</td><td class="num">${pct(m['recall@10'])}</td>
        <td><span class="cellbar" style="width:${(m['all_gold@5'] * 9).toFixed(2)}rem;background:${c}"></span><span class="mono">${pct(m['all_gold@5'])}</span></td></tr>`;
    }
    $('#metrics-table').innerHTML = h + '</tbody>';
    $('#metrics-cap').textContent = `${D.run.n_dev} MuSiQue dev questions, models trained on ${D.run.n_train.toLocaleString()} questions. With this few questions, differences of a few points are noise.`;
  }
  function renderConn() {
    const names = { connected: 'Connected', partial: 'Partial', disconnected: 'Disconnected' }, total = D.conn.reduce((a, c) => a + c.n, 0);
    let h = '';
    D.conn.forEach((c, gi) => {
      const e = D.examples.find(x => x.cls === c.cls);
      h += `<div class="mini"><h4>${names[c.cls]} <span class="cap">${c.n} questions · ${(100 * c.n / total).toFixed(0)}%</span></h4><div id="conn-g${gi}"></div>${e ? `<div class="cap">${esc(e.question)}</div>` : ''}</div>`;
    });
    $('#conn-examples').innerHTML = h;
    D.conn.forEach((c, gi) => { const e = D.examples.find(x => x.cls === c.cls); if (e) drawGraph($('#conn-g' + gi), e, { pairs: e.k3, mini: true, label: `${names[c.cls]} gold passages` }); });
    const W = 620, Hh = 220, x0 = 44, y0 = 16, ph = 150, gw = (W - x0 - 10) / 3, bw = 26, gap = 2;
    let s = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="all_gold@5 by connectivity group and reranker">`;
    for (let t = 0; t <= 4; t++) { const yy = y0 + ph - ph * t / 4; s += `<line x1="${x0}" x2="${W - 10}" y1="${yy}" y2="${yy}" class="${t ? 'grid' : 's-ink3'}"/><text x="${x0 - 6}" y="${yy + 4}" text-anchor="end" class="t-mono t-small">${t * 25}%</text>`; }
    D.conn.forEach((c, gi) => {
      const gx = x0 + gi * gw + (gw - 4 * bw - 3 * gap) / 2;
      METHODS.forEach((m, mi) => {
        const v = c[m.id]; if (v == null) return;
        const bh = Math.max(1, ph * v), bx = gx + mi * (bw + gap), by = y0 + ph - bh;
        s += `<path d="M${bx} ${y0 + ph}V${by + 4}q0-4 4-4h${bw - 8}q4 0 4 4V${y0 + ph}z" fill="${m.c}" data-tip="<b>${m.name}</b> · ${names[c.cls].toLowerCase()}<br>all_gold@5 ${pct(v)} · n = ${c.n}"/><text x="${bx + bw / 2}" y="${by - 4}" text-anchor="middle" class="t-mono t-small">${(v * 100).toFixed(0)}</text>`;
      });
      s += `<text x="${x0 + gi * gw + gw / 2}" y="${y0 + ph + 18}" text-anchor="middle" class="t-strong">${names[c.cls]}</text><text x="${x0 + gi * gw + gw / 2}" y="${y0 + ph + 34}" text-anchor="middle" class="t-small">n = ${c.n}</text>`;
    });
    $('#conn-chart').innerHTML = `<div class="scroll-x">${s}</svg></div>`;
    $('#conn-legend').innerHTML = METHODS.map(m => `<span><span class="dot" style="background:${m.c}"></span>${m.name}</span>`).join('');
    $('#conn-cap').textContent = `all_gold@5 on the ${total} dev questions of the small run, grouped by how the gold passages sit in the k = ${D.run.k} graph. The small graphs are one real question from each group, gold passages in amber.`;
  }
  function renderNoise() {
    const e = ex(), n = e.passages.length, rate = state.noiseRate / 100;
    const c = corrupt(e.k3, n, state.noiseMode, rate, state.noiseSeed * 7919 + state.ex);
    drawGraph($('#g-noise'), e, { pairs: c.pairs, del: c.del, add: c.add, label: `Graph with ${state.noiseRate}% of edges ${state.noiseMode === 'delete' ? 'deleted' : 'inserted'}` });
    const chips = t => t.order.slice(0, 5).map(j => chip(e, j, true)).join(' ');
    let h = '';
    for (const m of [METHODS[2], METHODS[3]]) {
      const clean = topK(e, e.scores[m.id], 5), noisy = topK(e, scoresOn(m.id, e, c.pairs), 5);
      h += `<div class="mini"><h4><span><span class="dot" style="background:${m.c}"></span> ${m.name}</span>${verdict(noisy)}</h4>
        <div class="it"><span class="cap" style="width:3.6rem">clean</span><span>${chips(clean)}</span></div>
        <div class="it"><span class="cap" style="width:3.6rem">noisy</span><span>${chips(noisy)}</span></div></div>`;
    }
    const ce = topK(e, e.scores.cross_encoder, 5);
    h += `<div class="mini"><h4><span><span class="dot" style="background:${METHODS[1].c}"></span> Cross-encoder</span>${verdict(ce)}</h4>
      <div class="it"><span class="cap" style="width:3.6rem">no graph</span><span>${chips(ce)}</span></div></div>
      <div class="cap">${c.del.length ? `${c.del.length} of ${e.k3.length} edges deleted.` : c.add.length ? `${c.add.length} random edges inserted next to the original ${e.k3.length}.` : 'The clean graph.'} Each row is that reranker's top 5.</div>`;
    $('#g-noise-lists').innerHTML = h;
  }
  function renderNoiseChart() {
    const N = D.noise, W = 300, Hh = 210, x0 = 40, y0 = 14, pw = 196, ph = 150;
    const all = [].concat(N.delete.gnn, N.delete.hybrid, N.insert.gnn, N.insert.hybrid, [N.flat.cosine, N.flat.cross_encoder]);
    const lo = Math.max(0, Math.floor((Math.min(...all) - 0.03) * 10) / 10), hi = Math.min(1, Math.ceil((Math.max(...all) + 0.03) * 10) / 10);
    const x = i => x0 + pw * i / (N.rates.length - 1), y = v => y0 + ph - ph * (v - lo) / (hi - lo);
    const panel = (mode, title) => {
      let s = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="all_gold@5 as edges are ${title}"><text x="${x0}" y="10" class="t-strong">Edges ${title}</text>`;
      const steps = Math.round((hi - lo) * 10);
      for (let t = 0; t <= steps; t++) { const v = lo + t / 10, yy = y(v); s += `<line x1="${x0}" x2="${x0 + pw}" y1="${yy}" y2="${yy}" class="${t ? 'grid' : 's-ink3'}"/><text x="${x0 - 6}" y="${yy + 4}" text-anchor="end" class="t-mono t-small">${Math.round(v * 100)}%</text>`; }
      N.rates.forEach((r, i) => { s += `<text x="${x(i)}" y="${y0 + ph + 16}" text-anchor="middle" class="t-mono t-small">${Math.round(r * 100)}%</text>`; });
      s += `<text x="${x0 + pw / 2}" y="${y0 + ph + 34}" text-anchor="middle" class="t-small">share of edges ${title}</text>`;
      const ends = [];
      for (const m of METHODS) {
        const flat = N.flat[m.id] != null, vals = flat ? N.rates.map(() => N.flat[m.id]) : N[mode][m.id];
        s += `<polyline fill="none" stroke="${m.c}" stroke-width="2"${flat ? ' stroke-dasharray="5 4"' : ''} points="${vals.map((v, i) => x(i) + ',' + y(v).toFixed(1)).join(' ')}"/>`;
        if (!flat) vals.forEach((v, i) => { s += `<circle cx="${x(i)}" cy="${y(v).toFixed(1)}" r="4" fill="${m.c}" stroke="var(--surface)" stroke-width="2" data-tip="<b>${m.name}</b> · ${Math.round(N.rates[i] * 100)}% ${title}<br>all_gold@5 ${pct(v)}"/><circle cx="${x(i)}" cy="${y(v).toFixed(1)}" r="11" fill="transparent" data-tip="<b>${m.name}</b> · ${Math.round(N.rates[i] * 100)}% ${title}<br>all_gold@5 ${pct(v)}"/>`; });
        ends.push({ m, y: y(vals[vals.length - 1]) });
      }
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      for (const en of ends) s += `<text x="${x0 + pw + 8}" y="${(en.y + 4).toFixed(1)}" class="t-small">${en.m.name === 'Cross-encoder' ? 'Cross-enc.' : en.m.name}</text>`;
      return s + '</svg>';
    };
    $('#noise-chart').innerHTML = `<div class="minis" style="grid-template-columns:repeat(auto-fit,minmax(15rem,1fr))">${panel('delete', 'deleted')}${panel('insert', 'inserted')}</div>`;
    $('#noise-legend').innerHTML = METHODS.map(m => `<span><span class="ln" style="border-color:${m.c}${N.flat[m.id] != null ? ';border-top-style:dashed' : ''}"></span>${m.name}${N.flat[m.id] != null ? ' (does not use the graph)' : ''}</span>`).join('');
    $('#noise-cap').textContent = `all_gold@5 on ${D.run.n_dev} dev questions, averaged over 5 random draws per noise level. Models from the small run, not retrained on noisy graphs.`;
  }

  /* ---------- wiring ---------- */
  function renderExample() { renderExampleBar(); renderRanking(); renderChain(); renderBump(); renderBuild(); renderAtt(); renderNoise(); }
  const sel = $('#ex-select');
  sel.innerHTML = D.examples.map((e, i) => `<option value="${i}">${e.hops}-hop · ${esc(e.question.length > 58 ? e.question.slice(0, 56) + '…' : e.question)}</option>`).join('');
  sel.onchange = () => { state.ex = +sel.value; state.sel = state.attNode = state.buildFocus = null; renderExample(); };
  $('#k-slider').oninput = ev => { state.k = +ev.target.value; $('#k-out').textContent = state.k; renderRanking(); };
  $('#gold-toggle').onchange = ev => { state.showGold = ev.target.checked; renderRanking(); };
  $('#rank-list').onclick = ev => { const r = ev.target.closest('.row'); if (r) { state.sel = +r.dataset.j; renderRanking(); } };
  $('#knn-k').oninput = ev => { state.knnK = +ev.target.value; $('#knn-out').textContent = state.knnK; renderBuild(); };
  const setMode = m => { state.noiseMode = m; $('#noise-del').setAttribute('aria-pressed', m === 'delete'); $('#noise-add').setAttribute('aria-pressed', m === 'insert'); renderNoise(); };
  $('#noise-del').onclick = () => setMode('delete'); $('#noise-add').onclick = () => setMode('insert');
  $('#noise-rate').oninput = ev => { state.noiseRate = +ev.target.value; $('#noise-out').textContent = state.noiseRate + '%'; renderNoise(); };
  $('#noise-resample').onclick = () => { state.noiseSeed++; renderNoise(); };
  document.querySelectorAll('[data-caveat]').forEach(el => {
    el.innerHTML = `<strong>About the numbers and model outputs here.</strong> They come from a small local run of our own code on ${D.run.dataset}: ${D.run.n_train.toLocaleString()} training questions, ${D.run.n_val} validation, ${D.run.n_dev} dev. Cross-encoder ${D.run.ce_epochs} epochs, GNN and hybrid ${D.run.gnn_epochs} epochs. They show how the method behaves. They are not the project's results, which need the full-data runs.`;
  });

  const VIEWS = ['problem', 'rerankers', 'graph', 'experiments'];
  function route() {
    const h = location.hash.slice(1), v = VIEWS.includes(h) ? h : 'problem';
    VIEWS.forEach(n => { $('#v-' + n).hidden = n !== v; });
    document.querySelectorAll('nav a').forEach(a => { if (a.getAttribute('href') === '#' + v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', route);
  renderExample(); renderMetrics(); renderConn(); renderNoiseChart(); route();
})();
