<script>
(function(){
"use strict";
const TAU = Math.PI*2;
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const reduceMotion = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

/* ---------- theme tokens ---------- */
const T = {}, RGB = {};
function readTokens(){
  const cs = getComputedStyle(document.documentElement);
  ['bg','panel','ink','muted','rule','grid','v1','v2','one','acc','ok','warn','bad','v1-soft','v2-soft','acc-soft','ok-soft','bad-soft'].forEach(k => T[k] = cs.getPropertyValue('--'+k).trim());
  ['panel','ink','v1','v2','acc','one','grid','ok','bad','warn'].forEach(k => { const h=T[k].replace('#',''); RGB[k]=[0,2,4].map(i=>parseInt(h.slice(i,i+2),16)); });
}
const F = {
  tick: '11px "IBM Plex Mono", ui-monospace, Menlo, monospace',
  small:'10px "IBM Plex Mono", ui-monospace, Menlo, monospace',
  num:  '600 12px "IBM Plex Mono", ui-monospace, Menlo, monospace',
  lab:  '600 13px "IBM Plex Sans", system-ui, sans-serif',
  big:  '600 15px "IBM Plex Sans", system-ui, sans-serif',
  ttl:  '600 13px "IBM Plex Sans", system-ui, sans-serif',
};

/* ---------- maths ---------- */
function fmt(x, d=2){
  if (!isFinite(x)) return x>0 ? '∞' : x<0 ? '−∞' : '—';
  let s = x.toFixed(d);
  if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
  return s.replace('-', '−');
}
function fmtG(x){ const a=Math.abs(x); if (!isFinite(x)) return fmt(x); if (a>=1e4 || (a<1e-3 && a>0)) return x.toExponential(1).replace('-','−').replace('e+','e'); return fmt(x, a>=100?0:a>=10?1:a>=1?2:3); }
function compact(n){ const u=[[1e12,'T'],[1e9,'B'],[1e6,'M'],[1e3,'k']]; for (const [v,s] of u) if (n>=v) return (n/v).toFixed(n/v>=100?0:1).replace(/\.0$/,'')+s; return String(Math.round(n)); }
function rng(seed){ return function(){ seed|=0; seed=seed+0x6D2B79F5|0; let t=Math.imul(seed^seed>>>15,1|seed); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }
function gauss(r){ let u=0; while(u===0) u=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(TAU*r()); }
function shuffle(a, r){ for (let i=a.length-1;i>0;i--){ const j=Math.floor(r()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } return a; }
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const sigm = x => 1/(1+Math.exp(-x));
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
// solve A x = b (Gaussian elimination, partial pivoting); A is copied
function solve(A, b){
  const n=b.length, M=A.map((r,i)=>[...r, b[i]]);
  for (let c=0;c<n;c++){
    let p=c; for (let r=c+1;r<n;r++) if (Math.abs(M[r][c])>Math.abs(M[p][c])) p=r;
    [M[c],M[p]]=[M[p],M[c]];
    const d=M[c][c]||1e-300;
    for (let r=c+1;r<n;r++){ const f=M[r][c]/d; if (f) for (let k=c;k<=n;k++) M[r][k]-=f*M[c][k]; }
  }
  const x=Array(n).fill(0);
  for (let r=n-1;r>=0;r--){ let s=M[r][n]; for (let k=r+1;k<n;k++) s-=M[r][k]*x[k]; x[r]=s/(M[r][r]||1e-300); }
  return x;
}
// least squares with a tiny ridge: rows of features Phi, targets y
function lstsq(Phi, y, ridge=1e-9){
  const p=Phi[0].length, A=[...Array(p)].map(()=>Array(p).fill(0)), b=Array(p).fill(0);
  Phi.forEach((r,i)=>{ for (let a=0;a<p;a++){ b[a]+=r[a]*y[i]; for (let c=0;c<p;c++) A[a][c]+=r[a]*r[c]; } });
  const tr=A.reduce((s,r,i)=>s+r[i],0)/p;
  for (let a=0;a<p;a++) A[a][a]+=ridge*(tr||1);
  return solve(A,b);
}
// Legendre features on x in [0,1] (well conditioned up to degree 12)
function legendre(x, d){ const z=2*x-1, P=[1]; if (d>=1) P.push(z); for (let n=1;n<d;n++) P.push(((2*n+1)*z*P[n]-n*P[n-1])/(n+1)); return P; }

/* ---------- canvas helpers ---------- */
function prep(cv){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  const W = Math.max(1, Math.round(w*dpr)), H = Math.max(1, Math.round(h*dpr));
  if (cv.width!==W || cv.height!==H){ cv.width=W; cv.height=H; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.fillStyle = T.panel; ctx.fillRect(0,0,w,h);
  return {ctx, w, h};
}
function makeView2(w,h,x0,x1,y0,y1,pad){
  pad = pad || {l:46,r:14,t:30,b:24};
  const sx=(w-pad.l-pad.r)/(x1-x0), sy=(h-pad.t-pad.b)/(y1-y0);
  return { x0,x1,y0,y1,pad,sx,sy, X:x=>pad.l+(x-x0)*sx, Y:y=>h-pad.b-(y-y0)*sy, iX:px=>(px-pad.l)/sx+x0, iY:py=>(h-pad.b-py)/sy+y0 };
}
// equal-aspect view centred on (cx, cy) showing at least ±span
function eqView(w,h,cx,cy,span,pad){
  pad = pad || {l:40,r:14,t:30,b:24};
  const W=w-pad.l-pad.r, H=h-pad.t-pad.b, s=Math.min(W,H)/(2*span);
  return makeView2(w,h,cx-W/2/s,cx+W/2/s,cy-H/2/s,cy+H/2/s,pad);
}
function tickStep(range,k=5){ const raw=range/k, p=Math.pow(10,Math.floor(Math.log10(raw))), f=raw/p; return (f<1.5?1:f<3.5?2:f<7.5?5:10)*p; }
function tickFmt(x, st){ const d=st>=1?0:Math.min(3,Math.ceil(-Math.log10(st)-1e-9)); return fmt(x,d); }
function logFmt(k){ const v=Math.pow(10,k); return v>=1 ? compact(v) : String(+v.toPrecision(1)); }
function drawAxes2(ctx,v,w,h,xlab,ylab,opt={}){
  const sx=opt.xStep||tickStep(v.x1-v.x0), sy=opt.logY?1:tickStep(v.y1-v.y0,4), L=v.pad.l, R=w-v.pad.r, Tp=v.pad.t, B=h-v.pad.b;
  ctx.lineWidth=1; ctx.font=F.tick; ctx.fillStyle=T.muted;
  ctx.textAlign='center'; ctx.textBaseline='top';
  for (let x=Math.ceil(v.x0/sx-1e-9)*sx; x<=v.x1+sx*1e-6; x+=sx){ const px=Math.round(v.X(x))+.5; ctx.strokeStyle=T.grid; ctx.beginPath(); ctx.moveTo(px,Tp); ctx.lineTo(px,B); ctx.stroke(); ctx.fillText(opt.xFmt?opt.xFmt(x):tickFmt(x,sx),px,B+5); }
  ctx.textAlign='right'; ctx.textBaseline='middle';
  for (let y=Math.ceil(v.y0/sy-1e-9)*sy; y<=v.y1+sy*1e-6; y+=sy){ const py=Math.round(v.Y(y))+.5; ctx.strokeStyle=T.grid; ctx.beginPath(); ctx.moveTo(L,py); ctx.lineTo(R,py); ctx.stroke(); ctx.fillText(opt.logY?logFmt(Math.round(y)):tickFmt(y,sy),L-6,py); }
  ctx.strokeStyle=T.rule; ctx.strokeRect(L+.5,Tp+.5,R-L-1,B-Tp-1);
  if (xlab) label(ctx,xlab,R-4,B-10,T.muted,F.ttl,'right');
  if (ylab) label(ctx,ylab,L+6,Tp+10,T.muted,F.ttl,'left');
}
function clipPlot(ctx,v,w,h){ ctx.save(); ctx.beginPath(); ctx.rect(v.pad.l,v.pad.t,w-v.pad.l-v.pad.r,h-v.pad.t-v.pad.b); ctx.clip(); }
function arrow(ctx, x0,y0,x1,y1, col, lw=2, head=10){
  const a=Math.atan2(y1-y0,x1-x0), L=Math.hypot(x1-x0,y1-y0);
  if (L<0.5) return;
  const hh=Math.min(head, L*0.45);
  ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=lw; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1-Math.cos(a)*hh*0.7, y1-Math.sin(a)*hh*0.7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1,y1);
  ctx.lineTo(x1-hh*Math.cos(a-0.38), y1-hh*Math.sin(a-0.38));
  ctx.lineTo(x1-hh*Math.cos(a+0.38), y1-hh*Math.sin(a+0.38));
  ctx.closePath(); ctx.fill();
}
function label(ctx, txt, x, y, col, font=F.lab, align='left', base='middle'){
  ctx.font=font; ctx.textAlign=align; ctx.textBaseline=base;
  ctx.lineJoin='round'; ctx.lineWidth=4; ctx.strokeStyle=T.panel; ctx.strokeText(txt,x,y);
  ctx.fillStyle=col; ctx.fillText(txt,x,y);
}
function title(ctx, txt){ label(ctx, txt, 12, 16, T.muted, F.ttl); }
function line(ctx, pts, col, lw=2, dash){
  ctx.save(); if (dash) ctx.setLineDash(dash); ctx.strokeStyle=col; ctx.lineWidth=lw; ctx.lineJoin='round'; ctx.lineCap='round';
  ctx.beginPath(); let pen=false;
  pts.forEach(p=>{ if (!p || !isFinite(p[0]) || !isFinite(p[1])){ pen=false; return; } pen?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); pen=true; });
  ctx.stroke(); ctx.restore();
}
function dotAt(ctx, x, y, col, r=5, ring){
  ctx.beginPath(); ctx.arc(x,y,r,0,TAU);
  if (ring){ ctx.lineWidth=2; ctx.strokeStyle=col; ctx.fillStyle=T.panel; ctx.fill(); ctx.stroke(); }
  else { ctx.fillStyle=col; ctx.fill(); ctx.lineWidth=1.5; ctx.strokeStyle=T.panel; ctx.stroke(); }
}
function ringAt(ctx, x, y, col, r=9, lw=2){ ctx.beginPath(); ctx.arc(x,y,r,0,TAU); ctx.lineWidth=lw; ctx.strokeStyle=col; ctx.stroke(); }
function cross(ctx, x, y, col, r=6){
  ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(x-r,y-r); ctx.lineTo(x+r,y+r); ctx.moveTo(x+r,y-r); ctx.lineTo(x-r,y+r); ctx.stroke();
}
function star(ctx, x, y, col, r=8){
  ctx.beginPath();
  for (let i=0;i<10;i++){ const a=-Math.PI/2+i*Math.PI/5, rr=i%2?r*0.45:r; ctx.lineTo(x+rr*Math.cos(a), y+rr*Math.sin(a)); }
  ctx.closePath(); ctx.fillStyle=col; ctx.fill(); ctx.lineWidth=1.5; ctx.strokeStyle=T.panel; ctx.stroke();
}
const mix = (a,b,t) => a.map((v,i)=>Math.round(v+(b[i]-v)*t));
const rgba = (c,a=1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
// paint f(x,y) -> [r,g,b] over the plot area at low resolution, then scale up smoothly
const OFF = document.createElement('canvas');
function heat(ctx, v, w, h, f, res=4){
  const L=v.pad.l, Tp=v.pad.t, W=w-L-v.pad.r, H=h-Tp-v.pad.b;
  const nx=Math.max(2,Math.ceil(W/res)), ny=Math.max(2,Math.ceil(H/res));
  OFF.width=nx; OFF.height=ny;
  const oc=OFF.getContext('2d'), img=oc.createImageData(nx,ny), d=img.data;
  for (let j=0;j<ny;j++) for (let i=0;i<nx;i++){
    const c=f(v.iX(L+(i+.5)*W/nx), v.iY(Tp+(j+.5)*H/ny)), k=4*(j*nx+i);
    d[k]=c[0]; d[k+1]=c[1]; d[k+2]=c[2]; d[k+3]=255;
  }
  oc.putImageData(img,0,0);
  ctx.save(); ctx.imageSmoothingEnabled=true; ctx.drawImage(OFF,L,Tp,W,H); ctx.restore();
}
// loss surface on a log scale: shading (darker = lower) plus contour lines by marching squares
function contourMap(ctx, v, w, h, f, col='acc', res=4){
  const L=v.pad.l, Tp=v.pad.t, W=w-L-v.pad.r, H=h-Tp-v.pad.b;
  const nx=Math.ceil(W/res), ny=Math.ceil(H/res), N=nx+1, vals=new Float64Array(N*(ny+1));
  const px=i=>L+i*W/nx, py=j=>Tp+j*H/ny;
  let lo=Infinity, hi=-Infinity;
  for (let j=0;j<=ny;j++) for (let i=0;i<=nx;i++){
    const u=Math.log(Math.max(f(v.iX(px(i)), v.iY(py(j))),1e-9)); vals[j*N+i]=u;
    if (u<lo) lo=u; if (u>hi) hi=u;
  }
  const P=RGB.panel, C=RGB[col], span=(hi-lo)||1;
  OFF.width=N; OFF.height=ny+1;
  const oc=OFF.getContext('2d'), img=oc.createImageData(N,ny+1), d=img.data;
  for (let k=0;k<vals.length;k++){ const c=mix(P,C,0.34*(1-(vals[k]-lo)/span)); d[4*k]=c[0]; d[4*k+1]=c[1]; d[4*k+2]=c[2]; d[4*k+3]=255; }
  oc.putImageData(img,0,0);
  ctx.save(); ctx.imageSmoothingEnabled=true; ctx.drawImage(OFF,L,Tp,W,H);
  ctx.strokeStyle=rgba(C,0.5); ctx.lineWidth=1; ctx.beginPath();
  for (let l=1;l<12;l++){
    const z=lo+span*Math.pow(l/12,1.3);
    for (let j=0;j<ny;j++) for (let i=0;i<nx;i++){
      const a=vals[j*N+i], b=vals[j*N+i+1], c=vals[(j+1)*N+i+1], e=vals[(j+1)*N+i];
      const pts=[];
      const t=(p,q)=>(z-p)/(q-p);
      if ((a<z)!==(b<z)) pts.push([px(i+t(a,b)),py(j)]);
      if ((b<z)!==(c<z)) pts.push([px(i+1),py(j+t(b,c))]);
      if ((e<z)!==(c<z)) pts.push([px(i+t(e,c)),py(j+1)]);
      if ((a<z)!==(e<z)) pts.push([px(i),py(j+t(a,e))]);
      for (let k=0;k+1<pts.length;k+=2){ ctx.moveTo(pts[k][0],pts[k][1]); ctx.lineTo(pts[k+1][0],pts[k+1][1]); }
    }
  }
  ctx.stroke(); ctx.restore();
}
function ptr(cv, e){ const r=cv.getBoundingClientRect(); return [e.clientX-r.left, e.clientY-r.top]; }
const set = (id, html) => { const el=document.getElementById(id); if (el) el.innerHTML = html; };
const stats = rows => rows.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');
const pill = (cls, txt) => `<span class="pill ${cls}">${txt}</span>`;

/* ---------- controls ---------- */
function bindRange(id, fmtFn, onInput){
  const el=$('#'+id), out=$('#'+id+'-o');
  const upd=()=>{ if (out) out.textContent=fmtFn(+el.value); };
  el.addEventListener('input',()=>{ upd(); onInput(+el.value); });
  upd(); return el;
}
function bindSeg(id, onPick){
  const box=$('#'+id);
  box.addEventListener('click',e=>{
    const b=e.target.closest('button'); if (!b) return;
    box.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', x===b));
    onPick(b.dataset.v);
  });
}
function pressSeg(id, v){ $$('#'+id+' button').forEach(x=>x.setAttribute('aria-pressed', x.dataset.v===String(v))); }

/* ---------- animation runners: one per topic, all stopped when the topic changes ---------- */
const RUNS = {};
function run(key, btn, stepFn, every=0, labels=['Run','Pause']){
  if (RUNS[key]){ stop(key); return; }
  const st={ btn, labels, last:0, id:0 };
  RUNS[key]=st; if (btn) btn.textContent=labels[1];
  const f=now=>{
    if (RUNS[key]!==st) return;
    if (now-st.last>=every){ st.last=now; if (stepFn()===false){ stop(key); return; } }
    st.id=requestAnimationFrame(f);
  };
  st.id=requestAnimationFrame(f);
}
function stop(key){ const st=RUNS[key]; if (!st) return; cancelAnimationFrame(st.id); delete RUNS[key]; if (st.btn) st.btn.textContent=st.labels[0]; }
function stopAll(){ Object.keys(RUNS).forEach(stop); }
// drag helper: onDown(px,py,e) returns a handler (px,py)=>{} or null
function draggable(cv, onDown, onUp){
  let h=null;
  cv.addEventListener('pointerdown',e=>{ const [x,y]=ptr(cv,e); h=onDown(x,y,e); if (h){ cv.setPointerCapture(e.pointerId); e.preventDefault(); } });
  cv.addEventListener('pointermove',e=>{ if (h){ const [x,y]=ptr(cv,e); h(x,y); } });
  ['pointerup','pointercancel'].forEach(t=>cv.addEventListener(t,()=>{ if (h){ h=null; onUp && onUp(); } }));
}
const near = (pts, x, y, r=14) => { let best=-1, bd=r*r; pts.forEach((p,i)=>{ const d=(p[0]-x)**2+(p[1]-y)**2; if (d<bd){ bd=d; best=i; } }); return best; };
