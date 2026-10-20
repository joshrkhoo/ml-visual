
/* =====================================================================
   4.5 Backprop on a computation graph
   ===================================================================== */
const BP_EX = {
  simple: { cols:3, rows:2, inputs:{x:-2,y:5,z:-4}, nodes:[
    {n:'x',op:'in',p:[0,0]},{n:'y',op:'in',p:[0,1]},{n:'z',op:'in',p:[0,2]},
    {n:'q',op:'add',in:['x','y'],p:[1,0.5]},{n:'f',op:'mul',in:['q','z'],p:[2,1.25]} ]},
  sigmoid: { cols:8, rows:4, inputs:{w0:2,x0:-1,w1:-3,x1:-2,w2:-3}, nodes:[
    {n:'w0',op:'in',p:[0,0]},{n:'x0',op:'in',p:[0,1]},{n:'w1',op:'in',p:[0,2]},{n:'x1',op:'in',p:[0,3]},{n:'w2',op:'in',p:[0,4]},
    {n:'a',op:'mul',in:['w0','x0'],p:[1,0.5]},{n:'b',op:'mul',in:['w1','x1'],p:[1,2.5]},{n:'c',op:'add',in:['a','b'],p:[2,1.5]},{n:'d',op:'add',in:['c','w2'],p:[3,2.5]},
    {n:'e',op:'mulc',k:-1,in:['d'],p:[4,2.5]},{n:'g',op:'exp',in:['e'],p:[5,2.5]},{n:'h',op:'addc',k:1,in:['g'],p:[6,2.5]},{n:'f',op:'inv',in:['h'],p:[7,2.5]} ]},
  sigwrap: { cols:5, rows:4, inputs:{w0:2,x0:-1,w1:-3,x1:-2,w2:-3}, nodes:[
    {n:'w0',op:'in',p:[0,0]},{n:'x0',op:'in',p:[0,1]},{n:'w1',op:'in',p:[0,2]},{n:'x1',op:'in',p:[0,3]},{n:'w2',op:'in',p:[0,4]},
    {n:'a',op:'mul',in:['w0','x0'],p:[1,0.5]},{n:'b',op:'mul',in:['w1','x1'],p:[1,2.5]},{n:'c',op:'add',in:['a','b'],p:[2,1.5]},{n:'d',op:'add',in:['c','w2'],p:[3,2.5]},
    {n:'f',op:'sig',in:['d'],p:[4,2.5]} ]},
  gates: { cols:4, rows:3, inputs:{x:3,y:-4,z:2,w:-1}, nodes:[
    {n:'x',op:'in',p:[0,0]},{n:'y',op:'in',p:[0,1]},{n:'z',op:'in',p:[0,2]},{n:'w',op:'in',p:[0,3]},
    {n:'m',op:'mul',in:['x','y'],p:[1,0.5]},{n:'r',op:'max',in:['z','w'],p:[1,2.5]},{n:'s',op:'add',in:['m','r'],p:[2,1.5]},{n:'f',op:'mulc',k:2,in:['s'],p:[3,1.5]} ]},
  branch: { cols:3, rows:2, inputs:{x:3,y:-2}, nodes:[
    {n:'x',op:'in',p:[0,0.3]},{n:'y',op:'in',p:[0,2]},
    {n:'q',op:'mul',in:['x','y'],p:[1,1.5]},{n:'f',op:'add',in:['q','x'],p:[2,0.8]} ]},
};
const OPS = {
  add:{ s:'+', f:a=>a.reduce((x,y)=>x+y,0), L:a=>a.map(()=>1), t:(n,a)=>a.join(' + ') },
  mul:{ s:'×', f:a=>a[0]*a[1], L:a=>[a[1],a[0]], t:(n,a)=>a.join(' × ') },
  max:{ s:'max', f:a=>Math.max(a[0],a[1]), L:a=>a[0]>=a[1]?[1,0]:[0,1], t:(n,a)=>`max(${a.join(', ')})` },
  mulc:{ s:n=>`×${n.k<0?'−'+(-n.k):n.k}`, f:(a,n)=>n.k*a[0], L:(a,n)=>[n.k], t:(n,a)=>`${fmt(n.k,0)} × ${a[0]}` },
  addc:{ s:n=>`+${n.k}`, f:(a,n)=>a[0]+n.k, L:()=>[1], t:(n,a)=>`${a[0]} + ${n.k}` },
  exp:{ s:'exp', f:a=>Math.exp(a[0]), L:a=>[Math.exp(a[0])], t:(n,a)=>`exp(${a[0]})` },
  inv:{ s:'1/x', f:a=>1/a[0], L:a=>[-1/(a[0]*a[0])], t:(n,a)=>`1 / ${a[0]}` },
  sig:{ s:'σ', f:a=>sigm(a[0]), L:a=>{ const s=sigm(a[0]); return [s*(1-s)]; }, t:(n,a)=>`σ(${a[0]})` },
};
const BP = { ex:'simple', wrap:false, step:0, vals:{} };
function bpGraph(){ return BP_EX[BP.ex==='sigmoid'&&BP.wrap?'sigwrap':BP.ex]; }
function bpInputsUI(){
  const G=bpGraph();
  BP.vals = Object.assign({}, G.inputs, BP.vals && BP.ex===BP._ex ? BP.vals : {});
  BP._ex=BP.ex;
  $('#bp-inputs').innerHTML = Object.keys(G.inputs).map(k=>`<label class="mono" style="font-size:14px">${k} <input class="num-in" type="number" step="0.5" id="bp-in-${k}" value="${BP.vals[k]}"></label>`).join('');
  $('#bp-sigwrap').hidden = BP.ex!=='sigmoid';
}
// replay the first s steps: forward over op nodes, seed, then backward over op nodes in reverse
function bpState(){
  const G=bpGraph(), byN={}; G.nodes.forEach(n=>byN[n.n]=n);
  const val={}; G.nodes.forEach(n=>{ val[n.n] = n.op==='in' ? +BP.vals[n.n] : OPS[n.op].f(n.in.map(i=>val[i]), n); });
  const ops=G.nodes.filter(n=>n.op!=='in'), seq=[...ops.map(n=>({t:'fwd',n})), {t:'seed'}, ...ops.slice().reverse().map(n=>({t:'bwd',n}))];
  const shown=new Set(G.nodes.filter(n=>n.op==='in').map(n=>n.n)), grad={}, eg={}; let note='', cur=null;
  const s=Math.min(BP.step, seq.length);
  for (let i=0;i<s;i++){
    const e=seq[i];
    if (e.t==='fwd'){ shown.add(e.n.n); cur=e.n.n;
      note=`forward: ${e.n.n} = ${OPS[e.n.op].t(e.n, e.n.in)} = ${OPS[e.n.op].t(e.n, e.n.in.map(k=>fmt(val[k])))} = ${fmt(val[e.n.n])}`; }
    else if (e.t==='seed'){ grad.f=1; cur='f'; note='backward starts at the output: ∂f/∂f = 1'; }
    else { const n=e.n, up=grad[n.n], Lc=OPS[n.op].L(n.in.map(k=>val[k]), n); cur=n.n;
      note = n.in.map((k,j)=>{ const had=k in grad, c=Lc[j]*up; grad[k]=(grad[k]||0)+c; eg[k+'>'+n.n]=c;
        return `∂f/∂${k} ${had?'+=':'='} [local ${fmt(Lc[j])}] × [upstream ${fmt(up)}] = ${fmt(c)}${had?` → total ${fmt(grad[k])}`:''}`; }).join('<br>'); }
  }
  if (s===0) note='Press Step to run the forward pass one node at a time.';
  if (s===seq.length) note+='<br><span style="font-family:var(--f-body);color:var(--muted)">Done: every input now holds ∂f/∂input.</span>';
  return { G, byN, val, shown, grad, eg, note, cur, total:seq.length, s };
}
function drawBackprop(){
  const S=bpState(), G=S.G, {ctx,w,h}=prep($('#cv-bp'));
  const small=w<620, padL=small?44:70, padR=small?50:80, top=48, bot=small?30:36;
  const P=p=>[padL+p[0]*(w-padL-padR)/Math.max(1,G.cols-1), top+p[1]*(h-top-bot)/G.rows];
  const R=small?14:19, fw=small?F.small:F.num;
  const edge=(a,b,src,dst)=>{
    const A=P(a.p), B=P(b.p), ax=A[0]+(a.op==='in'?(small?14:22):R), bx=B[0]-R-2;
    ctx.strokeStyle=T.rule; ctx.lineWidth=1.6; ctx.beginPath(); ctx.moveTo(ax,A[1]); ctx.lineTo(bx,B[1]); ctx.stroke();
    const mx=ax+(bx-ax)*0.5, my=A[1]+(B[1]-A[1])*0.5;
    if (S.shown.has(src)) label(ctx,fmt(S.val[src]),mx,my-9,T.v1,fw,'center');
    const k=src+'>'+dst; if (k in S.eg) label(ctx,fmt(S.eg[k]),mx,my+10,T.v2,fw,'center');
  };
  G.nodes.forEach(n=>{ if (n.in) n.in.forEach(k=>edge(S.byN[k],n,k,n.n)); });
  const out=S.byN.f, O=P(out.p);
  ctx.strokeStyle=T.rule; ctx.lineWidth=1.6; ctx.beginPath(); ctx.moveTo(O[0]+R,O[1]); ctx.lineTo(w-10,O[1]); ctx.stroke();
  if (S.shown.has('f')) label(ctx,fmt(S.val.f),O[0]+R+(w-10-O[0]-R)/2,O[1]-9,T.v1,fw,'center');
  if ('f' in S.grad) label(ctx,fmt(S.grad.f),O[0]+R+(w-10-O[0]-R)/2,O[1]+10,T.v2,fw,'center');
  G.nodes.forEach(n=>{
    const [X,Y]=P(n.p), isCur=S.cur===n.n;
    if (n.op==='in'){
      label(ctx,n.n,X,Y,isCur?T.acc:T.ink,small?F.lab:F.big,'center');
    } else {
      ctx.beginPath(); ctx.arc(X,Y,R,0,TAU); ctx.fillStyle=isCur?T['acc-soft']:T.panel; ctx.fill(); ctx.lineWidth=isCur?2.6:1.6; ctx.strokeStyle=isCur?T.acc:T.muted; ctx.stroke();
      const sym=typeof OPS[n.op].s==='function'?OPS[n.op].s(n):OPS[n.op].s;
      label(ctx,sym,X,Y,T.ink,sym.length>2?(small?F.small:F.num):(small?F.lab:F.big),'center');
      label(ctx,n.n,X,Y-R-8,T.muted,F.small,'center');
    }
  });
  // input gradients, written under each input name
  G.nodes.filter(n=>n.op==='in').forEach(n=>{ if (n.n in S.grad){ const [X,Y]=P(n.p); label(ctx,`∂=${fmt(S.grad[n.n])}`,X,Y+(small?13:16),T.v2,F.small,'center'); } });
  title(ctx,`step ${S.s} of ${S.total}  ·  ${S.s<=S.total/2-0.5?'forward pass':'backward pass'}`);
  set('bp-now', S.note);
  $('#bp-prev').disabled=S.s===0; $('#bp-next').disabled=S.s===S.total;
}
bindSeg('bp-ex', v=>{ stop('bp'); BP.ex=v; BP.step=0; bpInputsUI(); drawBackprop(); });
$('#bp-sig').addEventListener('change',e=>{ BP.wrap=e.target.checked; BP.step=0; drawBackprop(); });
$('#bp-inputs').addEventListener('input',e=>{ const k=e.target.id.replace('bp-in-',''); if (e.target.value!=='' && isFinite(+e.target.value)){ BP.vals[k]=+e.target.value; drawBackprop(); } });
$('#bp-next').addEventListener('click',()=>{ stop('bp'); BP.step++; drawBackprop(); });
$('#bp-prev').addEventListener('click',()=>{ stop('bp'); BP.step=Math.max(0,BP.step-1); drawBackprop(); });
$('#bp-reset').addEventListener('click',()=>{ stop('bp'); BP.step=0; drawBackprop(); });
$('#bp-play').addEventListener('click',e=>{ if (BP.step>=bpState().total) BP.step=0; run('bp',e.target,()=>{ BP.step++; drawBackprop(); return BP.step<bpState().total; },650,['Play all','Pause']); });
bpInputsUI();

/* =====================================================================
   4.6 Momentum and local minima
   ===================================================================== */
const WELLS = [[1.5,1.0,0.75,1.7],[-1.6,-0.9,0.8,1.15],[-0.9,1.9,0.55,0.95],[1.7,-1.9,0.6,0.8],[-2.3,0.9,0.45,0.6]];
function mmE(a,b){ let e=0.06*(a*a+b*b); WELLS.forEach(([cx,cy,s,A])=>{ e-=A*Math.exp(-((a-cx)**2+(b-cy)**2)/(2*s*s)); }); return e; }
function mmG(a,b){ let ga=0.12*a, gb=0.12*b; WELLS.forEach(([cx,cy,s,A])=>{ const g=A*Math.exp(-((a-cx)**2+(b-cy)**2)/(2*s*s))/(s*s); ga+=g*(a-cx); gb+=g*(b-cy); }); return [ga,gb]; }
const MM_GLOBAL = (()=>{ let p=[1.5,1.0]; for (let i=0;i<3000;i++){ const g=mmG(...p); p=[p[0]-0.05*g[0],p[1]-0.05*g[1]]; } return p; })(), MM_EMIN=mmE(...MM_GLOBAL);
const MM = { eta:0.08, a:0.9, start:[-2.4,2.8], runs:null, restarts:null, seed:9, it:0 };
function mmNew(start){ return { p:start.slice(), v:[0,0], path:[start.slice()], E:[mmE(...start)] }; }
function mmReset(){ MM.runs=[mmNew(MM.start), mmNew(MM.start)]; MM.it=0; }
function mmStepOne(r, alpha){ const g=mmG(...r.p); r.v=[-MM.eta*g[0]+alpha*r.v[0], -MM.eta*g[1]+alpha*r.v[1]]; r.p=[clamp(r.p[0]+r.v[0],-4,4),clamp(r.p[1]+r.v[1],-4,4)]; r.path.push(r.p.slice()); r.E.push(mmE(...r.p)); }
function mmStep(){ mmStepOne(MM.runs[0],0); mmStepOne(MM.runs[1],MM.a); MM.it++; }
const atGlobal = p => Math.hypot(p[0]-MM_GLOBAL[0],p[1]-MM_GLOBAL[1])<0.35;
const mmView = () => { const c=$('#cv-mm-surf'); return eqView(c.clientWidth,c.clientHeight,0,0,3.4,{l:36,r:14,t:30,b:24}); };
function drawMinima(){
  { const {ctx,w,h}=prep($('#cv-mm-surf')), v=mmView();
    contourMap(ctx,v,w,h,(a,b)=>mmE(a,b)-MM_EMIN+0.02,'one');
    drawAxes2(ctx,v,w,h,'w₁','w₂');
    clipPlot(ctx,v,w,h);
    if (MM.restarts) MM.restarts.forEach(([pl,mo])=>{ line(ctx,pl.path.map(p=>[v.X(p[0]),v.Y(p[1])]),rgba(RGB.one,.45),1); line(ctx,mo.path.map(p=>[v.X(p[0]),v.Y(p[1])]),rgba(RGB.acc,.45),1); });
    else { line(ctx,MM.runs[0].path.map(p=>[v.X(p[0]),v.Y(p[1])]),T.one,2.2); line(ctx,MM.runs[1].path.map(p=>[v.X(p[0]),v.Y(p[1])]),T.acc,2.4);
      MM.runs.forEach((r,i)=>dotAt(ctx,v.X(r.p[0]),v.Y(r.p[1]),i?T.acc:T.one,5.5)); }
    star(ctx,v.X(MM_GLOBAL[0]),v.Y(MM_GLOBAL[1]),T.ink,9);
    if (!MM.restarts) dotAt(ctx,v.X(MM.start[0]),v.Y(MM.start[1]),T.ink,4);
    else MM.restarts.forEach(([pl,mo])=>{ dotAt(ctx,v.X(pl.path[0][0]),v.Y(pl.path[0][1]),T.ink,3); });
    ctx.restore();
    title(ctx, MM.restarts ? '20 random starts: grey plain GD, teal momentum' : `iteration ${MM.it} · click to choose a start`);
  }
  { const {ctx,w,h}=prep($('#cv-mm-curve'));
    const src=MM.restarts?[]:MM.runs, N=Math.max(60,MM.it); let hi=0.3; src.forEach(r=>r.E.forEach(e=>hi=Math.max(hi,e)));
    const v=makeView2(w,h,0,N,MM_EMIN-0.15,hi+0.15,{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'iteration','E(w)');
    clipPlot(ctx,v,w,h);
    line(ctx,[[v.X(0),v.Y(MM_EMIN)],[v.X(N),v.Y(MM_EMIN)]],T.one,1.2,[5,4]);
    src.forEach((r,i)=>line(ctx,r.E.map((e,k)=>[v.X(k),v.Y(e)]),i?T.acc:T.one,2.2));
    ctx.restore();
    title(ctx, MM.restarts ? 'Run a single start to see its curve' : 'E(w) along each path');
  }
  if (MM.restarts){
    const pl=MM.restarts.filter(r=>atGlobal(r[0].p)).length, mo=MM.restarts.filter(r=>atGlobal(r[1].p)).length;
    set('mm-stats', stats([['plain GD reached ★', `${pl} of 20`],['momentum reached ★', `${mo} of 20`],['α, η', `${fmt(MM.a,2)}, ${fmt(MM.eta,2)}`]]));
  } else {
    const [pl,mo]=MM.runs, tag=r=>atGlobal(r.p)?pill('good','global minimum'):(MM.it>0&&Math.hypot(...mmG(...r.p))<2e-3?pill('warn','stuck in a local minimum'):pill('info','moving'));
    set('mm-stats', stats([
      ['plain GD', `E = ${fmt(pl.E[pl.E.length-1],3)} ${tag(pl)}`],
      ['momentum', `E = ${fmt(mo.E[mo.E.length-1],3)} ${tag(mo)}`],
      ['global minimum E', fmt(MM_EMIN,3)],
      ['iterations', MM.it],
    ]));
  }
}
bindRange('mm-eta', v=>fmt(v,2), v=>{ MM.eta=v; stop('mm'); MM.restarts=null; mmReset(); drawMinima(); });
bindRange('mm-a', v=>fmt(v,2), v=>{ MM.a=v; stop('mm'); MM.restarts=null; mmReset(); drawMinima(); });
$('#mm-run').addEventListener('click',e=>{ if (MM.restarts || MM.it>=300){ MM.restarts=null; mmReset(); } run('mm',e.target,()=>{ for (let i=0;i<2;i++) mmStep(); drawMinima(); return MM.it<300; },16); });
$('#mm-restarts').addEventListener('click',()=>{ stop('mm'); const r=rng(++MM.seed);
  MM.restarts=[...Array(20)].map(()=>{ const s=[-3.3+6.6*r(),-3.3+6.6*r()], a=mmNew(s), b=mmNew(s); for (let i=0;i<300;i++){ mmStepOne(a,0); mmStepOne(b,MM.a); } return [a,b]; });
  drawMinima(); });
$('#cv-mm-surf').addEventListener('click',e=>{ const c=e.currentTarget, [px,py]=ptr(c,e), v=mmView(); MM.start=[v.iX(px),v.iY(py)]; MM.restarts=null; mmReset(); drawMinima();
  run('mm',$('#mm-run'),()=>{ for (let i=0;i<2;i++) mmStep(); drawMinima(); return MM.it<300; },16); });
mmReset(); for (let i=0;i<300;i++) mmStep();

/* =====================================================================
   4.7 MLP playground
   ===================================================================== */
const MLP = { data:'xor', act:'tanh', layers:1, H:4, eta:0.5, a:0.9, noise:0.1, lines:true, seed:1, net:null, set:null, epoch:0, hist:[] };
function mlpData(){
  const r=rng(MLP.seed*17+MLP.data.length), n=200, pts=[], nz=()=>MLP.noise*gauss(r);
  for (let i=0;i<n;i++){
    let x, y, t;
    if (MLP.data==='xor'){ do { x=2*r()-1; y=2*r()-1; } while (Math.abs(x)<0.1||Math.abs(y)<0.1); t=x*y>0?1:0; }
    else if (MLP.data==='circle'){ t=i%2; const a=TAU*r(), rad=t?0.5*Math.sqrt(r()):0.75+0.3*r(); x=rad*Math.cos(a); y=rad*Math.sin(a); }
    else if (MLP.data==='moons'){ t=i%2; const a=Math.PI*r(); x=t?Math.cos(a)*0.8-0.35:0.45-Math.cos(a)*0.8; y=t?Math.sin(a)*0.8-0.15:0.2-Math.sin(a)*0.8; }
    else if (MLP.data==='spiral'){ t=i%2; const s=r(), a=s*3.2*Math.PI+(t?Math.PI:0), rad=0.12+s*0.95; x=rad*Math.cos(a); y=rad*Math.sin(a); }
    else { x=2*r()-1; y=2*r()-1; t=(x+0.6*y-0.1)>0?1:0; }
    pts.push([x+nz(), y+nz(), t]);
  }
  shuffle(pts, r);
  MLP.set={ train:pts.slice(0,140), test:pts.slice(140) };
}
const ACTF = {
  sigmoid:{ f:sigm, d:(a)=>a*(1-a) },
  tanh:{ f:Math.tanh, d:(a)=>1-a*a },
  relu:{ f:z=>z>0?z:0, d:(a)=>a>0?1:0 },
};
function mlpInit(){
  const r=rng(MLP.seed*101+7), sizes=[2, ...Array(MLP.layers).fill(MLP.H), 1];
  MLP.net={ sizes, W:[], V:[] };
  for (let l=1;l<sizes.length;l++){
    const s=1/Math.sqrt(sizes[l-1]);
    MLP.net.W.push([...Array(sizes[l])].map(()=>[...Array(sizes[l-1]+1)].map((_,j)=>j<sizes[l-1]?s*1.6*(2*r()-1):0.1*(2*r()-1))));
    MLP.net.V.push([...Array(sizes[l])].map(()=>Array(sizes[l-1]+1).fill(0)));
  }
  MLP.epoch=0; MLP.hist=[];
}
function mlpFwd(x){
  const A=[x], net=MLP.net, L=net.W.length;
  for (let l=0;l<L;l++){ const prev=A[l], out=net.W[l].map(row=>{ let z=row[prev.length]; for (let j=0;j<prev.length;j++) z+=row[j]*prev[j]; return l===L-1?sigm(z):ACTF[MLP.act].f(z); }); A.push(out); }
  return A;
}
const mlpLoss = S => S.reduce((s,p)=>s+0.5*(p[2]-mlpFwd([p[0],p[1]]).at(-1)[0])**2,0)/S.length;
const mlpAcc = S => S.filter(p=>(mlpFwd([p[0],p[1]]).at(-1)[0]>=0.5?1:0)===p[2]).length/S.length;
function mlpEpoch(){
  const net=MLP.net, L=net.W.length, G=net.W.map(Wl=>Wl.map(r=>r.map(()=>0))), S=MLP.set.train;
  S.forEach(p=>{
    const A=mlpFwd([p[0],p[1]]), o=A[L][0];
    let D=[(o-p[2])*o*(1-o)];
    for (let l=L-1;l>=0;l--){
      const prev=A[l];
      D.forEach((d,i)=>{ const g=G[l][i]; for (let j=0;j<prev.length;j++) g[j]+=d*prev[j]; g[prev.length]+=d; });
      if (l>0) D=prev.map((a,j)=>ACTF[MLP.act].d(a)*D.reduce((s,d,i)=>s+net.W[l][i][j]*d,0));
    }
  });
  for (let l=0;l<L;l++) net.W[l].forEach((row,i)=>row.forEach((_,j)=>{ const v=-MLP.eta*G[l][i][j]/S.length+MLP.a*net.V[l][i][j]; net.V[l][i][j]=v; row[j]+=v; }));
  MLP.epoch++;
  if (MLP.epoch%2===0 || MLP.epoch<20) MLP.hist.push([MLP.epoch, mlpLoss(S), mlpLoss(MLP.set.test)]);
}
const MLP_SPAN=1.45;
function unitColor(a, lo, hi){ const t=clamp((a-lo)/(hi-lo||1),0,1)*2-1; return t>=0?mix(RGB.panel,RGB.v1,0.85*t):mix(RGB.panel,RGB.v2,-0.85*t); }
function drawMLP(){
  if (!MLP.set) mlpData(); if (!MLP.net) mlpInit();
  const net=MLP.net, L=net.W.length;
  { const {ctx,w,h}=prep($('#cv-mlp-plane')), v=eqView(w,h,0,0,MLP_SPAN,{l:36,r:14,t:30,b:24});
    heat(ctx,v,w,h,(x,y)=>{ const o=mlpFwd([x,y])[L][0]; return mix(unitColor(o,0,1),RGB.panel,0.35); },5);
    drawAxes2(ctx,v,w,h,'x₁','x₂');
    clipPlot(ctx,v,w,h);
    if (MLP.lines) net.W[0].forEach(row=>{ const s=boundarySeg(v,[row[2],row[0],row[1]]); if (s) line(ctx,s.map(p=>[v.X(p[0]),v.Y(p[1])]),rgba(RGB.acc,.7),1.3,[5,4]); });
    MLP.set.train.forEach(p=>dotAt(ctx,v.X(p[0]),v.Y(p[1]),p[2]?T.v1:T.v2,3.6));
    MLP.set.test.forEach(p=>dotAt(ctx,v.X(p[0]),v.Y(p[1]),p[2]?T.v1:T.v2,3.6,true));
    ctx.restore();
    title(ctx,`epoch ${MLP.epoch}`);
  }
  { const {ctx,w,h}=prep($('#cv-mlp-net')), sizes=net.sizes, nc=sizes.length, G=18;
    const colX=i=>36+i*(w-72)/(nc-1), maxN=Math.max(...sizes), box=Math.min(34,(h-50)/maxN-6);
    const rowY=(l,i)=>30+(h-40)/2+(i-(sizes[l]-1)/2)*(box+6)+box/2-box/2;
    // activations over a grid for the node thumbnails
    const grid=[]; for (let j=0;j<G;j++) for (let i=0;i<G;i++) grid.push(mlpFwd([-MLP_SPAN+2*MLP_SPAN*(i+.5)/G, MLP_SPAN-2*MLP_SPAN*(j+.5)/G]));
    let wmax=0; net.W.forEach(Wl=>Wl.forEach(r=>r.slice(0,-1).forEach(x=>wmax=Math.max(wmax,Math.abs(x)))));
    net.W.forEach((Wl,l)=>Wl.forEach((row,i)=>row.slice(0,-1).forEach((wt,j)=>{
      ctx.strokeStyle=rgba(wt>=0?RGB.v1:RGB.v2, 0.25+0.6*Math.abs(wt)/(wmax||1)); ctx.lineWidth=0.5+4*Math.abs(wt)/(wmax||1);
      ctx.beginPath(); ctx.moveTo(colX(l)+box/2,rowY(l,j)); ctx.lineTo(colX(l+1)-box/2,rowY(l+1,i)); ctx.stroke(); })));
    sizes.forEach((n,l)=>{ for (let i=0;i<n;i++){
      const lo = l===0 ? -MLP_SPAN : (l<nc-1 && MLP.act==='tanh') ? -1 : 0;
      let hi = l===0 ? MLP_SPAN : 1; if (l>0 && l<nc-1 && MLP.act==='relu') hi=Math.max(1e-6,...grid.map(A=>A[l][i]));
      OFF.width=G; OFF.height=G; const oc=OFF.getContext('2d'), img=oc.createImageData(G,G);
      grid.forEach((A,k)=>{ const c=unitColor(A[l][i],lo,hi); img.data.set([c[0],c[1],c[2],255],4*k); });
      oc.putImageData(img,0,0);
      const X=colX(l)-box/2, Y=rowY(l,i)-box/2;
      ctx.imageSmoothingEnabled=true; ctx.drawImage(OFF,X,Y,box,box);
      ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.strokeRect(X+.5,Y+.5,box-1,box-1);
    }});
    label(ctx,'input',colX(0),h-10,T.muted,F.small,'center'); label(ctx,'output',colX(nc-1),h-10,T.muted,F.small,'center');
    for (let l=1;l<nc-1;l++) label(ctx,`hidden ${MLP.layers>1?l:''}`.trim(),colX(l),h-10,T.muted,F.small,'center');
    title(ctx,`${sizes.join(' → ')} · ${net.W.reduce((s,Wl)=>s+Wl.length*Wl[0].length,0)} weights`);
  }
  { const {ctx,w,h}=prep($('#cv-mlp-loss')), H=MLP.hist, lg=x=>Math.log10(Math.max(x,1e-5));
    const N=Math.max(50,MLP.epoch); let lo=-2, hi=-0.4; H.forEach(r=>{ lo=Math.min(lo,lg(r[1])-0.2,lg(r[2])-0.2); hi=Math.max(hi,lg(r[1])+0.1,lg(r[2])+0.1); });
    const v=makeView2(w,h,0,N,Math.floor(lo),Math.ceil(hi*2)/2,{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'epoch','loss',{logY:true});
    clipPlot(ctx,v,w,h);
    line(ctx,H.map(r=>[v.X(r[0]),v.Y(lg(r[1]))]),T.v1,2.2); line(ctx,H.map(r=>[v.X(r[0]),v.Y(lg(r[2]))]),T.v2,2);
    ctx.restore();
    title(ctx,'½(t − o)² averaged over examples');
  }
  const tr=mlpAcc(MLP.set.train), te=mlpAcc(MLP.set.test);
  set('mlp-stats', stats([
    ['epoch', MLP.epoch],
    ['training loss', fmtG(mlpLoss(MLP.set.train))],
    ['test loss', fmtG(mlpLoss(MLP.set.test))],
    ['training accuracy', `${fmt(100*tr,1)}%`],
    ['test accuracy', `${fmt(100*te,1)}% ${te>=0.95?pill('good','solved'):te>=0.8?pill('warn','getting there'):pill('info','learning')}`],
  ]));
}
const mlpRestart=()=>{ stop('mlp'); mlpInit(); drawMLP(); };
bindSeg('mlp-data', v=>{ MLP.data=v; mlpData(); mlpRestart(); });
bindSeg('mlp-act', v=>{ MLP.act=v; mlpRestart(); });
bindSeg('mlp-layers', v=>{ MLP.layers=+v; mlpRestart(); });
bindRange('mlp-h', v=>v, v=>{ MLP.H=v; mlpRestart(); });
bindRange('mlp-eta', v=>fmt(v,2), v=>{ MLP.eta=v; });
bindRange('mlp-a', v=>fmt(v,2), v=>{ MLP.a=v; });
bindRange('mlp-noise', v=>fmt(v,2), v=>{ MLP.noise=v; mlpData(); mlpRestart(); });
$('#mlp-lines').addEventListener('change',e=>{ MLP.lines=e.target.checked; drawMLP(); });
$('#mlp-step').addEventListener('click',()=>{ stop('mlp'); mlpEpoch(); drawMLP(); });
$('#mlp-reset').addEventListener('click',()=>{ MLP.seed++; mlpRestart(); });
$('#mlp-run').addEventListener('click',e=>run('mlp',e.target,()=>{ for (let i=0;i<6;i++) mlpEpoch(); drawMLP(); return MLP.epoch<4000; },0,['Train','Pause']));

/* =====================================================================
   4.8 Universal approximation
   ===================================================================== */
const UA_F = {
  sin: x=>0.65*Math.sin(2.6*Math.PI*x)+0.25*Math.sin(7*x),
  step: x=>x>0.1?0.8:-0.5,
  abs: x=>1.4*Math.abs(x)-0.6,
  bump: x=>0.95*Math.exp(-((x+0.4)**2)/0.02)-0.6*Math.exp(-((x-0.45)**2)/0.05),
  saw: x=>{ const u=(x+1)*1.75; return 1.2*(u-Math.floor(u))-0.6; },
};
const UA = { f:'sin', act:'sig', H:6, k:2, parts:true, cache:null };
const UA_X = [...Array(401)].map((_,i)=>-1+i/200);
function uaUnits(H){
  if (UA.act==='relu') return [...Array(H)].map((_,j)=>{ const c=-1+j*2/H; return x=>Math.max(0,x-c); });
  return [...Array(H)].map((_,j)=>{ const c=-1+(j+.5)*2/H; return x=>sigm(UA.k*H*(x-c)); });
}
function uaFit(H){
  const U=uaUnits(H), f=UA_F[UA.f], Phi=UA_X.map(x=>[1,...U.map(u=>u(x))]), c=lstsq(Phi,UA_X.map(f),1e-9);
  const pred=Phi.map(r=>dot(r,c)), err=Math.max(...pred.map((p,i)=>Math.abs(p-f(UA_X[i]))));
  return { U, c, pred, err };
}
function drawApprox(){
  if (!UA.cache) UA.cache=[...Array(40)].map((_,i)=>uaFit(i+1).err);
  const R=uaFit(UA.H), f=UA_F[UA.f];
  { const {ctx,w,h}=prep($('#cv-ua-fit')), v=makeView2(w,h,-1,1,-1.4,1.4,{l:36,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'x','');
    clipPlot(ctx,v,w,h);
    if (UA.parts) R.U.forEach((u,j)=>line(ctx,UA_X.filter((_,i)=>i%2===0).map(x=>[v.X(x),v.Y(R.c[j+1]*u(x))]),rgba(RGB.v1,.45),1.2));
    line(ctx,UA_X.map(x=>[v.X(x),v.Y(f(x))]),T.one,2,[6,4]);
    line(ctx,UA_X.map((x,i)=>[v.X(x),v.Y(R.pred[i])]),T.acc,2.6);
    ctx.restore();
    title(ctx,`${UA.H} hidden ${UA.act==='relu'?'ReLU':'sigmoid'} unit${UA.H>1?'s':''} · max error ${fmt(R.err,3)}`);
  }
  { const {ctx,w,h}=prep($('#cv-ua-err')), lg=x=>Math.log10(Math.max(x,1e-4)), E=UA.cache;
    const v=makeView2(w,h,0,41,Math.floor(Math.min(...E.map(lg))),Math.max(0.3,Math.ceil(Math.max(...E.map(lg))*2)/2),{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'hidden units H','max error',{logY:true,xStep:10});
    clipPlot(ctx,v,w,h);
    line(ctx,E.map((e,i)=>[v.X(i+1),v.Y(lg(e))]),T.acc,2.4);
    ctx.restore();
    dotAt(ctx,v.X(UA.H),v.Y(lg(E[UA.H-1])),T.v2,6);
    title(ctx,'More units, smaller worst-case error');
  }
  set('ua-stats', stats([
    ['hidden units H', UA.H],
    ['max error', fmt(R.err,4)],
    ['output weights', `${UA.H+1} (incl. bias b)`],
    ['largest |vⱼ|', fmt(Math.max(...R.c.slice(1).map(Math.abs)),2)],
  ]));
}
bindSeg('ua-f', v=>{ UA.f=v; UA.cache=null; drawApprox(); });
bindSeg('ua-act', v=>{ UA.act=v; UA.cache=null; $('#ua-k').disabled=v==='relu'; drawApprox(); });
bindRange('ua-h', v=>v, v=>{ UA.H=v; drawApprox(); });
bindRange('ua-k', v=>fmt(v,2), v=>{ UA.k=v; UA.cache=null; drawApprox(); });
$('#ua-parts').addEventListener('change',e=>{ UA.parts=e.target.checked; drawApprox(); });
