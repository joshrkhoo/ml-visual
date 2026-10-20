
/* =====================================================================
   2.1 Linear regression by gradient descent
   ===================================================================== */
const LR = { pts:[[1,1],[2,3],[4,3]], w:[1,0.57], path:[[1,0.57]], eta:0.1 };
const lrLoss = (w, P=LR.pts) => P.reduce((s,[x,y])=>s+(w[0]+w[1]*x-y)**2,0)/P.length;
const lrGrad = w => { const g=[0,0]; LR.pts.forEach(([x,y])=>{ const r=w[0]+w[1]*x-y; g[0]+=2*r/LR.pts.length; g[1]+=2*r*x/LR.pts.length; }); return g; };
function lrLS(){
  const P=LR.pts.map(([x])=>[1,x]), y=LR.pts.map(p=>p[1]);
  return lstsq(P,y,1e-12);
}
function lrStep(){ const g=lrGrad(LR.w); LR.w=[LR.w[0]-LR.eta*g[0], LR.w[1]-LR.eta*g[1]]; LR.path.push(LR.w.slice()); if (LR.path.length>400) LR.path.shift(); }
function lrViews(){
  const a=$('#cv-lr-data'), b=$('#cv-lr-loss');
  return { dv: makeView2(a.clientWidth,a.clientHeight,0,5,0,4.6,{l:36,r:14,t:30,b:24}),
           lv: makeView2(b.clientWidth,b.clientHeight,-2.5,4,-1,2,{l:40,r:14,t:30,b:24}) };
}
function drawLinreg(){
  const {dv,lv}=lrViews();
  { const {ctx,w,h}=prep($('#cv-lr-data')), v=dv;
    drawAxes2(ctx,v,w,h,'x','y'); clipPlot(ctx,v,w,h);
    LR.pts.forEach(([x,y])=>line(ctx,[[v.X(x),v.Y(y)],[v.X(x),v.Y(LR.w[0]+LR.w[1]*x)]],T.v2,1.6,[4,3]));
    line(ctx,[[v.X(v.x0),v.Y(LR.w[0]+LR.w[1]*v.x0)],[v.X(v.x1),v.Y(LR.w[0]+LR.w[1]*v.x1)]],T.acc,2.6);
    LR.pts.forEach(([x,y])=>dotAt(ctx,v.X(x),v.Y(y),T.v1,6));
    ctx.restore();
    title(ctx,`f(x) = ${fmt(LR.w[0])} + ${fmt(LR.w[1])}x   ·   TrainLoss = ${fmt(lrLoss(LR.w),3)}`);
  }
  { const {ctx,w,h}=prep($('#cv-lr-loss')), v=lv;
    contourMap(ctx,v,w,h,(a,b)=>lrLoss([a,b]));
    drawAxes2(ctx,v,w,h,'w₁ (intercept)','w₂ (slope)');
    clipPlot(ctx,v,w,h);
    const ls=lrLS(); cross(ctx,v.X(ls[0]),v.Y(ls[1]),T.ink);
    line(ctx,LR.path.map(p=>[v.X(p[0]),v.Y(p[1])]),T.acc,2);
    LR.path.forEach((p,i)=>{ if (i<LR.path.length-1) { ctx.beginPath(); ctx.arc(v.X(p[0]),v.Y(p[1]),2.2,0,TAU); ctx.fillStyle=T.acc; ctx.fill(); } });
    const g=lrGrad(LR.w), nx=[LR.w[0]-LR.eta*g[0], LR.w[1]-LR.eta*g[1]];
    arrow(ctx,v.X(LR.w[0]),v.Y(LR.w[1]),v.X(nx[0]),v.Y(nx[1]),T.v2,2,9);
    dotAt(ctx,v.X(LR.w[0]),v.Y(LR.w[1]),T.acc,7);
    ctx.restore();
    title(ctx,`TrainLoss(w) over weight space · step ${LR.path.length-1}`);
  }
  const g=lrGrad(LR.w), L=lrLoss(LR.w), ls=lrLS(), Lmin=lrLoss(ls);
  const n=LR.path.length, prev=n>2?lrLoss(LR.path[n-2]):Infinity;
  const status = !isFinite(L)||L>1e4 ? pill('bad','diverged') : (L>prev+1e-12 ? pill('bad','loss went up: η too large') : Math.hypot(...g)<1e-3 ? pill('good','converged') : pill('info','descending'));
  set('lr-stats', stats([
    ['w = [w₁, w₂]', `[${fmt(LR.w[0],3)}, ${fmt(LR.w[1],3)}]`],
    ['TrainLoss(w)', fmtG(L)],
    ['∇TrainLoss', `[${fmtG(g[0])}, ${fmtG(g[1])}]`],
    ['least-squares w', `[${fmt(ls[0],3)}, ${fmt(ls[1],3)}]`],
    ['minimum loss', fmt(Lmin,4)],
    ['status', status],
  ]));
}
bindRange('lr-eta', v=>fmt(v,2), v=>{ LR.eta=v; drawLinreg(); });
$('#lr-step').addEventListener('click',()=>{ stop('lr'); lrStep(); drawLinreg(); });
$('#lr-run').addEventListener('click',e=>run('lr',e.target,()=>{ lrStep(); drawLinreg(); const L=lrLoss(LR.w); return isFinite(L) && L<1e5 && Math.hypot(...lrGrad(LR.w))>1e-4 && LR.path.length<399; },90));
$('#lr-reset').addEventListener('click',()=>{ stop('lr'); LR.w=[1,0.57]; LR.path=[LR.w.slice()]; drawLinreg(); });
$('#lr-data').addEventListener('click',()=>{ stop('lr'); LR.pts=[[1,1],[2,3],[4,3]]; LR.w=[1,0.57]; LR.path=[LR.w.slice()]; drawLinreg(); });
{ const cv=$('#cv-lr-data');
  draggable(cv,(px,py)=>{
    const v=lrViews().dv, i=near(LR.pts.map(p=>[v.X(p[0]),v.Y(p[1])]),px,py);
    let k=i;
    if (k<0){ LR.pts.push([clamp(v.iX(px),0,5),clamp(v.iY(py),0,4.6)]); k=LR.pts.length-1; }
    LR.path=[LR.w.slice()];
    return (x,y)=>{ LR.pts[k]=[clamp(v.iX(x),0,5),clamp(v.iY(y),0,4.6)]; drawLinreg(); };
  });
  cv.addEventListener('dblclick',e=>{ const [px,py]=ptr(cv,e), v=lrViews().dv, i=near(LR.pts.map(p=>[v.X(p[0]),v.Y(p[1])]),px,py); if (i>=0 && LR.pts.length>2){ LR.pts.splice(i,1); drawLinreg(); } });
  draggable($('#cv-lr-loss'),(px,py)=>{ stop('lr'); const v=lrViews().lv;
    const f=(x,y)=>{ LR.w=[v.iX(x),v.iY(y)]; LR.path=[LR.w.slice()]; drawLinreg(); }; f(px,py); return f; });
}

/* =====================================================================
   2.2 Linear classification: score, margin, hinge
   ===================================================================== */
const LC_LECT = [{x:[0,2],y:1},{x:[-2,0],y:1},{x:[1,-1],y:-1}];
const LC = { pts:LC_LECT.map(p=>({x:p.x.slice(),y:p.y})), w:[0.5,1], loss:'hinge', cls:1, eta:0.2, steps:0, seed:3 };
const LOSS = {
  zo: m => m<=0?1:0,
  hinge: m => Math.max(1-m,0),
  logistic: m => Math.log(1+Math.exp(-m)),
};
const lcMargin = p => dot(LC.w,p.x)*p.y;
function lcTrainLoss(kind){ return LC.pts.reduce((s,p)=>s+LOSS[kind](lcMargin(p)),0)/LC.pts.length; }
function lcStep(){
  const g=[0,0];
  LC.pts.forEach(p=>{ const m=lcMargin(p), c = LC.loss==='hinge' ? (m<1?-1:0) : -sigm(-m); g[0]+=c*p.y*p.x[0]/LC.pts.length; g[1]+=c*p.y*p.x[1]/LC.pts.length; });
  LC.w=[LC.w[0]-LC.eta*g[0], LC.w[1]-LC.eta*g[1]]; LC.steps++;
  return Math.hypot(...g);
}
const lcView = () => { const c=$('#cv-lc-plane'); return eqView(c.clientWidth,c.clientHeight,0,0,3.1); };
function drawLinclf(){
  { const {ctx,w,h}=prep($('#cv-lc-plane')), v=lcView();
    const P=RGB.panel, A=mix(P,RGB.v1,0.13), B=mix(P,RGB.v2,0.13);
    heat(ctx,v,w,h,(x,y)=>LC.w[0]*x+LC.w[1]*y>=0?A:B,6);
    drawAxes2(ctx,v,w,h,'x₁','x₂');
    clipPlot(ctx,v,w,h);
    const [a,b]=LC.w, n2=a*a+b*b;
    if (n2>1e-9){
      const d=[-b,a], R=20;
      const seg=c=>[[v.X(a*c/n2-d[0]*R),v.Y(b*c/n2-d[1]*R)],[v.X(a*c/n2+d[0]*R),v.Y(b*c/n2+d[1]*R)]];
      if (LC.loss==='hinge'){ line(ctx,seg(1),T.acc,1.2,[5,4]); line(ctx,seg(-1),T.acc,1.2,[5,4]); }
      line(ctx,seg(0),T.acc,2.4);
    }
    LC.pts.forEach(p=>{
      const X=v.X(p.x[0]), Y=v.Y(p.x[1]), m=lcMargin(p);
      dotAt(ctx,X,Y,p.y>0?T.v1:T.v2,6);
      if (m<=0) ringAt(ctx,X,Y,T.bad,11,2);
      if (LC.pts.length<=8) label(ctx,`m = ${fmt(m)}`,X+12,Y-10,T.muted,F.small);
    });
    ctx.restore();
    arrow(ctx,v.X(0),v.Y(0),v.X(a),v.Y(b),T.acc,2.6,11);
    ringAt(ctx,v.X(a),v.Y(b),T.acc,9,1.5);
    label(ctx,`w = [${fmt(a)}, ${fmt(b)}]`,v.X(a)+12,v.Y(b)-12,T.acc,F.num);
    title(ctx, LC.loss==='hinge' ? 'Boundary w·φ(x) = 0 and margin-1 lines w·φ(x) = ±1' : 'Boundary w·φ(x) = 0');
  }
  { const {ctx,w,h}=prep($('#cv-lc-loss')), v=makeView2(ctx.canvas.clientWidth,ctx.canvas.clientHeight,-3,3,0,4,{l:36,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'margin (w·φ(x))y','loss');
    clipPlot(ctx,v,w,h);
    const xs=[...Array(241)].map((_,i)=>-3+i*6/240);
    line(ctx,[[v.X(-3),v.Y(1)],[v.X(0),v.Y(1)],[v.X(0),v.Y(0)],[v.X(3),v.Y(0)]],T.one,2.2);
    line(ctx,xs.map(m=>[v.X(m),v.Y(LOSS.hinge(m))]),T.acc,LC.loss==='hinge'?3:1.8);
    line(ctx,xs.map(m=>[v.X(m),v.Y(LOSS.logistic(m))]),T.v1,LC.loss==='logistic'?3:1.8,[6,4]);
    LC.pts.forEach(p=>{ const m=clamp(lcMargin(p),-2.95,2.95); dotAt(ctx,v.X(m),v.Y(LOSS[LC.loss](m)),p.y>0?T.v1:T.v2,5.5); });
    ctx.restore();
    title(ctx,`Each example sits at its margin on the ${LC.loss} curve`);
  }
  const errs=LC.pts.filter(p=>lcMargin(p)<=0).length;
  set('lc-stats', stats([
    ['w', `[${fmt(LC.w[0],3)}, ${fmt(LC.w[1],3)}]`],
    ['training errors', `${errs} of ${LC.pts.length} ${errs?pill('bad','misclassified'):pill('good','all correct')}`],
    ['zero-one TrainLoss', fmt(lcTrainLoss('zo'),3)],
    ['hinge TrainLoss', fmt(lcTrainLoss('hinge'),3)],
    ['logistic TrainLoss', fmt(lcTrainLoss('logistic'),3)],
    ['GD steps', LC.steps],
  ]));
}
bindSeg('lc-loss', v=>{ LC.loss=v; drawLinclf(); });
bindSeg('lc-cls', v=>{ LC.cls=+v; });
bindRange('lc-eta', v=>fmt(v,2), v=>{ LC.eta=v; });
$('#lc-step').addEventListener('click',()=>{ stop('lc'); lcStep(); drawLinclf(); });
$('#lc-run').addEventListener('click',e=>run('lc',e.target,()=>{ const g=lcStep(); drawLinclf(); return g>1e-4 && LC.steps<2000; },70));
$('#lc-reset').addEventListener('click',()=>{ stop('lc'); LC.w=[0.5,1]; LC.steps=0; drawLinclf(); });
$('#lc-data').addEventListener('click',()=>{ stop('lc'); LC.pts=LC_LECT.map(p=>({x:p.x.slice(),y:p.y})); LC.w=[0.5,1]; LC.steps=0; drawLinclf(); });
$('#lc-more').addEventListener('click',()=>{ stop('lc'); const r=rng(++LC.seed); LC.pts=[];
  for (let i=0;i<18;i++){ const y=i%2?1:-1; LC.pts.push({x:[clamp(-0.9*y+0.9*gauss(r),-2.9,2.9), clamp(0.8*y+0.9*gauss(r),-2.9,2.9)], y}); }
  LC.w=[0.5,1]; LC.steps=0; drawLinclf(); });
draggable($('#cv-lc-plane'),(px,py)=>{
  stop('lc'); const v=lcView();
  if (Math.hypot(px-v.X(LC.w[0]),py-v.Y(LC.w[1]))<16) return (x,y)=>{ LC.w=[v.iX(x),v.iY(y)]; LC.steps=0; drawLinclf(); };
  let i=near(LC.pts.map(p=>[v.X(p.x[0]),v.Y(p.x[1])]),px,py);
  if (i<0){ LC.pts.push({x:[v.iX(px),v.iY(py)],y:LC.cls}); i=LC.pts.length-1; drawLinclf(); }
  return (x,y)=>{ LC.pts[i].x=[clamp(v.iX(x),-3,3),clamp(v.iY(y),-3,3)]; drawLinclf(); };
});
$('#cv-lc-plane').addEventListener('dblclick',e=>{ const c=e.currentTarget, [px,py]=ptr(c,e), v=lcView(), i=near(LC.pts.map(p=>[v.X(p.x[0]),v.Y(p.x[1])]),px,py); if (i>=0 && LC.pts.length>1){ LC.pts.splice(i,1); drawLinclf(); } });

/* =====================================================================
   3.1 Overfitting, bias and variance (polynomials on sin 2πx)
   ===================================================================== */
const truthF = x => Math.sin(TAU*x);
const DMAX = 12;
function sampleSet(seed, n, sd){ const r=rng(seed); return [...Array(n)].map(()=>{ const x=r(); return [x, truthF(x)+sd*gauss(r)]; }); }
function polyFit(S, d){ return lstsq(S.map(([x])=>legendre(x,d)), S.map(p=>p[1]), 1e-10); }
const polyEval = (c, x) => dot(c, legendre(x, c.length-1));
const mse = (c, S) => S.reduce((s,[x,y])=>s+(polyEval(c,x)-y)**2,0)/S.length;
const OF = { seed:11, n:15, sd:0.24, deg:3, many:true, cache:null };
function ofCompute(){
  const train=sampleSet(OF.seed,OF.n,OF.sd), test=sampleSet(9001,400,OF.sd);
  // other training sets: same inputs x, fresh noise (fixed design), so bias and variance separate cleanly
  const others=[...Array(60)].map((_,r)=>{ const g=rng(OF.seed*131+r+1); return train.map(([x])=>[x, truthF(x)+OF.sd*gauss(g)]); });
  const grid=[...Array(91)].map((_,i)=>0.05+i/100);
  const rows=[];
  for (let d=0; d<=DMAX; d++){
    const c=polyFit(train,d);
    const fits=others.map(S=>polyFit(S,d));
    const preds=fits.map(f=>grid.map(x=>polyEval(f,x)));
    let b2=0, va=0;
    grid.forEach((x,i)=>{ const m=preds.reduce((s,p)=>s+p[i],0)/preds.length; b2+=(m-truthF(x))**2; va+=preds.reduce((s,p)=>s+(p[i]-m)**2,0)/preds.length; });
    rows.push({ d, c, fits, train:mse(c,train), test:mse(c,test), bias2:b2/grid.length, vari:va/grid.length });
  }
  OF.cache={ train, rows };
}
function drawOverfit(){
  if (!OF.cache) ofCompute();
  const {train, rows}=OF.cache, R=rows[OF.deg];
  { const {ctx,w,h}=prep($('#cv-of-fit')), v=makeView2(w,h,0,1,-1.9,1.9,{l:36,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'x','y'); clipPlot(ctx,v,w,h);
    const xs=[...Array(201)].map((_,i)=>i/200);
    if (OF.many) R.fits.slice(0,20).forEach(c=>line(ctx,xs.map(x=>[v.X(x),v.Y(polyEval(c,x))]),rgba(RGB.acc,.22),1.2));
    line(ctx,xs.map(x=>[v.X(x),v.Y(truthF(x))]),T.one,2,[6,4]);
    line(ctx,xs.map(x=>[v.X(x),v.Y(polyEval(R.c,x))]),T.acc,2.8);
    train.forEach(([x,y])=>dotAt(ctx,v.X(x),v.Y(y),T.v1,4.5));
    ctx.restore();
    title(ctx,`Degree ${OF.deg} polynomial, n = ${OF.n}`);
  }
  { const {ctx,w,h}=prep($('#cv-of-err'));
    const lg=x=>Math.log10(Math.max(x,1e-4)), v=makeView2(w,h,-0.3,DMAX+0.3,-4,1.3,{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'degree','error',{logY:true,xStep:2});
    clipPlot(ctx,v,w,h);
    ctx.fillStyle=T['acc-soft']; ctx.fillRect(v.X(OF.deg-0.4),v.pad.t,v.X(OF.deg+0.4)-v.X(OF.deg-0.4),h-v.pad.t-v.pad.b);
    if (OF.sd>0) line(ctx,[[v.X(-1),v.Y(lg(OF.sd**2))],[v.X(DMAX+1),v.Y(lg(OF.sd**2))]],T.muted,1,[2,4]);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.bias2))]),T.one,1.8,[6,4]);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.vari))]),T.acc,1.8,[6,4]);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.train))]),T.v1,2.4);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.test))]),T.v2,2.4);
    rows.forEach(r=>{ dotAt(ctx,v.X(r.d),v.Y(lg(r.train)),T.v1,3.5); dotAt(ctx,v.X(r.d),v.Y(lg(r.test)),T.v2,3.5); });
    if (OF.sd>0) label(ctx,'σ² (noise floor)',v.X(DMAX)-2,v.Y(lg(OF.sd**2))-9,T.muted,F.small,'right');
    ctx.restore();
    title(ctx,'Error against model complexity (log scale)');
  }
  const best=rows.reduce((a,b)=>b.test<a.test?b:a);
  const state = OF.deg<best.d && R.bias2>R.vari ? pill('warn','underfitting: high bias') : OF.deg>best.d && R.test>best.test*1.15 ? pill('bad','overfitting: high variance') : pill('good','near the sweet spot');
  set('of-stats', stats([
    ['training error', fmtG(R.train)],
    ['testing error', fmtG(R.test)],
    ['bias²', fmtG(R.bias2)],
    ['variance', fmtG(R.vari)],
    ['best degree on test', best.d],
    ['verdict', state],
  ]));
}
bindRange('of-deg', v=>v, v=>{ OF.deg=v; drawOverfit(); });
bindRange('of-n', v=>v, v=>{ OF.n=v; OF.cache=null; drawOverfit(); });
bindRange('of-sd', v=>fmt(v,2), v=>{ OF.sd=v; OF.cache=null; drawOverfit(); });
$('#of-many').addEventListener('change',e=>{ OF.many=e.target.checked; drawOverfit(); });
$('#of-new').addEventListener('click',()=>{ OF.seed++; OF.cache=null; drawOverfit(); });

/* =====================================================================
   3.2 k-fold cross-validation
   ===================================================================== */
const CVS = { seed:5, m:30, k:5, deg:3, round:0, sd:0.3, cache:null };
const CV_DMAX = 10;
function cvCompute(){
  const data=sampleSet(CVS.seed*7+2,CVS.m,CVS.sd), test=sampleSet(777,400,CVS.sd);
  const order=shuffle([...Array(CVS.m).keys()], rng(CVS.seed));
  const fold=Array(CVS.m); order.forEach((idx,pos)=>fold[idx]=pos%CVS.k);
  const rows=[];
  for (let d=0; d<=CV_DMAX; d++){
    const folds=[], fits=[];
    for (let i=0;i<CVS.k;i++){
      const tr=data.filter((_,j)=>fold[j]!==i), va=data.filter((_,j)=>fold[j]===i);
      const c=polyFit(tr,d); fits.push(c); folds.push(mse(c,va));
    }
    const all=polyFit(data,d);
    rows.push({ d, folds, fits, cv:folds.reduce((a,b)=>a+b,0)/CVS.k, train:mse(all,data), test:mse(all,test) });
  }
  CVS.cache={ data, fold, order, rows };
}
function drawCV(){
  if (!CVS.cache) cvCompute();
  const {data, fold, rows}=CVS.cache, k=CVS.k, R=rows[CVS.deg];
  CVS.round=Math.min(CVS.round,k-1);
  { const cv=$('#cv-cv-folds'), {ctx,w,h}=prep(cv), L=64, Rr=12, Tp=26, B=8;
    const sorted=[...Array(CVS.m).keys()].sort((a,b)=>fold[a]-fold[b]||a-b);
    const cw=(w-L-Rr)/CVS.m, rh=(h-Tp-B)/k;
    for (let i=0;i<k;i++){
      const y=Tp+i*rh;
      if (i===CVS.round){ ctx.fillStyle=T['acc-soft']; ctx.fillRect(0,y,w,rh); }
      if (rh>=9) label(ctx,`round ${i+1}`,8,y+rh/2,i===CVS.round?T.ink:T.muted,rh>=14?F.tick:F.small);
      sorted.forEach((j,c)=>{ const val=fold[j]===i; ctx.fillStyle=val?T.v2:rgba(RGB.v1,.55); ctx.fillRect(L+c*cw+0.5,y+1.5,Math.max(1,cw-1),Math.max(1,rh-3)); });
    }
    title(ctx,`${k} folds of ${CVS.m} examples: ${(CVS.m/k)%1?'about ':''}${fmt(CVS.m/k,CVS.m%k?1:0)} each`);
    cv._rows={Tp,rh,k};
  }
  { const {ctx,w,h}=prep($('#cv-cv-fit')), v=makeView2(w,h,0,1,-2,2,{l:36,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'x','y'); clipPlot(ctx,v,w,h);
    const xs=[...Array(201)].map((_,i)=>i/200), c=R.fits[CVS.round];
    line(ctx,xs.map(x=>[v.X(x),v.Y(truthF(x))]),T.one,1.6,[6,4]);
    line(ctx,xs.map(x=>[v.X(x),v.Y(polyEval(c,x))]),T.acc,2.6);
    data.forEach(([x,y],j)=>{ if (fold[j]===CVS.round) dotAt(ctx,v.X(x),v.Y(y),T.v2,5,true); else dotAt(ctx,v.X(x),v.Y(y),T.v1,4.5); });
    ctx.restore();
    title(ctx,`Round ${CVS.round+1}, degree ${CVS.deg}: fold error ${fmtG(R.folds[CVS.round])}`);
  }
  { const {ctx,w,h}=prep($('#cv-cv-curve')), lg=x=>Math.log10(Math.max(x,1e-4)), v=makeView2(w,h,-0.3,CV_DMAX+0.3,-2,1.6,{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'degree','error',{logY:true,xStep:2});
    clipPlot(ctx,v,w,h);
    ctx.fillStyle=T['acc-soft']; ctx.fillRect(v.X(CVS.deg-0.4),v.pad.t,v.X(CVS.deg+0.4)-v.X(CVS.deg-0.4),h-v.pad.t-v.pad.b);
    rows.forEach(r=>r.folds.forEach(f=>{ ctx.beginPath(); ctx.arc(v.X(r.d),v.Y(lg(f)),2.2,0,TAU); ctx.fillStyle=rgba(RGB.acc,.35); ctx.fill(); }));
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.train))]),T.v1,2);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.test))]),T.v2,2,[6,4]);
    line(ctx,rows.map(r=>[v.X(r.d),v.Y(lg(r.cv))]),T.acc,2.8);
    const pick=rows.reduce((a,b)=>b.cv<a.cv?b:a);
    ctx.restore();
    star(ctx,v.X(pick.d),v.Y(lg(pick.cv)),T.acc,9);
    title(ctx,`CV picks degree ${pick.d}`);
  }
  const pick=rows.reduce((a,b)=>b.cv<a.cv?b:a), tbest=rows.reduce((a,b)=>b.test<a.test?b:a);
  set('cv-stats', stats([
    ['training fraction 1 − 1/k', fmt(1-1/k,3)],
    ['CV error at this degree', fmtG(R.cv)],
    ['CV choice', `degree ${pick.d}`],
    ['best on fresh test data', `degree ${tbest.d}`],
    ['training runs', `${k} folds × ${CV_DMAX+1} degrees = ${k*(CV_DMAX+1)}`],
    ['mode', k===CVS.m?pill('info','leave-one-out'):pill('info',`${k}-fold`)],
  ]));
}
const cvK=bindRange('cv-k', v=>v, v=>{ CVS.k=v; CVS.cache=null; drawCV(); });
bindRange('cv-deg', v=>v, v=>{ CVS.deg=v; drawCV(); });
$('#cv-loo').addEventListener('click',()=>{ CVS.k=CVS.m; cvK.value=CVS.m; $('#cv-k-o').textContent=CVS.m; CVS.cache=null; drawCV(); });
$('#cv-best').addEventListener('click',()=>{ if (!CVS.cache) cvCompute(); const p=CVS.cache.rows.reduce((a,b)=>b.cv<a.cv?b:a); CVS.deg=p.d; $('#cv-deg').value=p.d; $('#cv-deg-o').textContent=p.d; drawCV(); });
$('#cv-new').addEventListener('click',()=>{ CVS.seed++; CVS.cache=null; drawCV(); });
$('#cv-cv-folds').addEventListener('click',e=>{ const c=e.currentTarget, g=c._rows; if (!g) return; const [,y]=ptr(c,e), i=Math.floor((y-g.Tp)/g.rh); if (i>=0 && i<g.k){ CVS.round=i; drawCV(); } });
