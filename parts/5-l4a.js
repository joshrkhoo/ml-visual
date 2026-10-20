
/* =====================================================================
   4.1 Perceptron learning rule
   ===================================================================== */
const PC_LECT = [[5,1,0],[2,1,0],[1,1,1],[3,3,1],[4,2,0],[2,3,1]];
const PC = { set:'lecture', ex:PC_LECT.map(e=>e.slice()), w:[0,0,0], prev:null, alpha:1, it:0, idx:0, streak:0, trace:[], hist:[[0,0,0]], miss:[], seed:1 };
const pcOut = (w,e) => (w[0]+w[1]*e[0]+w[2]*e[1])>=0?1:0;
const pcDone = () => PC.streak>=PC.ex.length;
function pcReset(w=[0,0,0]){ PC.w=w.slice(); PC.prev=null; PC.it=0; PC.idx=0; PC.streak=0; PC.trace=[]; PC.hist=[w.slice()]; PC.miss=[]; }
function pcStep(){
  if (pcDone()) return false;
  const e=PC.ex[PC.idx], I=[1,e[0],e[1]], O=pcOut(PC.w,e), Tt=e[2], wold=PC.w.slice();
  PC.it++;
  if (O!==Tt){ PC.prev=wold; PC.w=PC.w.map((wj,j)=>wj+PC.alpha*I[j]*(Tt-O)); PC.streak=0; PC.miss.push(PC.it); }
  else PC.streak++;
  PC.trace.push({ it:PC.it, e:PC.idx, wold, I, T:Tt, O, wnew:PC.w.slice() });
  PC.hist.push(PC.w.slice());
  PC.idx=(PC.idx+1)%PC.ex.length;
  return true;
}
function pcData(kind){
  PC.set=kind; const r=rng(++PC.seed);
  if (kind==='lecture') PC.ex=PC_LECT.map(e=>e.slice());
  else if (kind==='sep'){ const a=-0.6+r()*1.2, c=2.2+r()*0.6; PC.ex=[];
    while (PC.ex.length<10){ const x=0.4+r()*5.2, y=0.3+r()*3.9, s=y-(c+a*(x-3)); if (Math.abs(s)>0.35) PC.ex.push([x,y,s>0?1:0]); } }
  else { PC.ex=[]; for (let i=0;i<10;i++){ const t=i%2; PC.ex.push([clamp(3+(t?-0.5:0.5)+1.1*gauss(r),0.3,5.8), clamp(2.2+(t?0.4:-0.4)+0.9*gauss(r),0.3,4.3), t]); } }
  pcReset();
}
const pcView = () => { const c=$('#cv-pc-plane'); return makeView2(c.clientWidth,c.clientHeight,0,6.2,0,4.6,{l:34,r:14,t:30,b:24}); };
function boundarySeg(v,w){ // points on w0 + w1 x + w2 y = 0 inside the view
  const [w0,w1,w2]=w;
  if (Math.abs(w2)>Math.abs(w1)) return [[v.x0,-(w0+w1*v.x0)/w2],[v.x1,-(w0+w1*v.x1)/w2]];
  if (Math.abs(w1)>1e-12) return [[-(w0+w2*v.y0)/w1,v.y0],[-(w0+w2*v.y1)/w1,v.y1]];
  return null;
}
function drawPerceptron(){
  { const {ctx,w,h}=prep($('#cv-pc-plane')), v=pcView();
    const on=mix(RGB.panel,RGB.acc,0.14);
    if (PC.w.some(x=>x)) heat(ctx,v,w,h,(x,y)=>pcOut(PC.w,[x,y])?on:RGB.panel,6);
    drawAxes2(ctx,v,w,h,'I₁','I₂');
    clipPlot(ctx,v,w,h);
    if (PC.prev){ const s=boundarySeg(v,PC.prev); if (s) line(ctx,s.map(p=>[v.X(p[0]),v.Y(p[1])]),T.one,1.5,[5,4]); }
    const s=boundarySeg(v,PC.w); if (s) line(ctx,s.map(p=>[v.X(p[0]),v.Y(p[1])]),T.acc,2.6);
    const last=PC.trace[PC.trace.length-1];
    PC.ex.forEach((e,i)=>{
      const X=v.X(e[0]), Y=v.Y(e[1]);
      dotAt(ctx,X,Y,e[2]?T.v1:T.v2,6.5);
      if (pcOut(PC.w,e)!==e[2] && PC.w.some(x=>x)) ringAt(ctx,X,Y,T.bad,11,1.8);
      if (last && last.e===i) ringAt(ctx,X,Y,T.acc,15,2.5);
      if (PC.set==='lecture') label(ctx,`e${i+1}`,X+10,Y-11,T.muted,F.tick);
    });
    ctx.restore();
    title(ctx,`w = (${PC.w.map(x=>fmt(x,PC.alpha%1?1:0)).join(', ')})  ·  iteration ${PC.it}`);
  }
  { const {ctx,w,h}=prep($('#cv-pc-w')), H=PC.hist, N=Math.max(12,H.length-1);
    let lo=-1, hi=1; H.forEach(r=>r.forEach(x=>{ lo=Math.min(lo,x); hi=Math.max(hi,x); }));
    const pad=(hi-lo)*0.1, v=makeView2(w,h,0,N,lo-pad,hi+pad,{l:40,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'iteration','weight');
    clipPlot(ctx,v,w,h);
    line(ctx,[[v.X(0),v.Y(0)],[v.X(N),v.Y(0)]],T.rule,1);
    [T.one,T.v1,T.v2].forEach((c,j)=>{ const pts=[]; H.forEach((r,i)=>{ if (i) pts.push([v.X(i),v.Y(H[i-1][j])]); pts.push([v.X(i),v.Y(r[j])]); }); line(ctx,pts,c,2.2); });
    PC.miss.forEach(i=>{ ctx.beginPath(); ctx.arc(v.X(i),v.Y(lo-pad*0.5),3,0,TAU); ctx.fillStyle=T.bad; ctx.fill(); });
    ctx.restore();
    title(ctx,`${PC.miss.length} updates in ${PC.it} iterations`);
  }
  const rows=PC.trace.slice(-8), f=a=>'('+a.map(x=>fmt(x,PC.alpha%1?1:0)).join(' ')+')';
  set('pc-trace','<tr><th>it</th><th>w old</th><th>I</th><th>T</th><th>O</th><th>w new</th></tr>'+
    (rows.length?rows.map((r,i)=>`<tr${i===rows.length-1?' class="cur"':''}><td>${r.it}</td><td>${f(r.wold)}</td><td>(${r.I.map(x=>fmt(x,PC.set==='lecture'?0:1)).join(' ')})</td><td>${r.T}</td><td class="${r.T===r.O?'ok':'bad'}">${r.O}</td><td>${f(r.wnew)}</td></tr>`).join('')
      :'<tr><td colspan="6" style="text-align:left;font-family:var(--f-body);color:var(--muted)">Press “One example” or Run.</td></tr>'));
  set('pc-status', pcDone() ? pill('good',`Converged: a full pass with no mistakes after ${PC.it} iterations`)
    : PC.set==='non' && PC.it>=60 ? pill('bad','Not linearly separable: the weights never settle')
    : pill('info',`Next: e${PC.idx+1}`));
}
bindSeg('pc-set', v=>{ stop('pc'); pcData(v); drawPerceptron(); });
bindRange('pc-alpha', v=>fmt(v,1), v=>{ PC.alpha=v; });
$('#pc-step').addEventListener('click',()=>{ stop('pc'); pcStep(); drawPerceptron(); });
$('#pc-run').addEventListener('click',e=>run('pc',e.target,()=>{ const ok=pcStep(); drawPerceptron(); return ok && PC.it<300; },PC.set==='lecture'?380:160));
$('#pc-reset').addEventListener('click',()=>{ stop('pc'); pcReset(); drawPerceptron(); });
$('#pc-rand').addEventListener('click',()=>{ stop('pc'); const r=rng(++PC.seed); pcReset([-2+4*r(),-1+2*r(),-1+2*r()].map(x=>Math.round(x*10)/10)); drawPerceptron(); });
draggable($('#cv-pc-plane'),(px,py)=>{
  const v=pcView(), i=near(PC.ex.map(e=>[v.X(e[0]),v.Y(e[1])]),px,py);
  if (i<0) return null; stop('pc');
  return (x,y)=>{ PC.ex[i][0]=clamp(v.iX(x),0.1,6.1); PC.ex[i][1]=clamp(v.iY(y),0.1,4.5); pcReset(); drawPerceptron(); };
});

/* =====================================================================
   4.2 Boolean functions and the XOR network
   ===================================================================== */
const BOOL = { AND:(a,b)=>a&b, OR:(a,b)=>a|b, NAND:(a,b)=>1-(a&b), NOT:(a)=>1-a, XOR:(a,b)=>a^b };
const CORNERS = [[0,0],[0,1],[1,0],[1,1]];
const BO = { fn:'AND', w1:1, w2:1, th:1.5, sel:[1,0], epochs:0, learning:false };
const boOut = (a,b) => (BO.w1*a+BO.w2*b-BO.th)>=0?1:0;
function boSync(){ [['bo-w1',BO.w1],['bo-w2',BO.w2],['bo-th',BO.th]].forEach(([id,x])=>{ $('#'+id).value=x; $('#'+id+'-o').textContent=fmt(x,2); }); }
function boEpoch(){
  let wrong=0;
  CORNERS.forEach(([a,b])=>{ const T_=BOOL[BO.fn](a,b), O=boOut(a,b);
    if (O!==T_){ wrong++; const d=0.5*(T_-O); BO.th-=d; BO.w1+=d*a; BO.w2+=d*b; } });
  BO.epochs++;
  return wrong;
}
const boView = h => { const c=$('#cv-bo-single'); return eqView(c.clientWidth,h||c.clientHeight,0.5,0.5,0.9,{l:34,r:14,t:30,b:24}); };
function drawBoolean(){
  { const {ctx,w,h}=prep($('#cv-bo-single')), v=boView();
    const on=mix(RGB.panel,RGB.acc,0.16);
    heat(ctx,v,w,h,(x,y)=>(BO.w1*x+BO.w2*y-BO.th)>=0?on:RGB.panel,5);
    drawAxes2(ctx,v,w,h,'I₁','I₂',{xStep:0.5});
    clipPlot(ctx,v,w,h);
    const s=boundarySeg(v,[-BO.th,BO.w1,BO.w2]); if (s) line(ctx,s.map(p=>[v.X(p[0]),v.Y(p[1])]),T.acc,2.6);
    ctx.restore();
    CORNERS.forEach(([a,b])=>{ const t=BOOL[BO.fn](a,b), o=boOut(a,b), X=v.X(a), Y=v.Y(b);
      dotAt(ctx,X,Y,t?T.v1:T.v2,8); if (o!==t) ringAt(ctx,X,Y,T.bad,14,2.2);
      label(ctx,`O=${o}`,X+(a?-14:14),Y+(b?16:-16),o===t?T.muted:T.bad,F.num,a?'right':'left'); });
    title(ctx,`${BO.fn==='NOT'?'NOT I₁':BO.fn}: O = step(${fmt(BO.w1,1)}·I₁ + ${fmt(BO.w2,1)}·I₂ − ${fmt(BO.th,1)})`);
  }
  { const cv=$('#cv-bo-net'), {ctx,w,h}=prep(cv), [x1,x2]=BO.sel;
    const y3=(x1+x2-1.5)>=0?1:0, y=(x1+x2-2*y3-0.5)>=0?1:0;
    // network diagram (top band)
    const top=Math.min(h*0.46, 190), b0=30, bh=top-b0;
    const node=(X,Y,txt,val,r,sub,subRight)=>{ ctx.beginPath(); ctx.arc(X,Y,r,0,TAU); ctx.fillStyle=val?T.acc:T.panel; ctx.fill(); ctx.lineWidth=2; ctx.strokeStyle=T.acc; ctx.stroke();
      ctx.font=F.lab; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillStyle=val?T.panel:T.ink; ctx.fillText(txt,X,Y);
      if (sub) label(ctx,sub,subRight?X+r+6:X,subRight?Y:Y+r+10,T.muted,F.small,subRight?'left':'center'); };
    const edge=(ax,ay,bx,by,wt,col)=>{ arrow(ctx,ax,ay,bx,by,col,1.6,8); label(ctx,wt,ax+(bx-ax)*0.3,ay+(by-ay)*0.3-9,col,F.num,'center'); };
    const P1=[w*0.12,b0+bh*0.38], P2=[w*0.12,b0+bh*0.86], H3=[w*0.48,b0+bh*0.08+6], Y5=[w*0.84,b0+bh*0.6];
    const sh=(a,b,r=18)=>{ const d=Math.hypot(b[0]-a[0],b[1]-a[1]); return [a[0]+(b[0]-a[0])*r/d,a[1]+(b[1]-a[1])*r/d,b[0]-(b[0]-a[0])*r/d,b[1]-(b[1]-a[1])*r/d]; };
    edge(...sh(P1,H3),'1',T.muted); edge(...sh(P2,H3),'1',T.muted);
    edge(...sh(P1,Y5),'1',T.muted); edge(...sh(P2,Y5),'1',T.muted);
    edge(...sh(H3,Y5),'−2',T.v2);
    node(...P1,`x₁=${x1}`,x1,22); node(...P2,`x₂=${x2}`,x2,22); node(...H3,`y₃=${y3}`,y3,22,'θ = 1.5',true); node(...Y5,`y=${y}`,y,22,'θ = 0.5');
    // input space (bottom)
    const v=eqView(w,h,0.5,0.5,0.8,{l:40,r:14,t:top+20,b:24});
    const on=mix(RGB.panel,RGB.v1,0.16);
    heat(ctx,v,w,h,(a,b)=>{ const hh=(a+b-1.5)>=0?1:0; return (a+b-2*hh-0.5)>=0?on:RGB.panel; },5);
    drawAxes2(ctx,v,w,h,'x₁','x₂',{xStep:0.5});
    clipPlot(ctx,v,w,h);
    line(ctx,[[v.X(-1),v.Y(2.5)],[v.X(2.5),v.Y(-1)]],T.acc,2,[6,4]);
    line(ctx,[[v.X(-1),v.Y(1.5)],[v.X(1.5),v.Y(-1)]],T.v1,1.4,[2,3]);
    ctx.restore();
    label(ctx,'y₃: x₁ + x₂ = 1.5',v.X(1.3)+4,v.Y(0.3),T.acc,F.small,'left');
    CORNERS.forEach(([a,b])=>{ const t=a^b, X=v.X(a), Y=v.Y(b); dotAt(ctx,X,Y,t?T.v1:T.v2,7.5); if (a===x1&&b===x2) ringAt(ctx,X,Y,T.acc,13,2.5); });
    cv._v=v;
  }
  const wrong=CORNERS.filter(([a,b])=>boOut(a,b)!==BOOL[BO.fn](a,b)).length;
  set('bo-status', wrong===0 ? pill('good','All four inputs correct') + (BO.epochs?` <span class="hint">after ${BO.epochs} epochs</span>`:'')
    : BO.fn==='XOR' && BO.epochs>=12 ? pill('bad',`No line works: still ${wrong} wrong after ${BO.epochs} epochs`)
    : pill('warn',`${wrong} of 4 wrong`));
}
bindSeg('bo-fn', v=>{ stop('bo'); BO.fn=v; BO.epochs=0; drawBoolean(); });
['w1','w2','th'].forEach(k=>bindRange('bo-'+k, x=>fmt(x,2), x=>{ stop('bo'); BO[k]=x; BO.epochs=0; drawBoolean(); }));
$('#bo-zero').addEventListener('click',()=>{ stop('bo'); BO.w1=BO.w2=BO.th=0; BO.epochs=0; boSync(); drawBoolean(); });
$('#bo-learn').addEventListener('click',e=>{ BO.epochs=0; run('bo',e.target,()=>{ const wr=boEpoch(); boSync(); drawBoolean(); return wr>0 && BO.epochs<40; },350,['Learn with the rule','Pause']); });
$('#cv-bo-net').addEventListener('click',e=>{ const c=e.currentTarget, v=c._v; if (!v) return; const [px,py]=ptr(c,e); const i=near(CORNERS.map(([a,b])=>[v.X(a),v.Y(b)]),px,py,40); if (i>=0){ BO.sel=CORNERS[i]; drawBoolean(); } });

/* =====================================================================
   4.3 Adaline: learning rate, batch GD, SGD, mini-batch
   ===================================================================== */
const AD = { N:20, data:null, start:[-1.5,2.1], eta:Math.pow(10,-0.85), b:5, sched:false, show:{gd:true,sgd:true,mb:true}, M:null, t:0, seed:4 };
{ const r=rng(42); AD.data=[...Array(AD.N)].map(()=>{ const x=4*r(); return [x, 0.6+0.7*x+0.35*gauss(r)]; }); }
const adE = w => AD.data.reduce((s,[x,t])=>s+(t-w[0]-w[1]*x)**2,0)/(2*AD.N);
const AD_LS = lstsq(AD.data.map(([x])=>[1,x]), AD.data.map(d=>d[1]), 1e-12), AD_EMIN = adE(AD_LS);
const AD_LMAX = (()=>{ let a=0,b=0,d=0; AD.data.forEach(([x])=>{ a+=1; b+=x; d+=x*x; }); a/=AD.N; b/=AD.N; d/=AD.N; const m=(a+d)/2; return m+Math.sqrt(((a-d)/2)**2+b*b); })();
function adReset(){
  const r=rng(++AD.seed);
  const mk=(bs)=>({ w:AD.start.slice(), path:[AD.start.slice()], E:[adE(AD.start)], bs, order:[], r });
  AD.M={ gd:mk(AD.N), sgd:mk(1), mb:mk(AD.b) }; AD.t=0;
}
function adBatch(m){
  if (m.bs>=AD.N) return AD.data;
  const out=[]; while (out.length<m.bs){ if (!m.order.length) m.order=shuffle([...Array(AD.N).keys()],m.r); out.push(AD.data[m.order.pop()]); }
  return out;
}
function adStep(){
  const eta = AD.sched ? AD.eta/(1+AD.t/20) : AD.eta;
  Object.values(AD.M).forEach(m=>{
    if (!isFinite(m.E[m.E.length-1]) || m.E[m.E.length-1]>1e8){ m.E.push(Infinity); return; }
    const B=adBatch(m), g=[0,0];
    B.forEach(([x,t])=>{ const e=t-m.w[0]-m.w[1]*x; g[0]-=e/B.length; g[1]-=e*x/B.length; });
    m.w=[m.w[0]-eta*g[0], m.w[1]-eta*g[1]]; m.path.push(m.w.slice()); m.E.push(adE(m.w));
  });
  AD.t++;
}
const AD_COL = { gd:'acc', sgd:'v2', mb:'v1' };
const adView = () => { const c=$('#cv-ad-surf'); return makeView2(c.clientWidth,c.clientHeight,-2,3,-0.8,2.4,{l:40,r:14,t:30,b:24}); };
function drawAdaline(){
  { const {ctx,w,h}=prep($('#cv-ad-surf')), v=adView();
    contourMap(ctx,v,w,h,(a,b)=>adE([a,b])-AD_EMIN*0.98,'one');
    drawAxes2(ctx,v,w,h,'w₀','w₁');
    clipPlot(ctx,v,w,h);
    cross(ctx,v.X(AD_LS[0]),v.Y(AD_LS[1]),T.ink);
    ['mb','sgd','gd'].forEach(k=>{ if (!AD.show[k]) return; const m=AD.M[k];
      line(ctx,m.path.map(p=>[v.X(p[0]),v.Y(p[1])]),T[AD_COL[k]],k==='gd'?2.4:1.6);
      const p=m.path[m.path.length-1]; if (isFinite(p[0])&&isFinite(p[1])) dotAt(ctx,v.X(p[0]),v.Y(p[1]),T[AD_COL[k]],5.5); });
    ctx.restore();
    dotAt(ctx,v.X(AD.start[0]),v.Y(AD.start[1]),T.ink,4);
    title(ctx,`E(w) and ${AD.t} updates · × marks the minimum`);
  }
  { const {ctx,w,h}=prep($('#cv-ad-curve')), N=Math.max(40,AD.t), lg=x=>Math.log10(Math.max(x,1e-6));
    const v=makeView2(w,h,0,N,lg(AD_EMIN)-0.15,Math.max(1.2,lg(adE(AD.start))+0.3),{l:44,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'update t','E(w)',{logY:true});
    clipPlot(ctx,v,w,h);
    line(ctx,[[v.X(0),v.Y(lg(AD_EMIN))],[v.X(N),v.Y(lg(AD_EMIN))]],T.ink,1,[3,4]);
    ['mb','sgd','gd'].forEach(k=>{ if (AD.show[k]) line(ctx,AD.M[k].E.map((e,i)=>[v.X(i),v.Y(Math.min(lg(e),v.y1+1))]),T[AD_COL[k]],k==='gd'?2.4:1.5); });
    ctx.restore();
    title(ctx,'Error per update (dotted: minimum)');
  }
  const r=AD.eta*AD_LMAX, st = r<1 ? pill('good','smooth descent') : r<2 ? pill('warn','zig-zags, still converges') : pill('bad','diverges');
  const last=k=>{ const E=AD.M[k].E[AD.M[k].E.length-1]; return isFinite(E)&&E<1e8?fmtG(E):'diverged'; };
  set('ad-stats', stats([
    ['η', fmtG(AD.eta)+(AD.sched?` → ${fmtG(AD.eta/(1+AD.t/20))} now`:'')],
    ['λmax of the Hessian', fmt(AD_LMAX,2)],
    ['batch GD limit 2/λmax', fmt(2/AD_LMAX,3)],
    ['batch GD at this η', st],
    ['E: batch / SGD / mini-batch', `${last('gd')} / ${last('sgd')} / ${last('mb')}`],
    ['examples per update', `${AD.N} / 1 / ${AD.b}`],
    ['minimum E', fmt(AD_EMIN,4)],
  ]));
}
bindRange('ad-eta', v=>fmtG(Math.pow(10,v)), v=>{ AD.eta=Math.pow(10,v); stop('ad'); adReset(); adFill(); drawAdaline(); });
bindRange('ad-b', v=>v, v=>{ AD.b=v; stop('ad'); adReset(); adFill(); drawAdaline(); });
['gd','sgd','mb'].forEach(k=>$('#ad-'+k).addEventListener('change',e=>{ AD.show[k]=e.target.checked; drawAdaline(); }));
$('#ad-sched').addEventListener('change',e=>{ AD.sched=e.target.checked; stop('ad'); adReset(); drawAdaline(); });
$('#ad-step').addEventListener('click',()=>{ stop('ad'); adStep(); drawAdaline(); });
$('#ad-run').addEventListener('click',e=>run('ad',e.target,()=>{ adStep(); drawAdaline(); return AD.t<300; },60));
$('#ad-reset').addEventListener('click',()=>{ stop('ad'); adReset(); drawAdaline(); });
const adFill=()=>{ for (let i=0;i<40;i++) adStep(); };
$('#cv-ad-surf').addEventListener('click',e=>{ const c=e.currentTarget, [px,py]=ptr(c,e), v=adView(); AD.start=[v.iX(px),v.iY(py)]; stop('ad'); adReset(); adFill(); drawAdaline(); });
adReset(); adFill();

/* =====================================================================
   4.4 Activation functions
   ===================================================================== */
const ACTS = [
  { k:'step', name:'step', col:'one', f:x=>x>=0?1:0, d:x=>0, dTxt:'0 (undefined at 0)' },
  { k:'sign', name:'sign', col:'warn', f:x=>x>=0?1:-1, d:x=>0, dTxt:'0 (undefined at 0)' },
  { k:'sig', name:'sigmoid', col:'acc', f:sigm, d:x=>sigm(x)*(1-sigm(x)) },
  { k:'tanh', name:'tanh', col:'v1', f:Math.tanh, d:x=>1-Math.tanh(x)**2 },
  { k:'relu', name:'ReLU', col:'v2', f:x=>Math.max(0,x), d:x=>x>0?1:0 },
];
const AC = { x:1, show:{step:true,sign:false,sig:true,tanh:true,relu:true}, unit:'sig', s:1, c:[0.4,-0.3], dir:[1.4,0.8] };
$('#ac-show').innerHTML = ACTS.map(a=>`<label><input type="checkbox" data-act="${a.k}"${AC.show[a.k]?' checked':''}> <span style="color:var(--${a.col})">${a.name}</span></label>`).join('');
$('#ac-legend').innerHTML = '<span><i class="sw"></i>function</span><span><i class="sw dash"></i>its derivative</span><span>drag to move the input</span>';
const acView = () => { const c=$('#cv-ac-fn'); return makeView2(c.clientWidth,c.clientHeight,-5,5,-1.3,2.2,{l:36,r:14,t:30,b:24}); };
const acUView = () => { const c=$('#cv-ac-unit'); return eqView(c.clientWidth,c.clientHeight,0,0,3,{l:36,r:14,t:30,b:24}); };
function acUnitVal(x,y){
  if (AC.unit==='rbf') return Math.exp(-((x-AC.c[0])**2+(y-AC.c[1])**2)/(2*AC.s*AC.s));
  const d=AC.dir, n=Math.hypot(...d)||1; return sigm(((x-AC.c[0])*d[0]+(y-AC.c[1])*d[1])/n*3/AC.s);
}
function drawActiv(){
  { const {ctx,w,h}=prep($('#cv-ac-fn')), v=acView();
    drawAxes2(ctx,v,w,h,'net input','output');
    clipPlot(ctx,v,w,h);
    line(ctx,[[v.X(-5),v.Y(0)],[v.X(5),v.Y(0)]],T.rule,1.2);
    const xs=[...Array(401)].map((_,i)=>-5+i/40);
    ACTS.forEach(a=>{ if (!AC.show[a.k]) return; const c=T[a.col];
      const seg=(fn)=>{ const pts=[]; xs.forEach((x,i)=>{ if (i && (a.k==='step'||a.k==='sign') && xs[i-1]<0 && x>=0) pts.push(null); pts.push([v.X(x),v.Y(fn(x))]); }); return pts; };
      line(ctx,seg(a.f),c,2.4); line(ctx,seg(a.d),c,1.6,[5,4]); });
    line(ctx,[[v.X(AC.x),v.pad.t],[v.X(AC.x),h-v.pad.b]],T.ink,1,[2,3]);
    ACTS.forEach(a=>{ if (AC.show[a.k]) dotAt(ctx,v.X(AC.x),v.Y(a.f(AC.x)),T[a.col],5); });
    ctx.restore();
    title(ctx,`net input = ${fmt(AC.x)}`);
  }
  { const {ctx,w,h}=prep($('#cv-ac-unit')), v=acUView(), P=RGB.panel, A=RGB.acc;
    heat(ctx,v,w,h,(x,y)=>mix(P,A,0.55*acUnitVal(x,y)),4);
    drawAxes2(ctx,v,w,h,'x₁','x₂');
    clipPlot(ctx,v,w,h);
    if (AC.unit==='rbf'){ ctx.beginPath(); ctx.arc(v.X(AC.c[0]),v.Y(AC.c[1]),AC.s*v.sx,0,TAU); ctx.setLineDash([5,4]); ctx.strokeStyle=T.ink; ctx.lineWidth=1.4; ctx.stroke(); ctx.setLineDash([]); }
    else { const d=AC.dir, R=20; line(ctx,[[v.X(AC.c[0]-d[1]*R),v.Y(AC.c[1]+d[0]*R)],[v.X(AC.c[0]+d[1]*R),v.Y(AC.c[1]-d[0]*R)]],T.ink,1.4,[5,4]);
      arrow(ctx,v.X(AC.c[0]),v.Y(AC.c[1]),v.X(AC.c[0]+d[0]),v.Y(AC.c[1]+d[1]),T.ink,2,9); ringAt(ctx,v.X(AC.c[0]+d[0]),v.Y(AC.c[1]+d[1]),T.ink,8,1.4); }
    ctx.restore();
    dotAt(ctx,v.X(AC.c[0]),v.Y(AC.c[1]),T.ink,5);
    title(ctx, AC.unit==='rbf' ? `RBF unit: centre w_j, width σ_j = ${fmt(AC.s)}` : 'Sigmoid unit: output 0.5 on the dashed line');
  }
  set('ac-stats', stats(ACTS.filter(a=>AC.show[a.k]).map(a=>[`<span style="color:var(--${a.col})">${a.name}</span>`, `g = ${fmt(a.f(AC.x),3)} · g′ = ${a.dTxt && Math.abs(AC.x)<1e-9 ? a.dTxt : fmt(a.d(AC.x),3)}`])));
  set('ac-hint', AC.unit==='rbf' ? 'Gaussian RBF: a localized response, significant only when the input falls near the centre. Drag the centre.' : 'Sigmoid: non-zero over an infinitely large region of the input. Drag the centre dot, or the ring to turn it.');
}
$('#ac-show').addEventListener('change',e=>{ const k=e.target.dataset.act; if (k){ AC.show[k]=e.target.checked; drawActiv(); } });
bindRange('ac-x', v=>fmt(v,2), v=>{ AC.x=v; drawActiv(); });
bindRange('ac-s', v=>fmt(v,2), v=>{ AC.s=v; drawActiv(); });
bindSeg('ac-unit', v=>{ AC.unit=v; drawActiv(); });
draggable($('#cv-ac-fn'),(px)=>{ const f=x=>{ AC.x=clamp(Math.round(acView().iX(x)*20)/20,-5,5); $('#ac-x').value=AC.x; $('#ac-x-o').textContent=fmt(AC.x,2); drawActiv(); }; f(px); return f; });
draggable($('#cv-ac-unit'),(px,py)=>{
  const v=acUView(), tip=[v.X(AC.c[0]+AC.dir[0]),v.Y(AC.c[1]+AC.dir[1])];
  if (AC.unit==='sig' && Math.hypot(px-tip[0],py-tip[1])<14) return (x,y)=>{ AC.dir=[v.iX(x)-AC.c[0],v.iY(y)-AC.c[1]]; drawActiv(); };
  const f=(x,y)=>{ AC.c=[clamp(v.iX(x),-3,3),clamp(v.iY(y),-3,3)]; drawActiv(); }; f(px,py); return f;
});
