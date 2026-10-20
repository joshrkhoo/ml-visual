
/* =====================================================================
   5.1 1D convolution: flip, slide, multiply, add
   ===================================================================== */
const C1_F = {
  pulse: n=>n>=6&&n<=13?1:0,
  steps: n=>n<8?0.3:n<16?1:0.5,
  noisy: (()=>{ const r=rng(8), v=[...Array(24)].map((_,n)=>0.55+0.35*Math.sin(n/3.2)+0.25*gauss(r)); return n=>v[n]; })(),
  spike: n=>n===5?1:n===12?1:n===18?-0.7:0,
};
const C1_G = { box:[0.2,0.2,0.2,0.2,0.2], tri:[1,2,3,2,1].map(x=>x/9), decay:[1,0.6,0.36,0.22,0.13].map(x=>x/2.31), diff:[1,-1] };
const C1 = { f:'pulse', g:'box', flip:true, m:10, N:24 };
const c1Out = m => { const g=C1_G[C1.g], K=g.length; let s=0; for (let i=0;i<K;i++){ const n=m-K+1+i; if (n>=0&&n<C1.N) s+=C1_F[C1.f](n)*(C1.flip?g[K-1-i]:g[i]); } return s; };
function c1Max(){ return C1.N+C1_G[C1.g].length-2; }
function drawConv1d(){
  const g=C1_G[C1.g], K=g.length, M=c1Max(); C1.m=Math.min(C1.m,M);
  const el=$('#c1-m'); el.max=M; el.value=C1.m; $('#c1-m-o').textContent=C1.m;
  const {ctx,w,h}=prep($('#cv-c1')), L=46, R=14, n0=-(K-1), n1=C1.N+K-2;
  const X=n=>L+(n-n0+0.5)*(w-L-R)/(n1-n0+1), band=(h-30)/3, rows=[30, 30+band, 30+2*band];
  const fmax=1.1, gmax=Math.max(...g.map(Math.abs))*1.15;
  // overlap window
  const lo=C1.m-K+1, hi=C1.m;
  ctx.fillStyle=T['acc-soft']; ctx.fillRect(X(lo)-(X(1)-X(0))/2, rows[0], (X(hi)-X(lo))+(X(1)-X(0)), band*2-6);
  const stems=(r, vals, col, vmax, ring, faint)=>{
    const y0=r+band*0.62, sc=band*0.48/vmax;
    ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(L,y0); ctx.lineTo(w-R,y0); ctx.stroke();
    vals.forEach(([n,v,hl])=>{ const x=X(n), y=y0-v*sc; ctx.strokeStyle=faint?rgba(RGB.one,.4):col; ctx.lineWidth=hl?2.4:1.6; ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y); ctx.stroke();
      if (faint) { ctx.beginPath(); ctx.arc(x,y,2.5,0,TAU); ctx.fillStyle=rgba(RGB.one,.4); ctx.fill(); } else dotAt(ctx,x,y,col,hl?4.5:3.5,ring); });
  };
  const small=w<620;
  stems(rows[0], [...Array(C1.N)].map((_,n)=>[n,C1_F[C1.f](n),n>=lo&&n<=hi]), T.v1, fmax);
  label(ctx,'f(n)',8,rows[0]+band*0.3,T.v1,F.lab);
  const gv=[]; for (let i=0;i<K;i++) gv.push([lo+i, C1.flip?g[K-1-i]:g[i], true]);
  stems(rows[1], gv, T.v2, gmax);
  label(ctx,C1.flip?'g(m−n)':'g(n−m)',8,rows[1]+band*0.3,T.v2,small?F.small:F.lab);
  const outs=[]; for (let m=0;m<=M;m++) outs.push(m);
  { const y0=rows[2]+band*0.62, om=Math.max(1e-6,...outs.map(m=>Math.abs(c1Out(m))))*1.15, sc=band*0.48/om;
    ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(L,y0); ctx.lineTo(w-R,y0); ctx.stroke();
    outs.forEach(m=>{ const v=c1Out(m), x=X(m), y=y0-v*sc, done=m<=C1.m;
      ctx.strokeStyle=done?T.acc:rgba(RGB.one,.3); ctx.lineWidth=m===C1.m?2.6:1.5; ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y); ctx.stroke();
      if (done) dotAt(ctx,x,y,T.acc,m===C1.m?5:3.5); });
  }
  label(ctx,'(f∗g)(m)',8,rows[2]+band*0.3,T.acc,small?F.small:F.lab);
  ctx.font=F.small; ctx.fillStyle=T.muted; ctx.textAlign='center'; ctx.textBaseline='top';
  for (let n=Math.ceil(n0/4)*4;n<=n1;n+=4) ctx.fillText(n,X(n),h-12);
  title(ctx,`m = ${C1.m}: the kernel covers n = ${lo}…${hi}`);
  const terms=[]; for (let i=0;i<K;i++){ const n=lo+i; if (n<0||n>=C1.N) continue; const gi=C1.flip?K-1-i:i; terms.push(`f(${n})·g(${C1.flip?C1.m-n:n-lo}) = ${fmt(C1_F[C1.f](n))}×${fmt(g[gi])}`); }
  set('c1-sum', `(f∗g)(${C1.m}) = ${terms.length?terms.join(' + '):'0 (no overlap)'}<br>= <b class="ca">${fmt(c1Out(C1.m),3)}</b>`);
}
bindSeg('c1-f', v=>{ C1.f=v; drawConv1d(); });
bindSeg('c1-g', v=>{ C1.g=v; drawConv1d(); });
$('#c1-flip').addEventListener('change',e=>{ C1.flip=e.target.checked; drawConv1d(); });
bindRange('c1-m', v=>v, v=>{ stop('c1'); C1.m=v; drawConv1d(); });
$('#c1-play').addEventListener('click',e=>{ if (C1.m>=c1Max()) C1.m=0; run('c1',e.target,()=>{ C1.m++; drawConv1d(); return C1.m<c1Max(); },220,['Slide','Pause']); });
draggable($('#cv-c1'),(px)=>{ stop('c1'); const cv=$('#cv-c1'), K=C1_G[C1.g].length, n0=-(K-1), n1=C1.N+K-2;
  const f=x=>{ const n=Math.round((x-46)/((cv.clientWidth-60)/(n1-n0+1))-0.5+n0); C1.m=clamp(n,0,c1Max()); drawConv1d(); }; f(px); return f; });

/* =====================================================================
   5.2 Image filters
   ===================================================================== */
const FI_N = 28;
const KERNELS = { id:[0,0,0,0,1,0,0,0,0], box:Array(9).fill(1/9), w5:[1,1,1,1,5,1,1,1,1].map(x=>x/13), sharp:[0,-1,0,-1,5,-1,0,-1,0], sx:[-1,0,1,-2,0,2,-1,0,1], sy:[-1,-2,-1,0,0,0,1,2,1], lap:[0,1,0,1,-4,1,0,1,0] };
const FI = { img:null, kind:'seven', K:KERNELS.box.slice(), rep:1, hover:[16,13] };
function segDist(px,py,a,b){ const dx=b[0]-a[0], dy=b[1]-a[1], t=clamp(((px-a[0])*dx+(py-a[1])*dy)/(dx*dx+dy*dy),0,1); return Math.hypot(px-a[0]-t*dx,py-a[1]-t*dy); }
function fiMake(kind){
  const N=FI_N, I=new Float64Array(N*N), r=rng(3);
  for (let y=0;y<N;y++) for (let x=0;x<N;x++){
    let v=0;
    if (kind==='seven'){ const d=Math.min(segDist(x,y,[7,6],[21,6]),segDist(x,y,[21,6],[12,23])); v=clamp(1.7-d,0,1); }
    else if (kind==='shapes'){ if (x>=3&&x<=11&&y>=3&&y<=11) v=1; if (Math.hypot(x-19,y-8)<=4.5) v=0.7; if (Math.abs(y-(x*0.5+12))<1 && x<26) v=1; if (y>=21&&y<=25&&x>=3) v=Math.max(v,(x-3)/24); }
    else if (kind==='noise'){ v=(x>=8&&x<=19&&y>=8&&y<=19?0.8:0.15)+0.22*gauss(r); }
    I[y*N+x]=clamp(v,0,1);
  }
  FI.img=I; FI.kind=kind;
}
function fiConv(I){
  const N=FI_N, O=new Float64Array(N*N), K=FI.K;
  for (let y=0;y<N;y++) for (let x=0;x<N;x++){ let s=0; for (let m=-1;m<=1;m++) for (let n=-1;n<=1;n++){ const yy=y+m, xx=x+n; if (yy>=0&&yy<N&&xx>=0&&xx<N) s+=I[yy*N+xx]*K[(m+1)*3+n+1]; } O[y*N+x]=s; }
  return O;
}
function fiGeom(cv){ const w=cv.clientWidth, h=cv.clientHeight, s=Math.floor(Math.min(w-20,h-40)/FI_N); return { s, x0:Math.round((w-s*FI_N)/2), y0:Math.round(30+(h-30-s*FI_N)/2) }; }
function fiGridUI(){ $('#fi-grid').innerHTML = FI.K.map((k,i)=>`<input type="number" step="0.1" id="fi-k${i}" aria-label="kernel cell ${i+1}" value="${+k.toFixed(3)}">`).join(''); }
function drawFilters(){
  if (!FI.img) fiMake('seven');
  let O=FI.img; for (let r=0;r<FI.rep;r++) O=fiConv(O);
  const signed=FI.K.some(k=>k<0), N=FI_N;
  { const cv=$('#cv-fi-in'), {ctx,w,h}=prep(cv), g=fiGeom(cv);
    for (let y=0;y<N;y++) for (let x=0;x<N;x++){ ctx.fillStyle=rgba(mix(RGB.panel,RGB.ink,FI.img[y*N+x])); ctx.fillRect(g.x0+x*g.s,g.y0+y*g.s,g.s,g.s); }
    ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.strokeRect(g.x0-.5,g.y0-.5,g.s*N+1,g.s*N+1);
    if (FI.hover){ const [hx,hy]=FI.hover, rr=FI.rep; ctx.strokeStyle=T.acc; ctx.lineWidth=2.5; ctx.fillStyle=T['acc-soft'];
      const X0=g.x0+(hx-rr)*g.s, Y0=g.y0+(hy-rr)*g.s, S=(2*rr+1)*g.s; ctx.fillRect(X0,Y0,S,S); ctx.strokeRect(X0,Y0,S,S); }
    title(ctx,`input ${N}×${N}`);
  }
  { const cv=$('#cv-fi-out'), {ctx,w,h}=prep(cv), g=fiGeom(cv);
    const mx=Math.max(1e-9,...O.map(Math.abs));
    for (let y=0;y<N;y++) for (let x=0;x<N;x++){ const v=O[y*N+x];
      ctx.fillStyle = signed ? rgba(v>=0?mix(RGB.panel,RGB.v1,v/mx):mix(RGB.panel,RGB.v2,-v/mx)) : rgba(mix(RGB.panel,RGB.ink,clamp(v/Math.max(1,mx),0,1)));
      ctx.fillRect(g.x0+x*g.s,g.y0+y*g.s,g.s,g.s); }
    ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.strokeRect(g.x0-.5,g.y0-.5,g.s*N+1,g.s*N+1);
    if (FI.hover){ ctx.strokeStyle=T.acc; ctx.lineWidth=2.5; ctx.strokeRect(g.x0+FI.hover[0]*g.s,g.y0+FI.hover[1]*g.s,g.s,g.s); }
    title(ctx,`output S${FI.rep>1?` after ${FI.rep} passes`:''}${signed?` · range ±${fmt(mx,2)}`:''}`);
  }
  if (FI.hover && FI.rep===1){
    const [hx,hy]=FI.hover, terms=[];
    for (let m=-1;m<=1;m++) for (let n=-1;n<=1;n++){ const yy=hy+m, xx=hx+n, k=FI.K[(m+1)*3+n+1]; if (k===0) continue; const v=(yy>=0&&yy<N&&xx>=0&&xx<N)?FI.img[yy*N+xx]:0; terms.push(`${fmt(v)}×${fmt(k,2)}`); }
    set('fi-sum', `S(${hy}, ${hx}) = ${terms.join(' + ')||'0'}<br>= <b class="ca">${fmt(O[hy*N+hx],3)}</b>`);
  } else set('fi-sum', FI.hover ? `S(${FI.hover[1]}, ${FI.hover[0]}) = <b class="ca">${fmt(O[FI.hover[1]*N+FI.hover[0]],3)}</b> after ${FI.rep} passes; the shaded input block is its ${2*FI.rep+1}×${2*FI.rep+1} receptive field.` : '<span style="font-family:var(--f-body);color:var(--muted)">Hover a pixel of the output.</span>');
}
bindSeg('fi-k', v=>{ FI.K=KERNELS[v].slice(); fiGridUI(); drawFilters(); });
bindSeg('fi-img', v=>{ fiMake(v); drawFilters(); });
bindRange('fi-rep', v=>v+'×', v=>{ FI.rep=v; drawFilters(); });
$('#fi-grid').addEventListener('input',e=>{ const i=+e.target.id.replace('fi-k',''); if (e.target.value!==''&&isFinite(+e.target.value)){ FI.K[i]=+e.target.value; pressSeg('fi-k',''); drawFilters(); } });
{ const cv=$('#cv-fi-out'); cv.style.touchAction='auto';
  const hov=e=>{ const g=fiGeom(cv), [px,py]=ptr(cv,e), x=Math.floor((px-g.x0)/g.s), y=Math.floor((py-g.y0)/g.s); FI.hover=(x>=0&&y>=0&&x<FI_N&&y<FI_N)?[x,y]:null; drawFilters(); };
  cv.addEventListener('pointermove',hov); cv.addEventListener('pointerdown',hov); cv.addEventListener('pointerleave',e=>{ if (e.pointerType==='mouse'){ FI.hover=null; drawFilters(); } });
  const ci=$('#cv-fi-in');
  draggable(ci,(px,py)=>{ const paint=(x,y)=>{ const g=fiGeom(ci), gx=(x-g.x0)/g.s-0.5, gy=(y-g.y0)/g.s-0.5;
      for (let yy=Math.floor(gy-2);yy<=gy+2;yy++) for (let xx=Math.floor(gx-2);xx<=gx+2;xx++){ if (yy<0||xx<0||yy>=FI_N||xx>=FI_N) continue; const d=Math.hypot(xx-gx,yy-gy); FI.img[yy*FI_N+xx]=Math.max(FI.img[yy*FI_N+xx],clamp(1.6-d,0,1)); }
      drawFilters(); }; paint(px,py); return paint; });
}
fiGridUI();

/* =====================================================================
   5.3 Padding and stride
   ===================================================================== */
const ST = { n:7, k:3, p:0, s:1, pos:0, inp:null };
function stIn(){ const r=rng(21); ST.inp=[...Array(9)].map(()=>[...Array(9)].map(()=>Math.floor(r()*4))); }
const stKer = (m,n) => ((m+2*n)%3)-1;
const stOut = () => Math.floor((ST.n+2*ST.p-ST.k)/ST.s)+1;
const stAt = (y,x) => { const yy=y-ST.p, xx=x-ST.p; return (yy>=0&&xx>=0&&yy<ST.n&&xx<ST.n)?ST.inp[yy][xx]:0; };
function stVal(r,c){ let s=0; for (let m=0;m<ST.k;m++) for (let n=0;n<ST.k;n++) s+=stAt(r*ST.s+m,c*ST.s+n)*stKer(m,n); return s; }
function gridGeom(cv,N){ const w=cv.clientWidth, h=cv.clientHeight, s=Math.min(46,Math.floor(Math.min(w-24,h-44)/N)); return { s, x0:Math.round((w-s*N)/2), y0:Math.round(32+(h-32-s*N)/2) }; }
function drawStride(){
  const o=stOut(), P=ST.n+2*ST.p; ST.pos=Math.min(ST.pos,Math.max(0,o*o-1));
  const r=Math.floor(ST.pos/Math.max(1,o)), c=ST.pos%Math.max(1,o);
  { const cv=$('#cv-st-in'), {ctx,w,h}=prep(cv), g=gridGeom(cv,P);
    for (let y=0;y<P;y++) for (let x=0;x<P;x++){
      const pad=y<ST.p||x<ST.p||y>=ST.p+ST.n||x>=ST.p+ST.n, X=g.x0+x*g.s, Y=g.y0+y*g.s;
      ctx.fillStyle=pad?T.grid:T.panel; ctx.fillRect(X,Y,g.s,g.s); ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.strokeRect(X+.5,Y+.5,g.s-1,g.s-1);
      if (g.s>=16) label(ctx,String(stAt(y,x)),X+g.s/2,Y+g.s/2,pad?T.muted:T.ink,g.s>=30?F.num:F.small,'center'); }
    if (o>0){ const X=g.x0+c*ST.s*g.s, Y=g.y0+r*ST.s*g.s, S=ST.k*g.s;
      ctx.fillStyle=T['acc-soft']; ctx.fillRect(X,Y,S,S); ctx.strokeStyle=T.acc; ctx.lineWidth=3; ctx.strokeRect(X+1.5,Y+1.5,S-3,S-3);
      if (g.s>=26) for (let m=0;m<ST.k;m++) for (let n=0;n<ST.k;n++) label(ctx,(stKer(m,n)<0?'−':'')+(Math.abs(stKer(m,n))?'1':'0'),X+n*g.s+g.s-5,Y+m*g.s+8,T.acc,F.small,'right'); }
    title(ctx,`${ST.n}×${ST.n} input, padded to ${P}×${P}`);
  }
  { const cv=$('#cv-st-out'), {ctx,w,h}=prep(cv);
    if (o<=0){ title(ctx,'Kernel larger than the padded input: no output'); }
    else { const g=gridGeom(cv,Math.max(o,3));
      const x0=Math.round((w-g.s*o)/2);
      for (let y=0;y<o;y++) for (let x=0;x<o;x++){ const k=y*o+x, X=x0+x*g.s, Y=g.y0+y*g.s;
        ctx.fillStyle=k===ST.pos?T['acc-soft']:T.panel; ctx.fillRect(X,Y,g.s,g.s);
        ctx.strokeStyle=k===ST.pos?T.acc:T.rule; ctx.lineWidth=k===ST.pos?2.5:1; ctx.strokeRect(X+.5,Y+.5,g.s-1,g.s-1);
        if (k<=ST.pos && g.s>=16) label(ctx,fmt(stVal(y,x),0),X+g.s/2,Y+g.s/2,k===ST.pos?T.acc:T.ink,g.s>=30?F.num:F.small,'center'); }
      cv._g={x0,y0:g.y0,s:g.s,o};
      title(ctx,`${o}×${o} output`); }
  }
  { const {ctx,w,h}=prep($('#cv-st-depth')), n0=32, Lyr=6, f=(n,p,s)=>Math.floor((n+2*p-ST.k)/s)+1;
    const a=[n0], b=[n0]; for (let i=0;i<Lyr;i++){ a.push(Math.max(0,f(a[i],ST.p,ST.s))); b.push(Math.max(0,f(b[i],0,1))); }
    const v=makeView2(w,h,-0.6,Lyr+0.6,0,n0*1.12,{l:36,r:14,t:30,b:24});
    drawAxes2(ctx,v,w,h,'layer','size n',{xStep:1});
    const bw=(v.X(1)-v.X(0))*0.36;
    for (let i=0;i<=Lyr;i++){ ctx.fillStyle=T.one; ctx.globalAlpha=0.45; ctx.fillRect(v.X(i)-bw,v.Y(b[i]),bw,v.Y(0)-v.Y(b[i])); ctx.globalAlpha=1;
      ctx.fillStyle=T.acc; ctx.fillRect(v.X(i),v.Y(a[i]),bw,v.Y(0)-v.Y(a[i]));
      if (bw>=9) label(ctx,String(a[i]),v.X(i)+bw/2,v.Y(a[i])-8,T.acc,F.small,'center'); }
    title(ctx,'A 32×32 input through 6 such layers');
  }
  set('st-stats', stats([
    ['output size', o>0?`⌊(${ST.n} + 2·${ST.p} − ${ST.k}) / ${ST.s}⌋ + 1 = ${o}`:'none'],
    ['positions visited', o>0?o*o:0],
    ['multiply-adds', o>0?o*o*ST.k*ST.k:0],
    ['size kept?', o===ST.n?pill('good','same size'):o<ST.n?pill('warn',`shrinks by ${ST.n-o}`):pill('info','grows')],
  ]));
}
['n','k','p','s'].forEach(key=>bindRange('st-'+key, v=>v, v=>{ stop('st'); ST[key]=v; ST.pos=0; drawStride(); }));
$('#st-play').addEventListener('click',e=>{ const o=stOut(); if (ST.pos>=o*o-1) ST.pos=0; run('st',e.target,()=>{ ST.pos++; drawStride(); return ST.pos<stOut()**2-1; },300,['Slide','Pause']); });
$('#cv-st-out').addEventListener('click',e=>{ const c=e.currentTarget, g=c._g; if (!g) return; const [px,py]=ptr(c,e), x=Math.floor((px-g.x0)/g.s), y=Math.floor((py-g.y0)/g.s); if (x>=0&&y>=0&&x<g.o&&y<g.o){ stop('st'); ST.pos=y*g.o+x; drawStride(); } });
stIn();

/* =====================================================================
   5.4 Pooling
   ===================================================================== */
const PO = { N:12, type:'max', q:2, blobs:null, off:[0,0], prev:null, prevIn:null, seed:2, hover:null };
function poNew(){ const r=rng(++PO.seed); PO.blobs=[...Array(4)].map(()=>[1+Math.floor(r()*10),1+Math.floor(r()*10),0.6+0.4*r()]); PO.off=[0,0]; PO.prev=null; PO.prevIn=null; }
function poMap(){ const N=PO.N, M=[]; for (let y=0;y<N;y++){ const row=[]; for (let x=0;x<N;x++){ let v=0; PO.blobs.forEach(([bx,by,a])=>{ const dx=x-bx-PO.off[0], dy=y-by-PO.off[1]; v=Math.max(v, a*(dx===0&&dy===0?1:Math.abs(dx)+Math.abs(dy)===1?0.45:0)); }); row.push(v); } M.push(row); } return M; }
function poPool(M){ const q=PO.q, n=Math.floor(PO.N/q), P=[], A=[];
  for (let i=0;i<n;i++){ const pr=[], ar=[]; for (let j=0;j<n;j++){ let best=-1, at=[0,0], sum=0;
    for (let a=0;a<q;a++) for (let b=0;b<q;b++){ const v=M[i*q+a][j*q+b]; sum+=v; if (v>best){ best=v; at=[i*q+a,j*q+b]; } }
    pr.push(PO.type==='max'?best:sum/(q*q)); ar.push(at); } P.push(pr); A.push(ar); }
  return {P,A}; }
function drawPool(){
  const M=poMap(), {P,A}=poPool(M), q=PO.q, n=P.length;
  { const cv=$('#cv-po-in'), {ctx,w,h}=prep(cv), g=gridGeom(cv,PO.N);
    for (let y=0;y<PO.N;y++) for (let x=0;x<PO.N;x++){ ctx.fillStyle=rgba(mix(RGB.panel,RGB.v1,M[y][x]*0.9)); ctx.fillRect(g.x0+x*g.s,g.y0+y*g.s,g.s,g.s); }
    ctx.strokeStyle=T.grid; ctx.lineWidth=1; for (let i=0;i<=PO.N;i++){ ctx.beginPath(); ctx.moveTo(g.x0+i*g.s+.5,g.y0); ctx.lineTo(g.x0+i*g.s+.5,g.y0+PO.N*g.s); ctx.moveTo(g.x0,g.y0+i*g.s+.5); ctx.lineTo(g.x0+PO.N*g.s,g.y0+i*g.s+.5); ctx.stroke(); }
    ctx.strokeStyle=T.ink; ctx.lineWidth=1.6; for (let i=0;i<=n;i++){ ctx.beginPath(); ctx.moveTo(g.x0+i*q*g.s,g.y0); ctx.lineTo(g.x0+i*q*g.s,g.y0+n*q*g.s); ctx.moveTo(g.x0,g.y0+i*q*g.s); ctx.lineTo(g.x0+n*q*g.s,g.y0+i*q*g.s); ctx.stroke(); }
    if (PO.type==='max') A.forEach(r=>r.forEach(([y,x])=>{ if (M[y][x]>0) ringAt(ctx,g.x0+(x+.5)*g.s,g.y0+(y+.5)*g.s,T.acc,g.s*0.32,2); }));
    if (PO.hover){ const [hy,hx]=PO.hover; ctx.strokeStyle=T.acc; ctx.lineWidth=3; ctx.strokeRect(g.x0+hx*q*g.s,g.y0+hy*q*g.s,q*g.s,q*g.s); }
    title(ctx,`${PO.N}×${PO.N} feature map · shifted (${PO.off[0]}, ${PO.off[1]})`);
  }
  { const cv=$('#cv-po-out'), {ctx,w,h}=prep(cv), g=gridGeom(cv,n);
    for (let y=0;y<n;y++) for (let x=0;x<n;x++){ const X=g.x0+x*g.s, Y=g.y0+y*g.s;
      ctx.fillStyle=rgba(mix(RGB.panel,RGB.v1,P[y][x]*0.9)); ctx.fillRect(X,Y,g.s,g.s);
      ctx.strokeStyle=T.rule; ctx.lineWidth=1; ctx.strokeRect(X+.5,Y+.5,g.s-1,g.s-1);
      if (g.s>=26) label(ctx,fmt(P[y][x],2),X+g.s/2,Y+g.s/2,T.ink,F.small,'center');
      if (PO.prev){ const same=Math.abs(PO.prev[y][x]-P[y][x])<1e-9; if (P[y][x]>0||PO.prev[y][x]>0) { ctx.strokeStyle=same?T.ok:T.bad; ctx.lineWidth=2.5; ctx.strokeRect(X+3,Y+3,g.s-6,g.s-6); } } }
    if (PO.hover){ const [hy,hx]=PO.hover; ctx.strokeStyle=T.acc; ctx.lineWidth=3; ctx.strokeRect(g.x0+hx*g.s,g.y0+hy*g.s,g.s,g.s); }
    cv._g={...g,n};
    title(ctx,`${n}×${n} after ${PO.type} pooling`);
  }
  let si='—', so='—';
  if (PO.prev && PO.prevIn){
    const act=(a,b)=>a>0||b>0;
    let ti=0, ci=0; for (let y=0;y<PO.N;y++) for (let x=0;x<PO.N;x++) if (act(PO.prevIn[y][x],M[y][x])){ ti++; if (Math.abs(PO.prevIn[y][x]-M[y][x])>1e-9) ci++; }
    let to=0, co=0; for (let y=0;y<n;y++) for (let x=0;x<n;x++) if (act(PO.prev[y][x],P[y][x])){ to++; if (Math.abs(PO.prev[y][x]-P[y][x])>1e-9) co++; }
    si=`${ci} of ${ti} (${fmt(100*ci/Math.max(1,ti),0)}%)`; so=`${co} of ${to} (${fmt(100*co/Math.max(1,to),0)}%)`;
  }
  set('po-stats', stats([
    ['active input cells changed by the shift', si],
    ['active pooled cells changed', so],
    ['map size', `${PO.N}×${PO.N} → ${n}×${n}`],
  ]));
}
bindSeg('po-type', v=>{ PO.type=v; PO.prev=null; drawPool(); });
bindSeg('po-q', v=>{ PO.q=+v; PO.prev=null; drawPool(); });
$$('[data-shift]').forEach(b=>b.addEventListener('click',()=>{ const [dx,dy]=b.dataset.shift.split(',').map(Number), M=poMap(); PO.prevIn=M; PO.prev=poPool(M).P; PO.off=[clamp(PO.off[0]+dx,-1,1),clamp(PO.off[1]+dy,-1,1)]; drawPool(); }));
$('#po-new').addEventListener('click',()=>{ poNew(); drawPool(); });
{ const cv=$('#cv-po-out'); cv.style.touchAction='auto';
  cv.addEventListener('pointermove',e=>{ const g=cv._g; if (!g) return; const [px,py]=ptr(cv,e), x=Math.floor((px-g.x0)/g.s), y=Math.floor((py-g.y0)/g.s); const hv=(x>=0&&y>=0&&x<g.n&&y<g.n)?[y,x]:null; if (String(hv)!==String(PO.hover)){ PO.hover=hv; drawPool(); } });
  cv.addEventListener('pointerleave',()=>{ PO.hover=null; drawPool(); }); }
poNew();

/* =====================================================================
   5.5 Receptive field and parameter sharing
   ===================================================================== */
const RF = { k:3, L:3, s:1, sel:null, n:32, ci:3, co:16, pk:3 };
function rfLayers(){
  let top=9, sizes;
  for (;;){ sizes=[top]; for (let l=0;l<RF.L;l++) sizes.unshift((sizes[0]-1)*RF.s+RF.k); if (sizes[0]<=48 || top===1) break; top--; }
  return sizes; // sizes[0] = input, sizes[L] = top
}
function drawRfield(){
  const sizes=rfLayers(), L=RF.L; if (RF.sel===null || RF.sel>=sizes[L]) RF.sel=Math.floor(sizes[L]/2);
  { const cv=$('#cv-rf'), {ctx,w,h}=prep(cv), n0=sizes[0], padX=24;
    const X=u=>padX+(u+0.5)*(w-2*padX)/n0, Y=l=>h-24-l*(h-80)/Math.max(1,L);
    const stride=l=>Math.pow(RF.s,l), rfl=[1]; for (let l=1;l<=L;l++) rfl.push(rfl[l-1]+(RF.k-1)*stride(l-1));
    const centre=(l,i)=>i*stride(l)+(rfl[l]-1)/2;
    const hl=[]; hl[L]=new Set([RF.sel]); for (let l=L;l>0;l--){ hl[l-1]=new Set(); hl[l].forEach(i=>{ for (let j=0;j<RF.k;j++) hl[l-1].add(i*RF.s+j); }); }
    const rad=Math.max(2,Math.min(7,(w-2*padX)/n0*0.32));
    for (let l=1;l<=L;l++) for (let i=0;i<sizes[l];i++) for (let j=0;j<RF.k;j++){ const on=hl[l].has(i);
      ctx.strokeStyle=on?T.acc:rgba(RGB.one,.18); ctx.lineWidth=on?1.6:1; ctx.beginPath(); ctx.moveTo(X(centre(l,i)),Y(l)); ctx.lineTo(X(centre(l-1,i*RF.s+j)),Y(l-1)); ctx.stroke(); }
    for (let l=0;l<=L;l++){ for (let i=0;i<sizes[l];i++){ const on=hl[l].has(i); ctx.beginPath(); ctx.arc(X(centre(l,i)),Y(l),on?rad+1:rad,0,TAU); ctx.fillStyle=on?T.acc:T.panel; ctx.fill(); ctx.lineWidth=1.2; ctx.strokeStyle=on?T.acc:T.muted; ctx.stroke(); }
      label(ctx,l?`layer ${l}`:'input',6,Y(l)-12,T.muted,F.small); }
    cv._hit={ X, Y, centre, top:sizes[L], L };
    title(ctx,`receptive field of the selected unit: ${rfl[L]} input units`);
  }
  { const {ctx,w,h}=prep($('#cv-rf-params')), n=RF.n, ci=RF.ci, co=RF.co, k=RF.pk;
    const rows=[['fully connected',(n*n*ci)*(n*n*co),'v2'],['locally connected',n*n*co*(k*k*ci+1),'v1'],['convolution',co*(k*k*ci+1),'acc']];
    const mx=Math.ceil(Math.log10(Math.max(...rows.map(r=>r[1]))))+0.3;
    const v=makeView2(w,h,0,mx,0,3,{l:16,r:16,t:30,b:24}), bh=(v.Y(0)-v.Y(1))*0.5;
    drawAxes2(ctx,v,w,h,'parameters (log₁₀)','',{xStep:Math.max(1,Math.ceil(mx/6)),xFmt:x=>logFmt(x)});
    rows.forEach(([nm,c,col],i)=>{ const y=v.Y(2.5-i)-bh/2, x1=v.X(Math.log10(c));
      ctx.fillStyle=T[col]; ctx.fillRect(v.X(0),y,Math.max(2,x1-v.X(0)),bh);
      const txt=`${nm}: ${compact(c)}`; ctx.font=F.num; const inside=x1-v.X(0)>ctx.measureText(txt).width+16;
      if (inside){ ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.fillStyle=T.panel; ctx.fillText(txt,v.X(0)+8,y+bh/2); } else label(ctx,txt,x1+6,y+bh/2,T.ink,F.num,'left'); });
    title(ctx,`${n}×${n}×${ci} → ${n}×${n}×${co}, ${k}×${k} kernels`);
  }
}
['k','l','s'].forEach(key=>bindRange('rf-'+key, v=>v, v=>{ RF[key==='l'?'L':key]=v; RF.sel=null; drawRfield(); }));
bindRange('rf-n', v=>v, v=>{ RF.n=v; drawRfield(); });
bindRange('rf-ci', v=>v, v=>{ RF.ci=v; drawRfield(); });
bindRange('rf-co', v=>v, v=>{ RF.co=v; drawRfield(); });
bindRange('rf-pk', v=>`${v}×${v}`, v=>{ RF.pk=v; drawRfield(); });
$('#cv-rf').addEventListener('click',e=>{ const c=e.currentTarget, g=c._hit; if (!g) return; const [px,py]=ptr(c,e); if (Math.abs(py-g.Y(g.L))>24) return;
  let best=0, bd=1e9; for (let i=0;i<g.top;i++){ const d=Math.abs(g.X(g.centre(g.L,i))-px); if (d<bd){ bd=d; best=i; } } RF.sel=best; drawRfield(); });

/* =====================================================================
   topic list, theme, resize
   ===================================================================== */
const DRAW = { linreg:drawLinreg, linclf:drawLinclf, overfit:drawOverfit, cv:drawCV, perceptron:drawPerceptron, boolean:drawBoolean, adaline:drawAdaline, activ:drawActiv, backprop:drawBackprop, minima:drawMinima, mlp:drawMLP, approx:drawApprox, conv1d:drawConv1d, filters:drawFilters, stride:drawStride, pool:drawPool, rfield:drawRfield };
/* Topic list: one entry per lecture; add a lecture by adding a row here and its panels above */
const CHAPS = [
  { n:2, t:'Linear regression & classification', items:[['linreg','2.1','Fitting a line by GD'],['linclf','2.2','Score, margin & hinge']] },
  { n:3, t:'Model selection', items:[['overfit','3.1','Overfitting & bias–variance'],['cv','3.2','k-fold cross-validation']] },
  { n:4, t:'Neural networks', items:[['perceptron','4.1','Perceptron learning'],['boolean','4.2','AND, OR, XOR'],['adaline','4.3','Learning rate & SGD'],['activ','4.4','Activation functions'],['backprop','4.5','Backprop on a graph'],['minima','4.6','Momentum & local minima'],['mlp','4.7','Training an MLP'],['approx','4.8','Universal approximation']] },
  { n:5, t:'Convolutional networks', items:[['conv1d','5.1','Convolution'],['filters','5.2','Image filters'],['stride','5.3','Padding & stride'],['pool','5.4','Max pooling'],['rfield','5.5','Receptive field & sharing']] },
];
const ORDER = CHAPS.flatMap(c=>c.items.map(it=>({ id:it[0], sec:it[1], name:it[2], ch:c.n })));
const esc = x => x.replace(/&/g,'&amp;').replace(/</g,'&lt;');
$('#toc-list').innerHTML = CHAPS.map(c=>`<div class="chap ch${c.n}" data-ch="${c.n}" data-open="false">
  <button class="chap-h" aria-expanded="false"><span class="cn">L${c.n}</span><span>${esc(c.t)}</span><span class="chev" aria-hidden="true">▸</span></button>
  <div class="chap-items">${c.items.map(([id,sec,nm])=>`<button class="topic" id="t-${id}" data-topic="${id}" aria-controls="${id}"><span class="sec">${sec}</span><span>${esc(nm)}</span></button>`).join('')}</div></div>`).join('');
const side=$('.side'), sideBtn=$('.side-toggle'), find=$('#toc-find');
function openChap(el, on){ el.dataset.open=on; el.querySelector('.chap-h').setAttribute('aria-expanded', on); }
function setMenu(on){ side.dataset.open=on; sideBtn.setAttribute('aria-expanded', on); }
$$('.chap-h').forEach(b=>b.addEventListener('click',()=>{ const c=b.parentElement; openChap(c, c.dataset.open!=='true'); }));
sideBtn.addEventListener('click',()=>setMenu(side.dataset.open!=='true'));
document.addEventListener('click',e=>{ if (side.dataset.open==='true' && !side.contains(e.target)) setMenu(false); });
document.addEventListener('keydown',e=>{ if (e.key==='Escape' && side.dataset.open==='true'){ setMenu(false); sideBtn.focus(); } });
find.addEventListener('input',()=>{
  const q=find.value.trim().toLowerCase();
  $$('.chap').forEach(c=>{
    let any=false;
    c.querySelectorAll('.topic').forEach(t=>{ const hit=!q || t.textContent.toLowerCase().includes(q) || c.querySelector('.chap-h').textContent.toLowerCase().includes(q); t.hidden=!hit; any=any||hit; });
    c.hidden=!any; if (q) openChap(c, any); else openChap(c, c.contains($('.topic[aria-current="page"]')));
  });
});
find.addEventListener('keydown',e=>{ if (e.key==='Enter'){ const t=$$('.topic').find(x=>!x.hidden && !x.closest('.chap').hidden); if (t) t.click(); } });
$('#toc').addEventListener('keydown',e=>{
  if (e.key!=='ArrowDown' && e.key!=='ArrowUp') return;
  const vis=$$('.find, .chap-h, .topic').filter(x=>x.offsetParent!==null), i=vis.indexOf(document.activeElement);
  if (i<0) return; e.preventDefault(); vis[(i+(e.key==='ArrowDown'?1:-1)+vis.length)%vis.length].focus();
});
let active='linreg';
function show(id){
  if (!DRAW[id]) id='linreg';
  stopAll(); active=id;
  $$('.topic').forEach(t=>{ if (t.dataset.topic===id) t.setAttribute('aria-current','page'); else t.removeAttribute('aria-current'); });
  $$('.panel').forEach(p=>p.hidden = p.id!==id);
  const k=ORDER.findIndex(o=>o.id===id), cur=ORDER[k];
  $$('.chap').forEach(c=>{ if (+c.dataset.ch===cur.ch) openChap(c,true); });
  $('#toc-cur').textContent=`${cur.sec} · ${cur.name}`;
  const pv=ORDER[k-1], nx=ORDER[k+1];
  $('#pager').innerHTML=(pv?`<button data-topic="${pv.id}"><span class="dir">← Previous</span>${pv.sec} · ${esc(pv.name)}</button>`:'')+(nx?`<button class="nx" data-topic="${nx.id}"><span class="dir">Next →</span>${nx.sec} · ${esc(nx.name)}</button>`:'');
  try{ localStorage.setItem('glab-tab', id); }catch(e){}
  if (location.hash.slice(1)!==id) history.replaceState(null,'','#'+id);
  DRAW[id]();
}
function go(id){ show(id); setMenu(false); const top=$('.shell').getBoundingClientRect().top+window.scrollY-8; if (window.scrollY>top) window.scrollTo({top}); }
$$('.topic').forEach(t=>t.addEventListener('click',()=>go(t.dataset.topic)));
$('#pager').addEventListener('click',e=>{ const b=e.target.closest('[data-topic]'); if (b) go(b.dataset.topic); });
window.addEventListener('hashchange',()=>{ const id=location.hash.slice(1); if (DRAW[id] && id!==active) show(id); });

readTokens();
let start='linreg';
try{ start = location.hash.slice(1) || localStorage.getItem('glab-tab') || 'linreg'; }catch(e){}
show(start);

const redraw=()=>{ readTokens(); DRAW[active](); };
if (window.matchMedia) matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
let rz; new ResizeObserver(()=>{ cancelAnimationFrame(rz); rz=requestAnimationFrame(()=>DRAW[active]()); }).observe($('.wrap'));
if (document.fonts && document.fonts.ready) document.fonts.ready.then(()=>DRAW[active]());
// Hooks for scripted screenshots (harmless otherwise)
window.GLAB = { show, DRAW, LR, LC, OF, CVS, PC, BO, AD, AC, BP, MM, MLP, UA, C1, FI, ST, PO, RF, mlpEpoch, pcStep, adStep, mmStep };
})();
</script>
