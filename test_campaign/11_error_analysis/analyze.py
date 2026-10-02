import json,collections,re,random,numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/home/zera/laya-vision/'
rng=np.random.default_rng(0)
pred={}
for l in open(R+'eval/results/stage2_c_wise085.predictions.jsonl'):
    r=json.loads(l); pred[r['id']]=r
test={}
for t in ['oxford_pets','eurosat','vqav2_yesno','ag_news','typed_decisions']:
    for l in open(R+f'data/{t}.test.jsonl'):
        r=json.loads(l); test[r['id']]=r
def wil(k,n,z=1.96):
    if n==0:return (0,0)
    p=k/n;d=1+z*z/n;c=(p+z*z/2/n)/d;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/d;return (round(c-h,3),round(c+h,3))
def acc(k,n): return dict(n=n,acc=round(k/max(n,1),4),ci=wil(k,n))
out={}
def rec(i):
    p=pred[i];t=test[i];pr=np.array(p['probs']);tg=np.array(p['target'])
    return p,t,pr,tg,int(pr.argmax()),int(tg.argmax())
# ---- confusion for choice tasks
for task in ['oxford_pets','eurosat']:
    names=None;pairs=[];N=0
    rows=[]
    for i,p in pred.items():
        if p['task']!=task or p['qtype']!='choice':continue
        p,t,pr,tg,a,b=rec(i)
        names=list(t['question']['criteria'].keys())
        rows.append((i,names[b],names[a],pr.max(),pr[b],len(names)))
    # class set consistency: options may be shuffled; use names
    cls=sorted({r[1] for r in rows}); idx={c:k for k,c in enumerate(cls)}
    M=np.zeros((len(cls),len(cls)),int)
    for r in rows: M[idx[r[1]],idx[r[2]]]+=1
    per={c:(M[idx[c],idx[c]],M[idx[c]].sum()) for c in cls}
    pc={c:dict(**acc(*per[c])) for c in cls}
    conf=sorted([(M[i,j],cls[i],cls[j]) for i in range(len(cls)) for j in range(len(cls)) if i!=j and M[i,j]>0],reverse=True)[:12]
    k=sum(r[1]==r[2] for r in rows)
    out[task]=dict(overall=acc(k,len(rows)),per_class=pc,top_confusions=[dict(n=int(a),true=b,pred=c) for a,b,c in conf])
    fig,ax=plt.subplots(figsize=(11,10) if task=='oxford_pets' else (7,6))
    Mn=M/M.sum(1,keepdims=True);ax.imshow(Mn,cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(range(len(cls)));ax.set_xticklabels(cls,rotation=90,fontsize=7);ax.set_yticks(range(len(cls)));ax.set_yticklabels(cls,fontsize=7)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if M[i,j]: ax.text(j,i,M[i,j],ha='center',va='center',fontsize=6,color='w' if Mn[i,j]>.5 else 'k')
    ax.set_xlabel('predicted');ax.set_ylabel('true');ax.set_title(f'{task} confusion (counts, row-normalised colour), n={len(rows)}')
    plt.tight_layout();plt.savefig(f'confusion_{task}.png',dpi=110);plt.close()
    out[task]['_rows']=rows
# also noul (binary class-verification) accuracy by task
for task in ['oxford_pets','eurosat']:
    k=n=0;tp=fp=fn=tn=0
    for i,p in pred.items():
        if p['task']==task and p['qtype']=='noul':
            a=int(np.argmax(p['probs']));b=int(np.argmax(p['target']));n+=1;k+=a==b
            tp+=a==1&b==1 if False else (a==1 and b==1);fp+=(a==1 and b==0);fn+=(a==0 and b==1);tn+=(a==0 and b==0)
    out[task]['verify_binary']=dict(**acc(k,n),tp=tp,fp=fp,fn=fn,tn=tn)
# ---- VQAv2 by question type
def qt(q):
    q=q.lower().strip()
    for pre in ['is there','are there','is this','is the','is it','is he','is she','are the','are these','are they','are you','does the','does this','do the','do you','do these','is that','can you','was','has','have','will','could','would','can','did','does','do','is','are']:
        if q.startswith(pre+' ') : return pre
    return 'other'
V=collections.defaultdict(lambda:[0,0,0,0,[]])  # correct,n,gt_yes,pred_yes
vq=[]
for i,p in pred.items():
    if p['task']!='vqav2_yesno':continue
    t=test[i];q=t['question']['instructions'];tg=np.array(p['target']);pr=np.array(p['probs'])
    a=int(pr.argmax());b=int(tg.argmax());s=qt(q)
    vq.append((i,q,a,b,pr[1],tg[1]))
    for key in [s,'ALL']:
        v=V[key];v[0]+=a==b;v[1]+=1;v[2]+=b==1;v[3]+=a==1
vt={}
for key,v in sorted(V.items(),key=lambda x:-x[1][1]):
    vt[key]=dict(**acc(v[0],v[1]),gt_yes_rate=round(v[2]/v[1],3),pred_yes_rate=round(v[3]/v[1],3),
      yes_recall=None,)
# yes/no recall
ys=[(a,b) for _,_,a,b,_,_ in vq]
yr=np.mean([a==1 for a,b in ys if b==1]);nr=np.mean([a==0 for a,b in ys if b==0])
out['vqav2']=dict(by_type=vt,yes_recall=round(float(yr),3),no_recall=round(float(nr),3),
  majority_baseline=round(max(np.mean([b for a,b in ys]),1-np.mean([b for a,b in ys])),3))
# target-prob distribution (soft labels) check
print('vqa target yes values',collections.Counter(round(x[5],1) for x in vq).most_common(5))
# by keyword semantics: attribute questions
kw={'color':r'\bcolor\b','number/count-ish':r'\b(any|many|two|three|more)\b','people/animal':r'\b(man|woman|person|people|dog|cat|child|boy|girl)\b','"there"':r'\bthere\b','sky/weather':r'\b(sunny|cloudy|raining|snow|sky)\b','negation':r"\bnot\b|n't"}
kwres={}
for k,pat in kw.items():
    s=[(a==b) for _,q,a,b,_,_ in vq if re.search(pat,q.lower())]
    kwres[k]=acc(sum(s),len(s))
out['vqav2']['by_keyword']=kwres
# ---- calibration / confidence
def conf_stats(task,qtype):
    rs=[(pred[i]['answer_confidence'],int(np.argmax(pred[i]['probs']))==int(np.argmax(pred[i]['target']))) for i in pred if pred[i]['task']==task and pred[i]['qtype']==qtype]
    c=np.array([r[0] for r in rs]);ok=np.array([r[1] for r in rs])
    return c,ok
cs={}
for task,qt_ in [('oxford_pets','choice'),('eurosat','choice'),('vqav2_yesno','noul'),('ag_news','choice'),('boolq','noul')]:
    c,ok=conf_stats(task,qt_)
    d=dict(n=len(c),acc=round(ok.mean(),4),mean_conf=round(c.mean(),4))
    for th in [.7,.8,.9,.95]:
        m=c>=th;d[f'wrong_at_conf>={th}']=dict(n_wrong=int((m&~ok).sum()),n_above=int(m.sum()),err_rate_above=round(float((~ok[m]).mean()),4) if m.any() else None)
    for th in [.4,.5,.6]:
        m=c<th;d[f'correct_at_conf<{th}']=dict(n_correct=int((m&ok).sum()),n_below=int(m.sum()),acc_below=round(float(ok[m].mean()),4) if m.any() else None)
    # reliability
    bins=np.linspace(0,1,11);rel=[]
    cc=np.clip(c,0,.9999)
    for k in range(10):
        m=(cc>=bins[k])&(cc<bins[k+1])
        if m.sum():rel.append((round(bins[k],1),int(m.sum()),round(float(ok[m].mean()),3),round(float(c[m].mean()),3)))
    d['reliability(bin_lo,n,acc,conf)']=rel
    cs[task]=d
out['confidence']=cs
# ---- vs option count (task-level) and option position
oc={}
for task in ['vqav2_yesno','boolq','ag_news','eurosat','oxford_pets','typed_decisions']:
    ks=collections.defaultdict(lambda:[0,0])
    for i,p in pred.items():
        if p['task']==task and p['qtype']in('choice','noul'):
            k=len(p['probs']);ks[(p['qtype'],k)][0]+=int(np.argmax(p['probs']))==int(np.argmax(p['target']));ks[(p['qtype'],k)][1]+=1
    oc[task]={f'{a}_{b}opts':acc(*v) for (a,b),v in ks.items()}
out['by_option_count']=oc
pos={}
for task in ['oxford_pets','eurosat','ag_news','typed_decisions']:
    ks=collections.defaultdict(lambda:[0,0]);pp=collections.Counter()
    for i,p in pred.items():
        if p['task']==task and p['qtype']=='choice':
            b=int(np.argmax(p['target']));a=int(np.argmax(p['probs']));ks[b][0]+=a==b;ks[b][1]+=1;pp[a]+=1
    K=len(next(p for p in pred.values() if p['task']==task and p['qtype']=='choice')['probs'])
    # group into quartiles of true position
    g=collections.defaultdict(lambda:[0,0])
    for b,(c_,n_) in ks.items():
        q=min(3,b*4//K);g[q][0]+=c_;g[q][1]+=n_
    pos[task]=dict(by_true_position_quartile={f'Q{q+1}':acc(*v) for q,v in sorted(g.items())},
      pred_position_hist=[pp[k] for k in range(K)],K=K,
      true_position_hist=[ks[k][1] for k in range(K)],
      acc_by_true_position=[round(ks[k][0]/ks[k][1],3) if ks[k][1] else None for k in range(K)])
    # chi-square on predicted-vs-true position (uniformity)
out['by_position']=pos
# ---- image properties
def imginfo(path):
    im=Image.open(R+'data/images/'+path).convert('RGB');w,h=im.size
    g=np.asarray(im.convert('L').resize((64,64)),float)/255
    return dict(w=w,h=h,area=w*h,bright=g.mean(),contrast=g.std())
prop={}
for task,qt_ in [('oxford_pets','choice'),('eurosat','choice'),('vqav2_yesno','noul')]:
    X=[];Y=[]
    cache={}
    for i,p in pred.items():
        if p['task']!=task or p['qtype']!=qt_:continue
        im=test[i]['image']
        if im not in cache: cache[im]=imginfo(im)
        d=cache[im];X.append(d);Y.append(int(np.argmax(p['probs']))==int(np.argmax(p['target'])))
    Y=np.array(Y);res={}
    for key in ['area','bright','contrast']:
        v=np.array([d[key] for d in X]);qs=np.quantile(v,[.25,.5,.75]);bi=np.digitize(v,qs)
        res[key]=dict(quartile_edges=[round(float(x),3) for x in qs],acc_by_quartile=[acc(int(Y[bi==k].sum()),int((bi==k).sum())) for k in range(4)])
        # spearman
        from scipy.stats import spearmanr,pointbiserialr
        r,pv=spearmanr(v,Y);res[key]['spearman_r_vs_correct']=round(float(r),3);res[key]['p']=round(float(pv),4)
    res['n_unique_sizes']=len({(d['w'],d['h']) for d in X})
    if task=='oxford_pets':
        ar=np.array([d['w']/d['h'] for d in X]);qs=np.quantile(ar,[.25,.5,.75]);bi=np.digitize(ar,qs)
        res['aspect']=dict(acc_by_quartile=[acc(int(Y[bi==k].sum()),int((bi==k).sum())) for k in range(4)],edges=[round(float(x),2) for x in qs])
    prop[task]=res
out['image_props']=prop
# ---- worst errors contact sheet
worst=[]
for task in ['oxford_pets','eurosat']:
    rows=sorted([r for r in out[task]['_rows'] if r[1]!=r[2]],key=lambda r:-r[3])[:8]  # most confident wrong
    for r in rows: worst.append((task,r[0],f"{task} TRUE:{r[1]} PRED:{r[2]} conf={r[3]:.2f} p(true)={r[4]:.2f}"))
vw=sorted([v for v in vq if v[2]!=v[3]],key=lambda v:-abs(v[4]-.5))[:8]
for i,q,a,b,py,ty in vw: worst.append(('vqav2_yesno',i,f"VQA Q:{q} pred={'yes' if a else 'no'} p(yes)={py:.2f} truth={'yes' if b else 'no'}"))
import textwrap
W=300;H=230;cols=4;rows_=(len(worst)+cols-1)//cols
sheet=Image.new('RGB',(cols*W,rows_*(H+60)),'white');d=ImageDraw.Draw(sheet)
try:f=ImageFont.truetype('DejaVuSans.ttf',12)
except:f=None
for k,(task,i,cap) in enumerate(worst):
    im=Image.open(R+'data/images/'+test[i]['image']).convert('RGB');im=im.resize((H,H),Image.BICUBIC) if min(im.size)<100 else im; im.thumbnail((W-6,H))
    x=(k%cols)*W;y=(k//cols)*(H+60);sheet.paste(im,(x+3,y+3))
    d.text((x+3,y+H+5),'\n'.join(textwrap.wrap(cap,46)[:4]),fill='black',font=f)
sheet.save('worst_errors_contact_sheet.png')
out['worst_listed']=[w[2] for w in worst]
for t in ['oxford_pets','eurosat']: out[t].pop('_rows')
# worst-class summary
json.dump(out,open('results.json','w'),indent=1,default=lambda o:o.item() if hasattr(o,'item') else str(o))
