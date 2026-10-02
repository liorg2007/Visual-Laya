import json, numpy as np, collections, os
R='/home/zera/laya-vision/'
def load(p): return [json.loads(l) for l in open(p)]
def cl(i): return i.rsplit('/',1)[0]
def stats(r):
    t=np.array(r['target'],float); p=np.array(r['probs'],float); p=p/p.sum()
    g=t.argmax(); t=t/t.sum()
    return (float(p.argmax()==g), float(((p-t)**2).sum()), float(-np.log(max(p[g],1e-12))*1.0), 1.0/len(t))
rng=np.random.default_rng(0)
def agg(rows, B=2000):
    rows=[r for r in rows if r.get('probs')]
    if not rows: return None
    S=np.array([stats(r) for r in rows]); cls=sorted({cl(r['id']) for r in rows}); ci={c:i for i,c in enumerate(cls)}
    idx=np.array([ci[cl(r['id'])] for r in rows]); k=len(cls)
    sums=np.zeros((k,4)); cnt=np.zeros(k)
    np.add.at(sums,idx,S); np.add.at(cnt,idx,1)
    bs=rng.integers(0,k,(B,k)); num=sums[bs].sum(1); den=cnt[bs].sum(1)[:,None]; m=num/den
    ci95=lambda j:[float(np.percentile(m[:,j],2.5)),float(np.percentile(m[:,j],97.5))]
    mu=S.mean(0)
    return dict(n=len(rows),n_clusters=k,acc=mu[0],acc_ci=ci95(0),brier=mu[1],brier_ci=ci95(1),nll=mu[2],nll_ci=ci95(2),chance=mu[3])
def paired(a,b,B=2000):
    A={r['id']:stats(r)[0] for r in a if r.get('probs')}; Bm={r['id']:stats(r)[0] for r in b if r.get('probs')}
    ids=sorted(set(A)&set(Bm)); cls=sorted({cl(i) for i in ids}); ci={c:j for j,c in enumerate(cls)}
    d=np.array([A[i]-Bm[i] for i in ids]); idx=np.array([ci[cl(i)] for i in ids]); k=len(cls)
    s=np.zeros(k);c=np.zeros(k); np.add.at(s,idx,d); np.add.at(c,idx,1)
    bs=rng.integers(0,k,(B,k)); m=s[bs].sum(1)/c[bs].sum(1)
    return dict(diff=float(d.mean()),ci=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=len(ids))
IMG=['oxford_pets','eurosat','vqav2_yesno','cifar10','food101','aokvqa','coco']
models={}
full={'final':'eval/results/stage2_c_wise085.predictions.jsonl','caption':'eval/results/caption_laya.predictions.jsonl'}
for m,p in full.items(): models[m]=[r for r in load(R+p) if r['task'] in IMG]
extra_all={m:load(f'preds/{m}.predictions.jsonl') for m in ['final','caption'] if os.path.exists(f'preds/{m}.predictions.jsonl')}
extra={m:[r for r in v if r['task'] in ['cifar10','food101','aokvqa','coco']] for m,v in extra_all.items()}
for m in extra: models[m]=[r for r in models[m] if r['task'] not in extra[m][0:0] ]+extra[m]
# reproducibility: final fresh oxford subset vs saved
out={'main':{}, 'by_nopts':{}, 'paired_final_vs_caption':{}, 'repro':{}}
def key(r): return r['qtype'] if r['qtype']!='choice' else 'choice'
for m,rows in models.items():
    for t in IMG:
        for q in ['choice','noul']:
            sub=[r for r in rows if r['task']==t and key(r)==q]
            if sub and agg(sub): out['main'].setdefault(f'{t}|{q}',{})[m]=agg(sub)
        sub=[r for r in rows if r['task']==t and r['qtype']=='choice']
        for k in sorted({len(r['target']) for r in sub}):
            s2=[r for r in sub if len(r['target'])==k]
            out['by_nopts'].setdefault(f'{t}|{k}',{})[m]=agg(s2)
    # pooled over choice by nopts across tasks
    ch=[r for r in rows if r['qtype']=='choice']
    for k in sorted({len(r['target']) for r in ch}):
        out['by_nopts'].setdefault(f'ALL|{k}',{})[m]=agg([r for r in ch if len(r['target'])==k])
for t in IMG:
    for q in ['choice','noul']:
        a=[r for r in models['final'] if r['task']==t and key(r)==q]; 
        for other in ['caption']:
            if other in models:
                b=[r for r in models[other] if r['task']==t and key(r)==q]
                if a and b: out['paired_final_vs_'+other if other!='caption' else 'paired_final_vs_caption'].setdefault(f'{t}|{q}',paired(a,b)) if False else out.setdefault('paired_final_vs_'+other,{}).__setitem__(f'{t}|{q}',paired(a,b))
# repro
if 'final' in extra:
    saved={r['id']:r for r in load(R+full['final'])}
    d=[];agree=[]
    for r in extra_all['final']:
        if r['task']=='oxford_pets' and r['id'] in saved and r.get('probs') and saved[r['id']].get('probs'):
            d.append(np.abs(np.array(r['probs'])-np.array(saved[r['id']]['probs'])).max()); agree.append(np.argmax(r['probs'])==np.argmax(saved[r['id']]['probs']))
    out['repro']=dict(n=len(d),max_abs_prob_diff=float(max(d)) if d else None,argmax_agree=float(np.mean(agree)) if d else None)
# majority-class rate for noul
for t in IMG:
    rows=[r for r in models['final'] if r['task']==t and r['qtype']=='noul']
    if rows: out.setdefault('noul_gold_rate',{})[t]=float(np.mean([np.argmax(r['target'])==1 for r in rows]))
json.dump(out,open('results.json','w'),indent=1,default=float)
f=lambda d:f"{d['acc']:.3f} [{d['acc_ci'][0]:.3f},{d['acc_ci'][1]:.3f}]"
print('task|q|n|chance|final|caption|stock|final-caption')
for k,v in out['main'].items():
    fi=v['final']; print(k,fi['n'],round(fi['chance'],3),f(fi),'B%.3f N%.3f'%(fi['brier'],fi['nll']),'|',f(v['caption']) if 'caption' in v else '-', 'B%.3f'%v['caption']['brier'] if 'caption' in v else '','|',f(v['stock']) if 'stock' in v else '-','|',out['paired_final_vs_caption'].get(k))
print('--nopts')
for k,v in out['by_nopts'].items():
    fi=v['final']; print(k,fi['n'],round(fi['chance'],3),f(fi),'B%.3f'%fi['brier'],'|cap',f(v['caption']) if 'caption' in v else '-')
print(out['repro'],out.get('noul_gold_rate'))
