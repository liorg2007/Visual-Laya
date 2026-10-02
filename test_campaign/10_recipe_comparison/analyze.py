import json,os,sys,glob
import numpy as np
sys.path.insert(0,'/home/zera/laya-vision')
from laya_vision.eval.metrics import row_stats
R='/home/zera/laya-vision/eval/results/'
IMG=['eurosat','oxford_pets','vqav2_yesno']; TXT=['ag_news','boolq']; TD='typed_decisions'
def load(path):
    d={}
    for l in open(path):
        r=json.loads(l)
        if r['probs'] is None: continue
        d[r['id']]=(r['task'],row_stats(r)['correct'])
    return d
def key(i,t):
    return i if t in('ag_news','boolq') else i.rsplit('/',1)[0]
rng=np.random.default_rng(0)

def build(models,ids_by_task):
    """arrays per task: matrix [n_clusters? ] -> we return per-task (cluster idx, corr matrix)"""
    out={}
    for t,ids in ids_by_task.items():
        ids=[i for i in ids if all(i in m for m in models.values())]
        if not ids: continue
        keys={}; cl=np.array([keys.setdefault(key(i,t),len(keys)) for i in ids])
        C=np.array([[m[i][1] for i in ids] for m in models.values()])  # models x rows
        out[t]=(cl,C,list(models))
    return out
def stat(data,tasks,idx=None):
    # macro accuracy across tasks -> vector per model
    accs=[]
    for t in tasks:
        if t not in data: continue
        cl,C,_=data[t]; accs.append(C.mean(1))
    return np.mean(accs,0) if accs else None
def boot(data,tasks,B=2000):
    tasks=[t for t in tasks if t in data]
    point=stat(data,tasks); res=[]
    # precompute per-task cluster sums
    pre={}
    for t in tasks:
        cl,C,_=data[t]; k=cl.max()+1
        S=np.zeros((C.shape[0],k)); 
        for j in range(C.shape[0]): S[j]=np.bincount(cl,weights=C[j],minlength=k)
        n=np.bincount(cl,minlength=k).astype(float); pre[t]=(S,n,k)
    for b in range(B):
        a=[]
        for t in tasks:
            S,n,k=pre[t]; s=rng.integers(0,k,k)
            a.append(S[:,s].sum(1)/n[s].sum())
        res.append(np.mean(a,0))
    return point,np.array(res)
def table(models,ids_by_task,groups,tag):
    data=build(models,ids_by_task); names=list(models); out={'n':{t:len(data[t][0]) for t in data},'groups':{}}
    for g,tasks in groups.items():
        p,B=boot(data,tasks)
        if p is None: continue
        gi={'point':dict(zip(names,p.tolist())),'ci':{n:[float(np.percentile(B[:,i],2.5)),float(np.percentile(B[:,i],97.5))] for i,n in enumerate(names)},'diff':{}}
        for i,a in enumerate(names):
            for j,b in enumerate(names):
                if i<j:
                    d=B[:,i]-B[:,j]; gi['diff'][f'{a} - {b}']={'d':float(p[i]-p[j]),'ci':[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]}
        out['groups'][g]=gi
    # per-task single
    out['tasks']={}
    for t in data:
        p,B=boot({t:data[t]},[t],B=1000)
        out['tasks'][t]={'n':len(data[t][0]) and int(data[t][1].shape[1]),'acc':dict(zip(names,p.tolist())),'ci':{n:[float(np.percentile(B[:,i],2.5)),float(np.percentile(B[:,i],97.5))] for i,n in enumerate(names)}}
    return out
GR={'image_macro(3 tasks)':IMG,'text_retention(ag_news,boolq)':TXT,'typed_decisions':[TD]}
res={}
# A) full test set, existing predictions
full={'caption_laya':'caption_laya','stock_laya':'stock_laya','stage2_a':'stage2_a','stage2_b':'stage2_b','wise080(b)':'stage2_b_wise080','stage2_c wise085':'stage2_c_wise085'}
M={k:load(R+v+'.predictions.jsonl') for k,v in full.items()}
ids={t:[i for i,(tt,_) in M['stage2_c wise085'].items() if tt==t] for t in IMG+TXT+[TD]}
# image: stock has no image preds -> exclude from image table
imgm={k:v for k,v in M.items() if k!='stock_laya'}
txtm={k:v for k,v in M.items() if k!='caption_laya'}
res['full_image']=table(imgm,{t:ids[t] for t in IMG},{k:v for k,v in GR.items() if 'image' in k},'full')
res['full_text']=table(M if False else {k:v for k,v in M.items() if k in txtm},{t:ids[t] for t in TXT+[TD]},{k:v for k,v in GR.items() if 'image' not in k},'full')
# B) common subsample, fresh reruns incl stage1
sub_ids={}
for t in IMG+TXT+[TD]:
    if os.path.exists(f'sub/{t}.jsonl'): sub_ids[t]=[json.loads(l)['id'] for l in open(f'sub/{t}.jsonl')]
RR={}
for f in sorted(glob.glob('rerun/*.json')):
    n=os.path.basename(f)[:-5]
    p=f[:-5]+'.predictions.jsonl'
    if os.path.exists(p): RR[n]=load(p)
res['rerun_models']=list(RR)
if RR:
    SM=dict(RR); SM['caption_laya']=M['caption_laya']; SM['stock_laya(text only)']=M['stock_laya']
    imgs={k:v for k,v in SM.items() if k!='stock_laya(text only)'}
    res['sub_image']=table(imgs,{t:sub_ids[t] for t in IMG},{k:v for k,v in GR.items() if 'image' in k},'sub')
    res['sub_text']=table({k:v for k,v in SM.items() if k!='caption_laya'},{t:sub_ids[t] for t in TXT+[TD]},{k:v for k,v in GR.items() if 'image' not in k},'sub')
    # reproducibility: rerun vs existing predictions
    rep={}
    for n,src in [('stage2_a_calibrated','stage2_a'),('stage2_b_calibrated','stage2_b'),('stage2_b_wise080_calibrated','stage2_b_wise080'),('stage2_c_wise085_calibrated','stage2_c_wise085')]:
        if n in RR:
            old=load(R+src+'.predictions.jsonl'); ii=[i for t in sub_ids for i in sub_ids[t] if i in RR[n] and i in old]
            rep[n]={'n':len(ii),'agree_correct':float(np.mean([RR[n][i][1]==old[i][1] for i in ii]))}
    res['reproducibility']=rep
# C) coco stage1
c={k:load(R+v+'.predictions.jsonl') for k,v in {'siglip_zs':'s1_siglip_zs','stage1_a':'stage1_a','stage1_b':'stage1_b'}.items()}
cids=[i for i in c['siglip_zs'] if i in c['stage1_a'] and i in c['stage1_b']]
cm=table(c,{'coco':cids},{'coco_choice':['coco']},'coco'); res['coco_choice']=cm
json.dump(res,open('results.json','w'),indent=1)
def show(k):
    if k not in res: return
    print('==',k,res[k]['n'])
    for g,gi in res[k]['groups'].items():
        print(' ',g)
        for n,v in gi['point'].items(): print('    %-32s %.3f [%.3f,%.3f]'%(n,v,*gi['ci'][n]))
    for t,ti in res[k]['tasks'].items(): print('  task',t,{n:round(v,3) for n,v in ti['acc'].items()})
for k in ['full_image','full_text','sub_image','sub_text','coco_choice']: show(k)
print(res.get('reproducibility'))
