import sys,json,random,math,time,os
sys.path.insert(0,'/home/zera/laya-vision')
import numpy as np, torch
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import laya_vision
torch.set_num_threads(2)
R='/home/zera/laya-vision/'; D=R+'data/'; IM=D+'images/'
ck=sys.argv[1] if len(sys.argv)>1 else R+'runs/stage2_c/wise085_calibrated'
tag=sys.argv[2] if len(sys.argv)>2 else 'final'
NS=int(os.environ.get('NS','50'));rng=random.Random(0); np.random.seed(0)
a=laya_vision.load(ck,device='cuda')
def load(f): return [json.loads(l) for l in open(D+f)]
def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/2/n)/d; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d; return [round(c-h,3),round(c+h,3)]
def acc(xs): k=sum(xs); return {'acc':round(k/len(xs),3),'ci':wilson(k,len(xs)),'n':len(xs)}
def P(state,qs):
    with torch.autocast('cuda',dtype=torch.float16):
        return a.predict(state,qs)['answers']
def choice_q(ins,opts): return {"type":"choice","instructions":ins,"criteria":opts}
def noul_q(ins): return {"type":"noul","instructions":ins}
RF='results_%s.json'%tag
res=json.load(open(RF)) if os.path.exists(RF) else {'checkpoint':ck}
def save(): json.dump(res,open(RF+'.tmp','w'),indent=1); os.replace(RF+'.tmp',RF)
#tmp='/tmp/claude-1000/-home-zera-laya-vision/2343de08-0e29-45cc-b8d5-c7b37b3b00fb/scratchpad/'; 
def sample(f,pred,n):
    rows=[r for r in load(f) if pred(r)]; rng.shuffle(rows); return rows[:n]

if 'A_choice_real' not in res:
    rng=random.Random(100); np.random.seed(0)
    # ---------- A. choice on real data
    A={}
    for f in ['oxford_pets','coco','aokvqa','scienceqa']:
        rows=sample(f+'.test.jsonl',lambda r:r['question']['type']=='choice' and r['image'],NS)
        ok=[]
        for r in rows:
            o=P({'image':IM+r['image'],**({'text':r['text']} if r['text'] else {})},{'q':r['question']})['q']
            keys=list(r['question']['criteria']); ok.append(keys[int(np.argmax(r['target']))]==o['choice'])
        A[f]=acc(ok); A[f]['chance']=round(1/len(keys),3)
    res['A_choice_real']=A;save();print('done',list(res),flush=True)


if 'B_noul' not in res:
    rng=random.Random(101); np.random.seed(0)
    # ---------- B. noul matched vs counterfactual on pets / food / cifar / coco captions
    B={}
    for f,lab in [('oxford_pets',None),('food101',None),('cifar10',None)]:
        rows=sample(f+'.test.jsonl',lambda r:r['question']['type']=='choice',NS)
        pt=[];lab_=[];
        for r in rows:
            keys=list(r['question']['criteria']); tr=keys[int(np.argmax(r['target']))]
            wrong=rng.choice([k for k in keys if k!=tr])
            for l,y in [(tr,1),(wrong,0)]:
                p=P({'image':IM+r['image']},{'q':noul_q('The image shows %s.'%l)})['q']['noul']; pt.append(p);lab_.append(y)
        pt=np.array(pt);lab_=np.array(lab_)
        B[f]={'auc':round(roc_auc_score(lab_,pt),3),'acc@0.5':acc(list((pt>.5)==(lab_==1))),'mean_p_true_matched':round(pt[lab_==1].mean(),3),'mean_p_true_counterfactual':round(pt[lab_==0].mean(),3)}
    # coco caption noul: matched vs caption of another image
    rows=sample('coco.test.jsonl',lambda r:r['question']['type']=='noul',NS)
    caps={}
    pt=[];lab_=[]
    allc=[r['fields']['caption'] for r in rows]
    for i,r in enumerate(rows):
        c=r['fields']['caption']; w=allc[(i+7)%len(rows)]
        for cap,y in [(c,1),(w,0)]:
            p=P({'image':IM+r['image']},{'q':noul_q('This caption describes the image: "%s"'%cap)})['q']['noul'];pt.append(p);lab_.append(y)
    pt=np.array(pt);lab_=np.array(lab_)
    B['coco_caption']={'auc':round(roc_auc_score(lab_,pt),3),'acc@0.5':acc(list((pt>.5)==(lab_==1))),'mean_p_true_matched':round(pt[lab_==1].mean(),3),'mean_p_true_counterfactual':round(pt[lab_==0].mean(),3)}
    # VQAv2 yes/no with hard labels
    rows=sample('vqav2_yesno.test.jsonl',lambda r:max(r['target'])>=0.9,NS*2)
    pt=[];y=[]
    for r in rows:
        pt.append(P({'image':IM+r['image']},{'q':r['question']})['q']['noul']); y.append(int(r['target'][1]>.5))
    pt=np.array(pt);y=np.array(y)
    B['vqav2_yesno']={'auc':round(roc_auc_score(y,pt),3),'acc@0.5':acc(list((pt>.5)==(y==1))),'majority_baseline':round(max(y.mean(),1-y.mean()),3)}
    res['B_noul']=B;save();print('done',list(res),flush=True)


if 'C_score_synthetic' not in res:
    rng=random.Random(102); np.random.seed(0)
    # ---------- C. score: synthetic ordinal probes on real images
    base=sample('coco.test.jsonl',lambda r:True,12); base=list({r['image']:r for r in base}.values())[:40]
    def blur(im,l): return im.filter(ImageFilter.GaussianBlur([0,2,6,14][l]))
    def bright(im,l): return ImageEnhance.Brightness(im).enhance([1.0,.55,.25,.08][l])
    def contrast(im,l): return ImageEnhance.Contrast(im).enhance([1.0,.5,.2,.05][l])
    def noise(im,l):
        x=np.asarray(im).astype(float); x=x+np.random.RandomState(l).randn(*x.shape)*[0,25,60,110][l]; return Image.fromarray(np.clip(x,0,255).astype('uint8'))
    def sat(im,l): return ImageEnhance.Color(im).enhance([1.0,.5,.15,0.0][l])
    probes={ # name: (transform, question, criteria in order of increasing "score", sign: expected increasing with level index?)
     'blur':(blur,'How blurry is this image?',['not blurry','slightly blurry','blurry','extremely blurry']),
     'darkness':(bright,'How dark is this image?',['bright','a bit dark','dark','almost black']),
     'low_contrast':(contrast,'How washed-out / low in contrast is this image?',['normal contrast','slightly washed out','washed out','flat gray']),
     'noise':(noise,'How noisy / grainy is this image?',['clean','slightly grainy','grainy','extremely noisy']),
     'desaturation':(sat,'How colorless is this image?',['full color','slightly faded','faded','black and white']),
    }
    C={}
    for name,(fn,q,crit) in probes.items():
        xs=[];ys=[];per=[];modal=[]
        for r in base:
            im=Image.open(IM+r['image']).convert('RGB').resize((256,256))
            sc=[]
            for l in range(4):
                o=P({'image':fn(im,l)},{'s':{"type":"score","instructions":q,"criteria":crit}})['s']
                sc.append(o['score']); modal.append(int(np.argmax([o['probabilities'][str(i)] for i in range(4)])))
            per.append(sc); xs+= [0,1,2,3]; ys+=sc
        per=np.array(per)
        mono=float(np.mean([spearmanr(range(4),p).correlation>0.5 for p in per if np.std(p)>0] or [0]))
        C[name]={'spearman_pooled':round(spearmanr(xs,ys).correlation,3),'mean_score_by_level':[round(float(v),3) for v in per.mean(0)],'frac_images_monotone_rho>0.5':round(mono,3),'n_images':len(per),'modal_level_hist':np.bincount(modal,minlength=4).tolist()}
    # count of dots (counting ordinal)
    def dots(n,seed):
        r=random.Random(seed); im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
        for _ in range(n):
            x,y=r.randint(20,236),r.randint(20,236); d.ellipse([x-9,y-9,x+9,y+9],fill='red')
        return im
    cnt=[1,3,8,20]; xs=[];ys=[];per=[]
    for s in range(10):
        sc=[P({'image':dots(n,s*10+n)},{'s':{"type":"score","instructions":"How many red dots are in the image?","criteria":['one','a few','several','very many']}})['s']['score'] for n in cnt]
        per.append(sc);xs+=[0,1,2,3];ys+=sc
    per=np.array(per)
    C['dot_count']={'spearman_pooled':round(spearmanr(xs,ys).correlation,3),'mean_score_by_level':[round(float(v),3) for v in per.mean(0)],'n_images':10}
    # text-only score control (the text path was trained for score)
    texts=[("The product broke on day one and support was rude.",0),("It works but nothing special.",1),("Pretty good, I would buy again.",2),("Absolutely perfect, best purchase I have ever made!",3)]
    sc=[P(t,{'s':{"type":"score","instructions":"How positive is this review?","criteria":['very negative','negative','positive','very positive']}})['s']['score'] for t,_ in texts]
    C['text_control_sentiment']={'scores':[round(s,3) for s in sc]}
    res['C_score_synthetic']=C;save();print('done',list(res),flush=True)


if 'D_mixed' not in res:
    rng=random.Random(103); np.random.seed(0)
    # ---------- D. mixed states
    Dm={}
    rows=sample('oxford_pets.test.jsonl',lambda r:r['question']['type']=='choice',NS)
    def hint(l): return 'Caption: a photo of a %s.'%l
    cond={k:[] for k in ['image_only','image+true_text','image+wrong_text','text_only_true','text_only_wrong']}
    follow_wrong=[];flip=[];follow_img_when_wrong=[]
    for r in rows:
        q=r['question']; keys=list(q['criteria']); tr=keys[int(np.argmax(r['target']))]
        wr=rng.choice([k for k in keys if k!=tr]); im=IM+r['image']
        def ch(st): return P(st,{'q':q})['q']['choice']
        cond['image_only'].append(ch({'image':im})==tr)
        cond['image+true_text'].append(ch({'image':im,'text':hint(tr)})==tr)
        w=ch({'image':im,'text':hint(wr)}); cond['image+wrong_text'].append(w==tr); follow_wrong.append(w==wr)
        cond['text_only_true'].append(ch(hint(tr))==tr); cond['text_only_wrong'].append(ch(hint(wr))==wr)
    Dm['pets_text_hint']={k:acc(v) for k,v in cond.items()}
    Dm['pets_text_hint']['image+wrong_text_follows_text']=acc(follow_wrong)
    Dm['pets_text_hint']['_note']='text_only_wrong acc = accuracy at choosing the (wrong) label stated in text; image+wrong_text acc = still picks image-true label'
    # gray image + text
    gray=Image.new('RGB',(224,224),(128,128,128)); ok=[]
    for r in rows:
        q=r['question']; keys=list(q['criteria']); tr=keys[int(np.argmax(r['target']))]
        ok.append(P({'image':gray,'text':hint(tr)},{'q':q})['q']['choice']==tr)
    Dm['pets_gray_image+true_text']=acc(ok)
    # scienceqa: text present vs removed (only rows with text)
    rows=sample('scienceqa.test.jsonl',lambda r:r['image'] and r['text'],NS); a1=[];a2=[];a3=[]
    for r in rows:
        keys=list(r['question']['criteria']);tr=keys[int(np.argmax(r['target']))]
        a1.append(P({'image':IM+r['image'],'text':r['text']},{'q':r['question']})['q']['choice']==tr)
        a2.append(P({'image':IM+r['image']},{'q':r['question']})['q']['choice']==tr)
        a3.append(P(r['text'],{'q':r['question']})['q']['choice']==tr)
    Dm['scienceqa_with_text_rows']={'image+text':acc(a1),'image_only':acc(a2),'text_only':acc(a3)}
    # OCR-like: rendered words
    words=['STOP','OPEN','EXIT','HOTEL','PIZZA','TAXI','BANK','SALE']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',56)
    def render(w,bg='white',fg='black',size=224):
        im=Image.new('RGB',(size,size),bg);d=ImageDraw.Draw(im);bb=d.textbbox((0,0),w,font=font);d.text(((size-bb[2])/2,(size-bb[3])/2),w,fill=fg,font=font);return im
    opts={'ABCDEFGH'[i]:w.lower() for i,w in enumerate(words)}
    ocr=[];ocr_t=[];ocr_conf=[];ocr_conf_follow_img=[];ocr_conf_follow_txt=[]
    for i,w in enumerate(words):
        for rep in range(3):
            ks=list(range(8)); rr=random.Random(i*10+rep); rr.shuffle(ks)
            crit={'ABCDEFGH'[j]:words[k].lower() for j,k in enumerate(ks)}
            keys=list(crit); tr=keys[[words[k] for k in ks].index(w)]
            other=words[(i+1+rep)%8]; trother=keys[[words[k] for k in ks].index(other)]
            q=choice_q('Which word is written in the image?',crit)
            im=render(w,*(['white','black'] if rep%2==0 else ['black','yellow']))
            ocr.append(P({'image':im},{'q':q})['q']['choice']==tr)
            ocr_t.append(P({'image':im,'text':'The sign reads: %s.'%w.lower()},{'q':q})['q']['choice']==tr)
            o=P({'image':im,'text':'The sign reads: %s.'%other.lower()},{'q':q})['q']['choice']
            ocr_conf_follow_img.append(o==tr); ocr_conf_follow_txt.append(o==trother)
    Dm['ocr_rendered_word_8way']={'image_only':acc(ocr),'image+agreeing_text':acc(ocr_t),'image+contradicting_text_picks_image_word':acc(ocr_conf_follow_img),'image+contradicting_text_picks_text_word':acc(ocr_conf_follow_txt),'chance':0.125}
    # text-only reading control
    res['D_mixed']=Dm;save();print('done',list(res),flush=True)


if 'E_multi_question' not in res:
    rng=random.Random(104); np.random.seed(0)
    # ---------- E. multi-question calls
    E={}
    rows=sample('oxford_pets.test.jsonl',lambda r:r['question']['type']=='choice',15)
    dmax=[];dord=[];agree=[];
    for r in rows:
        q=r['question'];keys=list(q['criteria']);tr=keys[int(np.argmax(r['target']))]
        qs={'c':q,'n_true':noul_q('The image shows %s.'%tr),'n_false':noul_q('The image shows %s.'%rng.choice([k for k in keys if k!=tr])),
            's':{"type":"score","instructions":"How blurry is this image?","criteria":['not blurry','blurry','very blurry']},
            'n_animal':noul_q('Is there an animal in the image?')}
        st={'image':IM+r['image']}
        full=P(st,qs)
        d=0;same=0
        for k,v in qs.items():
            s=P(st,{k:v})[k]
            if v['type']=='choice': d=max(d,max(abs(s['probabilities'][o]-full[k]['probabilities'][o]) for o in s['probabilities'])); same+=s['choice']==full[k]['choice']
            elif v['type']=='noul': d=max(d,abs(s['noul']-full[k]['noul']))
            else: d=max(d,max(abs(s['probabilities'][o]-full[k]['probabilities'][o]) for o in s['probabilities']))
        dmax.append(d)
        rev=P(st,dict(reversed(list(qs.items()))))
        dord.append(max(abs(rev['n_true']['noul']-full['n_true']['noul']),abs(rev['c']['probabilities'][keys[0]]-full['c']['probabilities'][keys[0]])))
        agree.append(full['c']['choice']==tr)
    E['n_images']=len(rows)
    E['max_abs_prob_diff_single_vs_batched_call']={'mean':round(float(np.mean(dmax)),4),'median':round(float(np.median(dmax)),4),'max':round(float(np.max(dmax)),4)}
    E['max_abs_diff_question_order_reversed']={'mean':round(float(np.mean(dord)),4),'max':round(float(np.max(dord)),4)}
    E['choice_acc_inside_5q_call']=acc(agree)
    res['E_multi_question']=E
    save()
    print(json.dumps(res,indent=1))

