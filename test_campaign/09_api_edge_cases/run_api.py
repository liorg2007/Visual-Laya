import io, os, json, math, base64, copy, threading, time, traceback, hashlib, sys
import numpy as np
from PIL import Image
import torch, laya_vision
R="/home/zera/laya-vision"
CK=R+"/runs/stage2_c/wise085_calibrated"
IMG=R+"/data/images/oxford_pets/basset_hound_129.jpg"
IMG2=R+"/data/images/cifar10/train-70.png"
a=laya_vision.load(CK,device="cuda")
results=[]
def rec(name, status, detail=""):
    results.append({"test":name,"status":status,"detail":str(detail)[:600]}); print(status, name, str(detail)[:300], flush=True)
def Q(opts=("cat","dog"), t="choice", ins="What animal?"):
    return {"q":{"type":t,"instructions":ins,"criteria":{o:None for o in opts}}}
def run(name, fn, expect="ok"):
    """expect: 'ok' -> must succeed (fn returns (bool,detail) or None); 'err' -> must raise ValueError-ish; returns value"""
    try:
        v=fn()
    except Exception as e:
        msg="%s: %s"%(type(e).__name__, e)
        if expect=="err": rec(name,"PASS","raised "+msg); 
        else: rec(name,"FAIL","unexpected "+msg)
        return None
    if expect=="err": rec(name,"FAIL","no error; result=%s"%str(v)[:200]); return v
    if isinstance(v,tuple) and len(v)==2 and isinstance(v[0],bool):
        rec(name,"PASS" if v[0] else "FAIL",v[1])
    else: rec(name,"PASS",str(v)[:200])
    return v
def ans(r,k="q"): return r["answers"][k]
def probs(r,k="q"): return ans(r,k)["probabilities"]
def pvec(r,k="q"): return list(probs(r,k).values())
S={"image":IMG}
base=a.predict(S,Q())
bp=pvec(base)

# ---- determinism
def det():
    rs=[pvec(a.predict(S,Q())) for _ in range(5)]
    d=max(max(abs(x-y) for x,y in zip(r,bp)) for r in rs)
    return d==0, "max |dp| over 5 repeats = %g"%d
run("determinism: 5 repeated predict()",det)
def bat():
    rs=a.predict_batch([S]*3,Q()); d=max(abs(x-y) for r in rs for x,y in zip(pvec(r),bp))
    return d<=2e-3, "predict_batch x3 vs single: max|dp|=%g (rounded 4dp output)"%d
run("determinism: predict_batch same image x3 vs predict",bat)
def bat2():
    S2={"image":IMG2}; sing2=pvec(a.predict(S2,Q(("cat","dog","ship"))))
    rs=a.predict_batch([S,S2,"a plain text state",S2,S],Q(("cat","dog","ship")))
    sing1=pvec(a.predict(S,Q(("cat","dog","ship"))))
    d=max(max(abs(x-y) for x,y in zip(pvec(rs[0]),sing1)),max(abs(x-y) for x,y in zip(pvec(rs[1]),sing2)),max(abs(x-y) for x,y in zip(pvec(rs[4]),sing1)))
    return d<=2e-3, "mixed batch (img,img2,text,img2,img) vs singles: max|dp|=%g"%d
run("batch composition invariance (mixed image/text)",bat2)
def bs():
    st=[{"image":IMG},{"image":IMG2},{"image":IMG,"text":"a dog"},{"image":IMG2,"text":"photo"}]
    q=Q(("cat","dog","ship"))
    ref=[pvec(a.predict(s,q)) for s in st]
    out=[]
    for b in (1,2,3,8):
        rs=a.predict_batch(st,q,batch_size=b); out.append(max(abs(x-y) for r,rf in zip(rs,ref) for x,y in zip(pvec(r),rf)))
    return max(out)<=2e-3, "batch_size 1,2,3,8 dev=%s"%out
run("batch_size invariance",bs)
def sbl():
    st=[{"image":IMG,"text":"x "*i} for i in (1,30,5)]
    q=Q(); r1=a.predict_batch(st,q); r2=a.predict_batch(st,q,sort_by_length=True)
    d=max(abs(x-y) for u,v in zip(r1,r2) for x,y in zip(pvec(u),pvec(v)))
    return d<=2e-3 and len(r2)==3, "sort_by_length order/devs=%g"%d
run("sort_by_length keeps order",sbl)
def multiq_inv():
    qs={"a":Q()["q"],"b":Q(("x","y","z"),ins="Other?")["q"]}
    r=a.predict(S,qs); ra=a.predict(S,{"a":qs["a"]}); 
    d=max(abs(x-y) for x,y in zip(pvec(r,"a"),pvec(ra,"a")))
    return d<=2e-3,"question invariance to co-questions d=%g"%d
run("multi-question vs single invariance",multiq_inv)
# hash dedupe: same image different forms
def forms():
    im=Image.open(IMG); raw=open(IMG,"rb").read()
    outs={}
    outs["path"]=pvec(a.predict({"image":IMG},Q()))
    outs["pathlib"]=pvec(a.predict({"image":__import__("pathlib").Path(IMG)},Q()))
    outs["bytes"]=pvec(a.predict({"image":raw},Q()))
    outs["bytearray"]=pvec(a.predict({"image":bytearray(raw)},Q()))
    outs["b64"]=pvec(a.predict({"image":base64.b64encode(raw).decode()},Q()))
    outs["dataurl"]=pvec(a.predict({"image":"data:image/jpeg;base64,"+base64.b64encode(raw).decode()},Q()))
    outs["PIL"]=pvec(a.predict({"image":im},Q()))
    outs["ndarray"]=pvec(a.predict({"image":np.asarray(im.convert("RGB"))},Q()))
    d={k:max(abs(x-y) for x,y in zip(v,bp)) for k,v in outs.items()}
    return max(d.values())<=2e-3, str(d)
run("input forms path/Path/bytes/bytearray/b64/dataurl/PIL/ndarray agree",forms)
def one(name,val,expect="ok"):
    return run(name,lambda:(lambda r:(math.isclose(sum(pvec(r)),1,abs_tol=2e-3), "probs=%s"%probs(r)))(a.predict({"image":val},Q())),expect)
# ---- formats
base_im=Image.open(IMG).convert("RGB")
def enc(im,fmt,**kw):
    b=io.BytesIO(); im.save(b,fmt,**kw); return b.getvalue()
one("fmt PNG bytes",enc(base_im,"PNG"))
one("fmt JPEG bytes",enc(base_im,"JPEG"))
one("fmt WEBP bytes",enc(base_im,"WEBP"))
one("fmt GIF bytes",enc(base_im.convert("P"),"GIF"))
one("fmt BMP bytes",enc(base_im,"BMP"))
one("fmt TIFF bytes (expect reject: not allowed)",enc(base_im,"TIFF"),"err")
one("mode RGBA PIL",base_im.convert("RGBA"))
rg=base_im.convert("RGBA"); rg.putalpha(0)
one("mode RGBA fully transparent PIL",rg)
one("mode RGBA PNG bytes",enc(base_im.convert("RGBA"),"PNG"))
one("mode L grayscale PIL",base_im.convert("L"))
one("mode L PNG bytes",enc(base_im.convert("L"),"PNG"))
one("mode LA PNG bytes",enc(base_im.convert("LA"),"PNG"))
one("mode P palette PNG bytes",enc(base_im.convert("P"),"PNG"))
pt=base_im.convert("P"); pt.info["transparency"]=0
one("mode P + transparency PNG bytes",enc(pt,"PNG",transparency=0))
one("mode CMYK JPEG bytes",enc(base_im.convert("CMYK"),"JPEG"))
one("mode CMYK PIL",base_im.convert("CMYK"))
one("mode 1 (bilevel) PIL",base_im.convert("1"))
one("mode I;16 PNG bytes",enc(Image.fromarray((np.asarray(base_im.convert("L")).astype(np.uint16)*257)),"PNG"))
one("mode I PIL",base_im.convert("L").convert("I"))
one("mode F PIL",base_im.convert("L").convert("F"))
one("mode HSV PIL",base_im.convert("HSV"))
one("mode YCbCr PIL",base_im.convert("YCbCr"))
one("tiny 1x1 PIL",Image.new("RGB",(1,1),(255,0,0)))
one("tiny 1x1 PNG bytes",enc(Image.new("RGB",(1,1),(255,0,0)),"PNG"))
one("1xN 1x1000 PIL",Image.new("RGB",(1,1000),(10,200,10)))
one("non-square 4000x100",base_im.resize((4000,100)))
one("non-square 100x4000",base_im.resize((100,4000)))
one("big 6000x6000 (36MP) PNG bytes",enc(Image.new("RGB",(6000,6000),(120,60,20)),"PNG"))
one("over limit 7000x7000 (49MP) PIL (expect reject)",Image.new("RGB",(7000,7000)),"err")
one("over limit 7000x7000 PNG bytes (expect reject)",enc(Image.new("RGB",(7000,7000)),"PNG"),"err")
rn=os.urandom(11*1024*1024)
one("over 10MB bytes (expect reject)",rn,"err")
big=Image.fromarray(np.random.default_rng(0).integers(0,255,(2000,2000,3),dtype=np.uint8)); bb=enc(big,"PNG")
one("random-noise 2000x2000 PNG (%.1f MB; reject iff >10MB)"%(len(bb)/1e6),bb,"err" if len(bb)>10*1024*1024 else "ok")
# animated gif
fr=[Image.new("RGB",(64,64),c) for c in [(255,0,0),(0,255,0)]]
b=io.BytesIO(); fr[0].save(b,"GIF",save_all=True,append_images=fr[1:]); one("animated GIF (frame 0)",b.getvalue())
b=io.BytesIO(); fr[0].save(b,"WEBP",save_all=True,append_images=fr[1:]); one("animated WEBP (frame 0)",b.getvalue())
# ndarray variants
arr=np.asarray(base_im)
one("ndarray uint8 HxWx3",arr)
one("ndarray float [0,1]",arr/255.0)
one("ndarray HxW gray",arr[...,0])
one("ndarray CHW",np.moveaxis(arr,2,0))
one("ndarray HxWx4",np.concatenate([arr,np.full(arr.shape[:2]+(1,),255,np.uint8)],2))
one("ndarray bad shape 5D (expect reject)",np.zeros((2,2,2,2,2)),"err")
one("ndarray empty 0x0 (expect reject)",np.zeros((0,0,3),np.uint8),"err")
# invalid
one("missing path (expect error)","/nonexistent/img.jpg","err")
one("directory path (expect error)","/tmp","err")
c=open(IMG,"rb").read()
one("corrupt JPEG: truncated half",c[:len(c)//2],"err")
one("corrupt: random bytes",os.urandom(500),"err")
one("corrupt: JPEG header then garbage",c[:100]+os.urandom(2000),"err")
one("empty bytes (expect error)",b"","err")
one("empty string (expect error)","","err")
one("None image (expect error)",None,"err")
one("int image (expect error)",5,"err")
one("text file as path (expect error)",__file__,"err")
one("SVG bytes (expect reject)",b'<svg xmlns="http://www.w3.org/2000/svg"/>',"err")
one("invalid base64 string (expect error)","!!!notb64***","err")
one("data URL non-base64 (expect error)","data:image/png,abc","err")
one("list of images (expect error v1)",[IMG,IMG],"err")
# state structure
def st(name,state,expect="ok",q=None):
    return run(name,lambda:(lambda r:(math.isclose(sum(pvec(r)),1,abs_tol=2e-3),"probs=%s"%probs(r)))(a.predict(state,q or Q())),expect)
st("state: extra key (expect error)",{"image":IMG,"foo":1},"err")
st("state: image+text str",{"image":IMG,"text":"A photo of a pet."})
st("state: image+text empty str",{"image":IMG,"text":""})
st("state: image+text None",{"image":IMG,"text":None})
st("state: image+text dict",{"image":IMG,"text":{"a":1}})
st("state: image+text list (conversation)",{"image":IMG,"text":["user: hi","bot: hello"]})
st("state: image+very long text 20k words",{"image":IMG,"text":"word "*20000})
st("state: image+unicode text",{"image":IMG,"text":"写真 🐶 مرحبا ñ \u0000 ퟿"})
st("state: image+text containing [MASK]",{"image":IMG,"text":"hello [MASK] [SEP] [PAD]"})
st("state: no image text-only str","hello world")
st("state: text-only dict",{"text":"hello"})
st("state: empty string","")
st("state: None (expect error?)",None)
st("state: image=None dict (expect error)",{"image":None},"err")
# ---- options
def qs(name,opts,expect="ok",t="choice"):
    q={"q":{"type":t,"instructions":"Which?","criteria":opts}}
    return st(name,S,expect,q)
qs("opts: 2 options",{"a":None,"b":None})
qs("opts: 1 option choice (expect error?)",{"a":None},"err")
qs("opts: 10 options",{str(i):None for i in range(10)})
qs("opts: 11 options",{str(i):"opt %d"%i for i in range(11)})
qs("opts: 20 options",{str(i):None for i in range(20)})
qs("opts: 21 options",{str(i):None for i in range(21)})
qs("opts: 26 options",{chr(65+i):None for i in range(26)})
qs("opts: 50 options",{"label%d"%i:"desc %d"%i for i in range(50)})
qs("opts: 200 options (expect error)",{"label%d"%i:"desc %d"%i for i in range(200)},"err")
qs("opts: empty dict (expect error)",{},"err")
qs("opts: empty-string label",{"":None,"b":None})
qs("opts: unicode labels",{"猫":None,"犬":None,"🐱":"emoji cat"})
qs("opts: RTL labels",{"قطة":None,"كلب":None})
qs("opts: very long label (5000 chars)",{"x"*5000:None,"b":None})
qs("opts: very long descriptions",{"a":"long "*2000,"b":"other "*2000})
qs("opts: whitespace-only label",{" ":None,"b":None})
qs("opts: label w/ [MASK] [SEP]",{"[MASK]":None,"[SEP]":None})
qs("opts: label w/ newline/null",{"a\nb":None,"c\x00d":None})
qs("opts: duplicate-like labels differing by case",{"Cat":None,"cat":None})
qs("opts: non-string label key int (expect error?)",{1:None,2:None},"err")
qs("opts: long descriptions -> image does not fit",{"o%d"%i:"very long description of the option "*20 for i in range(5)},"err")
# noul / other types
for t in ("noul","choice","multi","action"):
    pass
# [test fix] noul answers carry 'noul' (a probability), not 'probabilities'
r=run("noul 2 options",lambda:(lambda r:(0<=ans(r)["noul"]<=1,ans(r)))(a.predict(S,{"q":{"type":"noul","instructions":"Is it a dog?","criteria":{"false":"no","true":"yes"}}})))
run("noul with 3 options (expect error)",lambda:a.predict(S,{"q":{"type":"noul","instructions":"?","criteria":{"a":"1","b":"2","c":"3"}}}),"err")
run("noul w/o criteria (expect error?)",lambda:a.predict(S,{"q":{"type":"noul","instructions":"Is it a dog?"}}),"err")
run("unknown qtype (expect error)",lambda:a.predict(S,{"q":{"type":"bogus","instructions":"?","criteria":{"a":None,"b":None}}}),"err")
run("missing instructions (expect error?)",lambda:a.predict(S,{"q":{"type":"choice","criteria":{"a":None,"b":None}}}),"err")
run("empty instructions",lambda:a.predict(S,{"q":{"type":"choice","instructions":"","criteria":{"a":None,"b":None}}}) and True)
run("very long instructions 20k chars (expect error or truncation)",lambda:a.predict(S,{"q":{"type":"choice","instructions":"why "*5000,"criteria":{"a":None,"b":None}}}),"err")
run("unicode instructions",lambda:(lambda r:(True,probs(r)))(a.predict(S,{"q":{"type":"choice","instructions":"这是什么动物？🐕","criteria":{"猫":None,"狗":None}}})))
run("questions: empty dict",lambda:a.predict(S,{}))
run("questions: not a dict (expect error)",lambda:a.predict(S,[Q()["q"]]),"err")
run("questions: None (expect error)",lambda:a.predict(S,None),"err")
# many questions
def many(n):
    qs_={"q%d"%i:Q(("cat","dog","bird")[: 2+i%2],ins="Question number %d about the animal?"%i)["q"] for i in range(n)}
    t=time.time(); r=a.predict(S,qs_); 
    ok=len(r["answers"])==n and all(math.isclose(sum(v["probabilities"].values()),1,abs_tol=2e-3) for v in r["answers"].values())
    return ok,"n=%d answers=%d %.2fs usage=%s"%(n,len(r["answers"]),time.time()-t,r["usage"])
for n in (5,20,64,200): run("many questions per call n=%d"%n,lambda n=n:many(n))
run("many questions (text-only) n=100",lambda:(lambda r:(len(r["answers"])==100,"ok"))(a.predict("hello",{"q%d"%i:Q()["q"] for i in range(100)})))
run("many states predict_batch n=64 (same image) ",lambda:(lambda rs:(len(rs)==64 and max(abs(x-y) for r in rs for x,y in zip(pvec(r),bp))<=2e-3,"ok"))(a.predict_batch([S]*64,Q())))
run("predict_batch empty list",lambda:a.predict_batch([],Q()))
run("predict_batch 1 text + 1 img mixed",lambda:(lambda rs:(len(rs)==2,"ok"))(a.predict_batch(["hi",S],Q())))
# unsupported modes
run("predict_long with image (expect error)",lambda:a.predict_long(S,Q()),"err")
def shortlist():
    import laya.shortlist as sl
    return sl.predict_shortlist(a,S,Q())
run("predict_shortlist with image (expect clear error)",shortlist,"err")
# min_confidence
def mc():
    r0=a.predict(S,Q()); c=ans(r0)["confidence"]
    r_hi=a.predict(S,Q(),min_confidence=0.999); r_lo=a.predict(S,Q(),min_confidence=0.0)
    return True,"conf=%s; min_conf=.999 -> %s; 0.0 -> %s"%(c,json.dumps(ans(r_hi))[:260],json.dumps(ans(r_lo))[:200])
run("min_confidence behavior (info)",mc)
def mcb():
    rs=a.predict_batch([S,{"image":IMG2}],Q(),min_confidence=0.5); return True,json.dumps([ans(r) for r in rs])[:400]
run("min_confidence in predict_batch",mcb)
run("min_confidence=1.5 (expect error)",lambda:a.predict(S,Q(),min_confidence=1.5),"err")
run("min_confidence=-1 (expect error)",lambda:a.predict(S,Q(),min_confidence=-1),"err")
run("min_confidence='x' (expect error)",lambda:a.predict(S,Q(),min_confidence="x"),"err")
# probability sums
def psum():
    bad=[]
    import random; random.seed(0)
    for n in (2,3,5,8,12,20):
        r=a.predict(S,Q(tuple("o%d"%i for i in range(n))))
        s=sum(pvec(r)); 
        if abs(s-1)>2e-3 or any(p<0 or p>1 or math.isnan(p) for p in pvec(r)): bad.append((n,s))
    return not bad, "bad=%s"%bad
run("probabilities sum to 1 / in [0,1] across n options",psum)
# max_len / head_max_len
# [test fix] max_len=64 fits exactly (head + 49 image tokens = 63); 40 does not
run("max_len=64 fits exactly (63 tokens)",lambda:(lambda r:(r["usage"]["input_tokens"]<=64,"usage=%s"%r["usage"]))(a.predict(S,Q(),max_len=64)))
run("max_len=40 -> image does not fit (expect clear error)",lambda:a.predict(S,Q(),max_len=40),"err")
run("max_len=80 borderline",lambda:a.predict(S,Q(),max_len=80) and True)
run("max_len=1024 > cfg",lambda:a.predict(S,Q(),max_len=1024) and True)
run("max_len=0 (expect error)",lambda:a.predict(S,Q(),max_len=0),"err")
run("max_len=-5 (expect error)",lambda:a.predict(S,Q(),max_len=-5),"err")
run("head_max_len=8 (expect error)",lambda:a.predict(S,Q(),head_max_len=8),"err")
run("max_len=100000 (expect error or ok)",lambda:a.predict(S,Q(),max_len=100000),"err")
# batch error isolation
def iso():
    return a.predict_batch([S,{"image":"/nonexistent.jpg"},S],Q())
run("predict_batch with one bad image (expect error, whole batch)",iso,"err")
# lang
run("lang='fr' with image",lambda:a.predict(S,Q(),lang="fr") and True)
# hooks
run("hooks on_predict_start/end with image",lambda:(lambda log:(a.predict(S,Q(),on_predict_start=lambda *x,**k:log.append("s"),on_predict_end=lambda *x,**k:log.append("e")),(log==["s","e"],str(log)))[1])([]))
# thread safety
def thr():
    errs=[];outs=[]
    def w():
        try: outs.append(pvec(a.predict(S,Q())))
        except Exception as e: errs.append(repr(e))
    ts=[threading.Thread(target=w) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    d=max([max(abs(x-y) for x,y in zip(o,bp)) for o in outs] or [9])
    return (not errs and d<=2e-3),"errs=%s d=%g"%(errs[:2],d)
run("concurrent predict() from 6 threads",thr)
# text-only matches stock
run("text-only state agrees with stock Laya path (super)",lambda:(lambda r:(True,json.dumps(r)[:200]))(a.predict("The match ended 2-1.",{"q":{"type":"choice","instructions":"Topic?","criteria":{"sports":None,"world":None}}})))
# modes
import laya
for kw in ({"fast":True},{"compile":True}):
    def ld(kw=kw):
        x=laya_vision.load(CK,device="cuda",**kw)
        try:
            return x.predict(S,Q())
        finally:
            del x; torch.cuda.empty_cache()
    run("load(%s) then image predict (expect clear error)"%kw,ld,"err")
run("load nonexistent path (expect FileNotFoundError)",lambda:laya_vision.load("/no/such/dir"),"err")
run("load text-only Laya dir w/o vision block",lambda:laya_vision.load(R+"/runs/stage2_c") ,"err")
# [script change] the HTTP section that used to follow is identical to run_part2.py, which runs it.
exec(open("post_fix_checks.py").read())
json.dump(results,open("results_api.json","w"),indent=1)
print("TOTAL",len(results),"PASS",sum(r["status"]=="PASS" for r in results),"FAIL",sum(r["status"]=="FAIL" for r in results))
