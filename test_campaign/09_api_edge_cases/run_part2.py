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
# [test fix] enc()/base_im were used below but never defined in this script (NameError in 7 checks)
base_im=Image.open(IMG).convert("RGB")
def enc(im,fmt,**kw):
    b=io.BytesIO(); im.save(b,fmt,**kw); return b.getvalue()
base=a.predict(S,Q())
bp=pvec(base)


# diagnostics on max_len handling
def diag():
    out=[]
    try: out.append("a._fast=%r _compiled=%r"%(a._fast,a._compiled))
    except Exception as e: out.append(repr(e))
    for ml in (64,40,30):
        try:
            r=a.predict_batch([S],Q(),max_len=ml); out.append("batch max_len=%d -> ok %s"%(ml,pvec(r[0])))
        except Exception as e: out.append("batch max_len=%d -> %s: %s"%(ml,type(e).__name__,e))
    try:
        it=a._encode_state(S,["q"],{"q":__import__("laya_vision.sequence",fromlist=["x"]).to_internal(Q()["q"])},max_len=64)
        out.append("_encode_state direct max_len=64 -> len ids %d"%len(it[0]["ids"]))
    except Exception as e: out.append("_encode_state direct max_len=64 -> %s: %s"%(type(e).__name__,e))
    try: a.accelerate(strict=True); out.append("accelerate ok")
    except Exception as e: out.append("accelerate(strict) -> %s: %s"%(type(e).__name__,str(e)[:200]))
    return True,"; ".join(out)
run("DIAG max_len handling / fast availability",diag)
run("min_confidence=1.5",lambda:a.predict(S,Q(),min_confidence=1.5),"err")
import uvicorn
from laya_vision.ui import create_ui_app
import requests
app=create_ui_app(agent=a,checkpoint=CK)
PORT=8795
cfg=uvicorn.Config(app,host="127.0.0.1",port=PORT,log_level="warning")
srv=uvicorn.Server(cfg); th=threading.Thread(target=srv.run,daemon=True); th.start()
for _ in range(50):
    try: requests.get("http://127.0.0.1:%d/health"%PORT,timeout=1); break
    except Exception: time.sleep(0.2)
U="http://127.0.0.1:%d"%PORT
def H(name,fn,want=None):
    try:
        r=fn()
    except Exception as e:
        rec("HTTP "+name,"FAIL","exception %r"%e); return None
    ok = (r.status_code in want) if want else r.status_code==200
    rec("HTTP "+name,"PASS" if ok else "FAIL","status=%s body=%s"%(r.status_code,r.text[:240]))
    return r
raw=open(IMG,"rb").read(); b64=base64.b64encode(raw).decode()
def post(body,**kw): return requests.post(U+"/v1/systemone",json=body,timeout=120,**kw)
H("GET /health",lambda:requests.get(U+"/health"))
r=H("GET / (web UI html)",lambda:requests.get(U+"/")); 
if r is not None: rec("HTTP / content is html w/ form","PASS" if "<html" in r.text.lower() and "systemone" in r.text else "FAIL","len=%d"%len(r.text))
H("GET /docs",lambda:requests.get(U+"/docs"))
H("GET /openapi.json",lambda:requests.get(U+"/openapi.json"))
H("GET /nonexistent -> 404",lambda:requests.get(U+"/nope"),(404,))
H("GET /v1/systemone -> 405",lambda:requests.get(U+"/v1/systemone"),(405,))
r=H("POST image b64 raw",lambda:post({"state":{"image":b64},"questions":Q()}))
if r is not None and r.status_code==200:
    d=max(abs(x-y) for x,y in zip(list(r.json()["answers"]["q"]["probabilities"].values()),bp)); rec("HTTP result equals in-process","PASS" if d<=2e-3 else "FAIL","d=%g; headers Server-Timing=%s"%(d,r.headers.get("Server-Timing")))
H("POST image dataurl+text",lambda:post({"state":{"image":"data:image/jpeg;base64,"+b64,"text":"a pet"},"questions":Q()}))
H("POST image PNG",lambda:post({"state":{"image":base64.b64encode(enc(base_im,"PNG")).decode()},"questions":Q()}))
H("POST image BMP (server allows only JPEG/PNG/WEBP/GIF) expect 400",lambda:post({"state":{"image":base64.b64encode(enc(base_im,"BMP")).decode()},"questions":Q()}),(400,))
H("POST image TIFF expect 400",lambda:post({"state":{"image":base64.b64encode(enc(base_im,"TIFF")).decode()},"questions":Q()}),(400,))
H("POST image RGBA PNG",lambda:post({"state":{"image":base64.b64encode(enc(base_im.convert("RGBA"),"PNG")).decode()},"questions":Q()}))
H("POST image CMYK JPEG",lambda:post({"state":{"image":base64.b64encode(enc(base_im.convert("CMYK"),"JPEG")).decode()},"questions":Q()}))
H("POST image 1x1",lambda:post({"state":{"image":base64.b64encode(enc(Image.new("RGB",(1,1)),"PNG")).decode()},"questions":Q()}))
H("POST path as image (must NOT read file) expect 400",lambda:post({"state":{"image":IMG},"questions":Q()}),(400,))
H("POST invalid base64 expect 400",lambda:post({"state":{"image":"@@@"},"questions":Q()}),(400,))
H("POST corrupt image expect 400",lambda:post({"state":{"image":base64.b64encode(raw[:3000]+os.urandom(500)).decode()},"questions":Q()}),(400,))
H("POST truncated image expect 400",lambda:post({"state":{"image":base64.b64encode(raw[:len(raw)//2]).decode()},"questions":Q()}),(400,))
H("POST empty image expect 400",lambda:post({"state":{"image":""},"questions":Q()}),(400,))
H("POST image list expect 400",lambda:post({"state":{"image":[b64]},"questions":Q()}),(400,))
H("POST image extra key expect 400",lambda:post({"state":{"image":b64,"x":1},"questions":Q()}),(400,))
H("POST 7000x7000 PNG expect 413",lambda:post({"state":{"image":base64.b64encode(enc(Image.new("RGB",(7000,7000)),"PNG")).decode()},"questions":Q()}),(413,))
H("POST >10MB image expect 413",lambda:post({"state":{"image":base64.b64encode(os.urandom(11*1024*1024)).decode()},"questions":Q()}),(413,))
H("POST body >16MB expect 413",lambda:requests.post(U+"/v1/systemone",data=b'{"a":"'+b"x"*(17*1024*1024)+b'"}',headers={"Content-Type":"application/json"},timeout=120),(413,))
H("POST invalid JSON expect 400",lambda:requests.post(U+"/v1/systemone",data=b"{bad",headers={"Content-Type":"application/json"}),(400,))
H("POST JSON array expect 400",lambda:post([1,2]),(400,))
H("POST missing questions expect 400",lambda:post({"state":{"image":b64}}),(400,))
H("POST missing state, questions only",lambda:post({"questions":Q()}),(200,400,422))
H("POST questions empty",lambda:post({"state":{"image":b64},"questions":{}}),(200,400,422))
H("POST questions not dict expect 4xx",lambda:post({"state":{"image":b64},"questions":[1]}),(400,422))
H("POST bad question type expect 4xx",lambda:post({"state":{"image":b64},"questions":{"q":{"type":"zzz","instructions":"?","criteria":{"a":None,"b":None}}}}),(400,422))
H("POST 20 options",lambda:post({"state":{"image":b64},"questions":Q(tuple("o%d"%i for i in range(20)))}))
H("POST 60 options (limit?)",lambda:post({"state":{"image":b64},"questions":Q(tuple("o%d"%i for i in range(60)))}),(200,400,413,422))
# [test fix] max_len=64 fits exactly (63 tokens) -> 200; 40 does not -> 422
H("POST max_len=64 (fits exactly) expect 200",lambda:post({"state":{"image":b64},"questions":Q(),"max_len":64}),(200,))
H("POST image does not fit (max_len=40) expect 422",lambda:post({"state":{"image":b64},"questions":Q(),"max_len":40}),(422,))
H("POST max_len string expect 400",lambda:post({"state":{"image":b64},"questions":Q(),"max_len":"a"}),(400,422))
H("POST max_len=10**9 expect 4xx",lambda:post({"state":{"image":b64},"questions":Q(),"max_len":10**9}),(400,422))
H("POST unicode options",lambda:post({"state":{"image":b64},"questions":Q(("猫","犬","🐱"))}))
H("POST empty-string option",lambda:post({"state":{"image":b64},"questions":Q(("","b"))}),(200,400,422))
H("POST 100 questions",lambda:post({"state":{"image":b64},"questions":{"q%d"%i:Q()["q"] for i in range(100)}}),(200,400,413,422))
H("POST text-only state",lambda:post({"state":"hello","questions":Q()}))
H("POST text-only dict state",lambda:post({"state":{"a":1},"questions":Q()}))
H("POST text-only > 2MiB expect 413",lambda:post({"state":"x"*(2*1024*1024+10),"questions":Q()}),(413,))
H("POST image + text huge expect 4xx",lambda:post({"state":{"image":b64,"text":"x"*300000},"questions":Q()}),(200,400,413,422))
H("POST wrong content-type",lambda:requests.post(U+"/v1/systemone",data=json.dumps({"state":"hi","questions":Q()}),headers={"Content-Type":"text/plain"}),(200,400,415))
H("POST min_confidence field ignored?",lambda:post({"state":{"image":b64},"questions":Q(),"min_confidence":0.99}),(200,400,422))
# concurrency
def conc():
    res=[]
    def w(): 
        try: res.append(post({"state":{"image":b64},"questions":Q()}).status_code)
        except Exception as e: res.append(repr(e))
    ts=[threading.Thread(target=w) for _ in range(12)]
    [t.start() for t in ts]; [t.join() for t in ts]
    return res
res=conc(); rec("HTTP 12 concurrent requests","PASS" if all(x in (200,503) for x in res) else "FAIL",str({x:res.count(x) for x in set(res)}))
H("health after load",lambda:requests.get(U+"/health"))
exec(open("post_fix_http.py").read())
# CORS / UI api key
srv.should_exit=True; th.join(5)
json.dump(results,open("results_part2.json","w"),indent=1)
print("TOTAL",len(results),"PASS",sum(r["status"]=="PASS" for r in results),"FAIL",sum(r["status"]=="FAIL" for r in results))
