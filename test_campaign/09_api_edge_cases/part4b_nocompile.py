# Remainder of part4 (compile=True text-only predict timed out at 600s in part4; not re-run) + surrogate repro at Python level
import json, torch, laya_vision
R="/home/zera/laya-vision"; CK=R+"/runs/stage2_c/wise085_calibrated"; IMG=R+"/data/images/oxford_pets/basset_hound_129.jpg"
results=[]
def rec(n,s,d): results.append({"test":n,"status":s,"detail":str(d)[:600]}); print(s,n,str(d)[:400],flush=True)
Q={"q":{"type":"choice","instructions":"What animal?","criteria":{"cat":None,"dog":None}}}
S={"image":IMG}
a=laya_vision.load(CK,device="cuda")
a._fast=object()
try: a.predict(S,Q); rec("simulated _fast set + image (expect clear ValueError)","FAIL","no error")
except Exception as e: rec("simulated _fast set + image (expect clear ValueError)","PASS" if isinstance(e,ValueError) else "FAIL","%s: %s"%(type(e).__name__,e))
a._fast=None
try: a.accelerate(strict=True); rec("accelerate(strict=True)","INFO","ok")
except Exception as e: rec("accelerate(strict=True) without tilelang","INFO","%s: %s"%(type(e).__name__,str(e)[:200]))
# [post-fix] strict: PASS only for ValueError naming "surrogate"; image-state text added.
def sur(name,fn):
    try: fn(); rec("Python API "+name+" (expect ValueError)","FAIL","no error")
    except Exception as e: rec("Python API "+name+" (expect ValueError)","PASS" if isinstance(e,ValueError) and "surrogate" in str(e) else "FAIL","%s: %s"%(type(e).__name__,str(e)[:200]))
sur("surrogate in option",lambda:a.predict(S,{"q":{"type":"choice","instructions":"?","criteria":{"a\ud800":None,"b":None}}}))
sur("surrogate in instructions",lambda:a.predict(S,{"q":{"type":"choice","instructions":"x\ud800","criteria":{"a":None,"b":None}}}))
sur("surrogate in state text",lambda:a.predict({"text":"hi\ud800"},Q))
sur("surrogate in image-state text",lambda:a.predict({"image":IMG,"text":"hi\ud800"},Q))
# (predict_shortlist check dropped: it needs an embed_fn, which this test cannot supply meaningfully)
sur("text-only predict_long surrogate",lambda:a.predict_long("x\ud800",Q))
json.dump(results,open("results_part4b.json","w"),indent=1)
