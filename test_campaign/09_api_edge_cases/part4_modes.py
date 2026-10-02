# Part 4: execution modes. Post-fix rerun: the compile=True text-only predict is timed (load, 1st call,
# 2nd call) with a generous budget; faulthandler dumps stacks every 120 s so a hang can be told apart
# from a slow inductor compile.
import faulthandler, json, sys, time, torch, laya_vision
faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
R="/home/zera/laya-vision"; CK=R+"/runs/stage2_c/wise085_calibrated"; IMG=R+"/data/images/oxford_pets/basset_hound_129.jpg"
results=[]
def rec(n,s,d):
    results.append({"test":n,"status":s,"detail":str(d)[:600]}); print(s,n,str(d)[:400],flush=True)
    json.dump(results,open("results_part4.json","w"),indent=1)
Q={"q":{"type":"choice","instructions":"What animal?","criteria":{"cat":None,"dog":None}}}
QT={"q":{"type":"choice","instructions":"Topic?","criteria":{"sports":None,"world":None}}}
S={"image":IMG}
t=time.time()
a=laya_vision.load(CK,device="cuda",compile=True)
rec("load(compile=True) succeeded","INFO","_compiled=%r load_s=%.1f"%(a._compiled,time.time()-t))
try:
    r=a.predict(S,Q); rec("compile=True + image predict (expect clear ValueError)","FAIL","no error: %s"%json.dumps(r)[:200])
except Exception as e: rec("compile=True + image predict (expect clear ValueError)","PASS" if isinstance(e,ValueError) else "FAIL","%s: %s"%(type(e).__name__,e))
try:
    a.predict("x\ud800",QT); rec("compile=True + text-only surrogate (expect ValueError)","FAIL","no error")
except Exception as e: rec("compile=True + text-only surrogate (expect ValueError)","PASS" if isinstance(e,ValueError) and "surrogate" in str(e) else "FAIL","%s: %s"%(type(e).__name__,e))
times=[]
for i in range(3):
    t=time.time()
    try:
        r=a.predict("The match ended 2-1 after extra time.",QT); dt=time.time()-t; times.append(dt)
        rec("compile=True text-only predict call %d"%(i+1),"PASS","%.1fs probs=%s"%(dt,json.dumps(r["answers"]["q"]["probabilities"])))
    except Exception as e:
        rec("compile=True text-only predict call %d"%(i+1),"FAIL","%s: %s after %.1fs"%(type(e).__name__,str(e)[:300],time.time()-t)); break
faulthandler.cancel_dump_traceback_later()
# reference without compile, same process (after freeing the compiled model)
del a; torch.cuda.empty_cache()
b=laya_vision.load(CK,device="cuda")
rb=b.predict("The match ended 2-1 after extra time.",QT)["answers"]["q"]["probabilities"]
rec("compile=True vs eager text-only probabilities (info)","INFO","eager=%s compiled_last=%s"%(json.dumps(rb),json.dumps(r["answers"]["q"]["probabilities"]) if times else "n/a"))
print("TIMES",times)
