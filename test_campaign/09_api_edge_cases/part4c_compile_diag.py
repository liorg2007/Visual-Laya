# Is the compile=True text-only failure laya_vision-specific? Same text predict under compile=True with
# (1) stock laya.Agent on the vision checkpoint dir, (2) stock laya.Agent on convaiinnovations/laya (HF cache),
# (3) VisionAgent (full traceback).
import json, sys, time, traceback, torch, laya, laya_vision
R="/home/zera/laya-vision"; CK=R+"/runs/stage2_c/wise085_calibrated"
QT={"q":{"type":"choice","instructions":"Topic?","criteria":{"sports":None,"world":None}}}
out={}
for name,mk in [("laya.Agent(vision ckpt dir)",lambda:laya.Agent(CK,device="cuda",compile=True)),
                ("laya.Agent(convaiinnovations/laya)",lambda:laya.Agent("convaiinnovations/laya",device="cuda",compile=True)),
                ("laya_vision.load(ckpt)",lambda:laya_vision.load(CK,device="cuda",compile=True))]:
    t=time.time()
    try:
        a=mk(); tl=time.time()-t; ts=[]
        for i in range(3):
            t1=time.time(); r=a.predict("The match ended 2-1 after extra time.",QT); ts.append(round(time.time()-t1,2))
        out[name]="OK load %.1fs predict times %s probs %s"%(tl,ts,json.dumps(r["answers"]["q"]["probabilities"]))
    except Exception as e:
        out[name]="%s: %s"%(type(e).__name__,str(e)[:300]); traceback.print_exc()
    print(name,"->",out[name],flush=True)
    try: del a
    except NameError: pass
    torch.cuda.empty_cache(); torch._dynamo.reset()
json.dump(out,open("results_part4c.json","w"),indent=1)
