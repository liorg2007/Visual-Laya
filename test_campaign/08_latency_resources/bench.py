import sys, os, time, json, gc, subprocess, random
import numpy as np, torch
sys.path.insert(0, "/home/zera/laya-vision")
import laya, laya_vision
from PIL import Image
CK = "/home/zera/laya-vision/runs/stage2_c/wise085_calibrated"
DEV = sys.argv[1]; ROUNDS = int(sys.argv[2]); IT = int(sys.argv[3]); OUT = sys.argv[4]
CPU = DEV == "cpu"
if CPU: torch.set_num_threads(int(os.environ.get("NT", "8")))
random.seed(0)
imdir = "/home/zera/laya-vision/data/images/coco"
files = sorted(os.listdir(imdir)); random.Random(0).shuffle(files)
paths = [os.path.join(imdir, f) for f in files[:200]]
imgs = [Image.open(p).convert("RGB") for p in paths[:64]]
sync = torch.cuda.synchronize if not CPU else (lambda: None)
def gpu_used():
    try: return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"]).decode().split(",")[1])
    except Exception: return -1
def mkq(n, k=4):
    return {"q%d"%i: {"type":"choice","instructions":"What is shown in the image? (%d)"%i,
            "criteria":{chr(65+j):"option %d number %d"%(j,i) for j in range(k)}} for i in range(n)}
def tm(fn, it, wu=3):
    for _ in range(wu): fn()
    ts=[]
    for i in range(it):
        sync(); t=time.perf_counter(); fn(i); sync(); ts.append((time.perf_counter()-t)*1e3)
    a=np.array(ts); return {"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),"mean":float(a.mean()),"std":float(a.std()),"n":len(a)}
def tm2(f, it, wu=3):  # f takes i
    for i in range(wu): f(i)
    ts=[]
    for i in range(it):
        sync(); t=time.perf_counter(); f(i+wu); sync(); ts.append((time.perf_counter()-t)*1e3)
    a=np.array(ts); return {"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),"mean":float(a.mean()),"std":float(a.std()),"n":len(a)}
t0=time.perf_counter()
if not CPU: torch.zeros(1,device="cuda"); torch.cuda.synchronize()
init_s=time.perf_counter()-t0
res = {"cuda_init_s":init_s,"device":DEV,"iters":IT,"rounds":ROUNDS,"threads":torch.get_num_threads()}
# cold
if not CPU: torch.cuda.reset_peak_memory_stats(); base0=torch.cuda.memory_allocated()
t=time.perf_counter(); agent=laya_vision.load(CK, device=DEV); sync(); res["load_s"]=time.perf_counter()-t
if not CPU:
    res["mem_after_load_MB"]=torch.cuda.memory_allocated()/2**20
    res["peak_after_load_MB"]=torch.cuda.max_memory_allocated()/2**20
q1=mkq(1)
t=time.perf_counter(); agent.predict({"image":imgs[0]}, q1); sync(); res["first_call_ms"]=(time.perf_counter()-t)*1e3
t=time.perf_counter(); agent.predict({"image":imgs[1]}, q1); sync(); res["second_call_ms"]=(time.perf_counter()-t)*1e3
tp=sum(p.numel() for p in agent.vlm.parameters()); res["params_total_vlm"]=tp
for nm,m in [("laya",agent.vlm.laya),("encoder",agent.vlm.laya.encoder),("vision_tower",agent.vlm.vision),("projector",agent.vlm.projector),("pooler",agent.vlm.pooler)]:
    res["params_"+nm]=sum(p.numel() for p in m.parameters())
res["dtype"]=str(next(agent.vlm.parameters()).dtype)
rounds=[]
TEXT="Customer writes: the blender arrived with a cracked jar and the motor smells burnt after one use. I want my money back."
for r in range(ROUNDS):
    R={"gpu_util_start":gpu_used()}
    if not CPU: torch.cuda.reset_peak_memory_stats()
    # questions per image (distinct image each call -> includes decode+preprocess+vision)
    for n in (1,2,4,8,16,32):
        qs=mkq(n)
        R["img_q%d"%n]=tm2(lambda i: agent.predict({"image":imgs[i%64]}, qs), IT)
        if not CPU: R["img_q%d"%n]["peak_MB"]=torch.cuda.max_memory_allocated()/2**20
    # same image repeated (identical input)
    qs=mkq(8)
    R["img_q8_same"]=tm2(lambda i: agent.predict({"image":imgs[0]}, qs), IT)
    R["img_q8_path"]=tm2(lambda i: agent.predict({"image":paths[i%64]}, qs), IT)
    R["text_q8"]=tm2(lambda i: agent.predict(TEXT, qs), IT)
    R["img_text_q8"]=tm2(lambda i: agent.predict({"image":imgs[i%64],"text":TEXT}, qs), IT)
    R["text_q1"]=tm2(lambda i: agent.predict(TEXT, mkq(1)), IT)
    # option counts, 1 question
    for k in (2,3,4,6,8,10,12):
        qk={"q":{"type":"choice","instructions":"What is shown?","criteria":{("o%d"%j):"option number %d"%j for j in range(k)}}}
        R["img_opts%d"%k]=tm2(lambda i: agent.predict({"image":imgs[i%64]}, qk), IT)
    R["img_noul"]=tm2(lambda i: agent.predict({"image":imgs[i%64]}, {"q":{"type":"noul","instructions":"Is there a dog?"}}), IT)
    # batch of images, 1q & 4q
    for B in (1,2,4,8,16):
        for nq in (1,4):
            qs=mkq(nq)
            R["batch%d_q%d"%(B,nq)]=tm2(lambda i: agent.predict_batch([{"image":imgs[(i*B+j)%64]} for j in range(B)], qs), max(IT//2,6))
            if not CPU: R["batch%d_q%d"%(B,nq)]["peak_MB"]=torch.cuda.max_memory_allocated()/2**20
    # same image across B states in batch (dedup)
    R["batch8_sameimg_q4"]=tm2(lambda i: agent.predict_batch([{"image":imgs[0]}]*8, mkq(4)), max(IT//2,6))
    if not CPU: R["peak_round_MB"]=torch.cuda.max_memory_allocated()/2**20
    R["gpu_util_end"]=gpu_used()
    rounds.append(R); print("round",r,"done",R["img_q1"]["p50"],R["img_q32"]["p50"],flush=True)
res["rounds"]=rounds
# component timing: preprocess / vision encode alone
from laya_vision.images import load_image_and_hash, preprocess
comp={}
comp["decode_hash_path"]=tm2(lambda i: load_image_and_hash(paths[i%64]), 30)
comp["decode_hash_pil"]=tm2(lambda i: load_image_and_hash(imgs[i%64]), 30)
pv=preprocess([imgs[0]],224)
comp["preprocess"]=tm2(lambda i: preprocess([imgs[i%64]],224), 30)
dv=agent.device
def enc(i):
    with torch.no_grad(): agent.vlm.encode_images(pv.to(dv))
comp["encode_images_amp_none"]=tm2(enc,30)
res["components"]=comp
with open(OUT,"w") as f: json.dump(res,f,indent=1)
