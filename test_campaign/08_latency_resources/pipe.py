# Caption->Laya pipeline cost vs Laya-Vision vs stock Laya text; one model family at a time.
import sys, os, time, json, random, gc
import numpy as np, torch
sys.path.insert(0, "/home/zera/laya-vision")
import laya, laya_vision
from PIL import Image
from laya_vision.eval.baselines import BlipCaptioner, CaptionLaya
DEV=sys.argv[1]; IT=int(sys.argv[2]); OUT=sys.argv[3]
CK="/home/zera/laya-vision/runs/stage2_c/wise085_calibrated"
imdir="/home/zera/laya-vision/data/images/coco"
files=sorted(os.listdir(imdir)); random.Random(0).shuffle(files)
paths=[os.path.join(imdir,f) for f in files[:32]]
imgs=[Image.open(p).convert("RGB") for p in paths]
sync=torch.cuda.synchronize if DEV=="cuda" else (lambda:None)
def mkq(n,k=4):
    return {"q%d"%i:{"type":"choice","instructions":"What is shown in the image? (%d)"%i,"criteria":{chr(65+j):"option %d number %d"%(j,i) for j in range(k)}} for i in range(n)}
def tm(f,it,wu=3):
    for i in range(wu): f(i)
    ts=[]
    for i in range(it):
        sync(); t=time.perf_counter(); f(i+wu); sync(); ts.append((time.perf_counter()-t)*1e3)
    a=np.array(ts); return {"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),"mean":float(a.mean()),"std":float(a.std()),"n":len(a)}
res={"device":DEV}
if DEV=="cuda": torch.zeros(1,device="cuda"); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
# stock text Laya (checkpoint's own text path = stock weights? use hub stock)
t=time.perf_counter(); stock=laya.Agent("convaiinnovations/laya",device=DEV); sync(); res["stock_load_s"]=time.perf_counter()-t
if DEV=="cuda": res["stock_mem_MB"]=torch.cuda.memory_allocated()/2**20; res["stock_peak_MB"]=torch.cuda.max_memory_allocated()/2**20
res["stock_params"]=sum(p.numel() for p in stock.model.parameters())
cap="A man riding a skateboard down the side of a ramp in a park near trees."
for n in (1,8,32):
    res["stock_text_q%d"%n]=tm(lambda i: stock.predict(cap,mkq(n)),IT)
if DEV=="cuda": torch.cuda.reset_peak_memory_stats()
t=time.perf_counter(); blip=BlipCaptioner(device=DEV); sync(); res["blip_load_s"]=time.perf_counter()-t
res["blip_params"]=sum(p.numel() for p in blip.model.parameters())
if DEV=="cuda": res["blip_mem_MB"]=torch.cuda.memory_allocated()/2**20; res["blip_peak_MB"]=torch.cuda.max_memory_allocated()/2**20
res["blip_caption_only"]=tm(lambda i: blip(imgs[i%32]),IT)
cl=CaptionLaya(stock,captioner=blip)  # no cache dir -> no caption caching
for n in (1,8,32):
    res["caption_laya_miss_q%d"%n]=tm(lambda i: cl.predict({"image":imgs[i%32]},mkq(n)),IT)
# cache hit: caption cached in-process
caps=[blip(im) for im in imgs]
cl2=CaptionLaya(stock,captioner=lambda im: "x",cache_dir="/tmp/claude-1000/-home-zera-laya-vision/2343de08-0e29-45cc-b8d5-c7b37b3b00fb/scratchpad/capcache")
for p in paths[:8]: cl2.predict({"image":p},mkq(1))  # populates (captioner stub) -> measures cache-hit path
res["caption_laya_cachehit_q8_path"]=tm(lambda i: cl2.predict({"image":paths[i%8]},mkq(8)),IT)
del blip,cl,cl2,stock; gc.collect()
if DEV=="cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
agent=laya_vision.load(CK,device=DEV)
if DEV=="cuda": res["vision_mem_MB"]=torch.cuda.memory_allocated()/2**20
for n in (1,8,32):
    res["vision_img_q%d"%n]=tm(lambda i: agent.predict({"image":imgs[i%32]},mkq(n)),IT)
if DEV=="cuda": res["vision_peak_MB"]=torch.cuda.max_memory_allocated()/2**20
json.dump(res,open(OUT,"w"),indent=1)
