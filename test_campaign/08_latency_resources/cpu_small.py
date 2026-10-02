import sys, os, time, json, random, resource
import numpy as np, torch
sys.path.insert(0,"/home/zera/laya-vision")
import laya_vision
from PIL import Image
torch.set_num_threads(int(os.environ.get("NT","8")))
CK="/home/zera/laya-vision/runs/stage2_c/wise085_calibrated"
imdir="/home/zera/laya-vision/data/images/coco"
files=sorted(os.listdir(imdir)); random.Random(0).shuffle(files)
imgs=[Image.open(os.path.join(imdir,f)).convert("RGB") for f in files[:16]]
def mkq(n,k=4): return {"q%d"%i:{"type":"choice","instructions":"What is shown in the image? (%d)"%i,"criteria":{chr(65+j):"option %d number %d"%(j,i) for j in range(k)}} for i in range(n)}
t=time.perf_counter(); a=laya_vision.load(CK,device="cpu"); res={"threads":torch.get_num_threads(),"load_s":time.perf_counter()-t,"dtype":str(next(a.vlm.parameters()).dtype)}
def tm(f,it=6,wu=2):
    for i in range(wu): f(i)
    ts=[]; c0=time.process_time()
    for i in range(it): t=time.perf_counter(); f(i+wu); ts.append((time.perf_counter()-t)*1e3)
    x=np.array(ts); return {"p50":float(np.percentile(x,50)),"p95":float(np.percentile(x,95)),"n":it,"cpu_s_per_call":(time.process_time()-c0)/it}
for n in (1,8): res["img_q%d"%n]=tm(lambda i:a.predict({"image":imgs[i%16]},mkq(n)))
res["text_q8"]=tm(lambda i:a.predict("The blender arrived cracked and smells burnt.",mkq(8)))
res["batch8_q1"]=tm(lambda i:a.predict_batch([{"image":imgs[(i*8+j)%16]} for j in range(8)],mkq(1)),4,1)
res["max_rss_MB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
json.dump(res,open("cpu_small.json","w"),indent=1); print(res)
