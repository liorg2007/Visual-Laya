import json, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze import summarize
d,o=summarize("gpu_bench.json")
pipe=json.load(open("pipe_cuda.json")) if os.path.exists("pipe_cuda.json") else None
out={"checkpoint":"runs/stage2_c/wise085_calibrated","device":"RTX 4060 Ti 8GB (shared with ~11 agents)","setup":{k:d[k] for k in d if k not in("rounds","components")},
     "summary":o,"components":d["components"],"pipeline":pipe,
     "caveat":"GPU shared with other agents; absolute latencies inflated and noisy; round-to-round range given."}
json.dump(out,open("results.json","w"),indent=1)
for k,v in o.items(): print("%-22s p50 %7.1f [%7.1f-%7.1f] p95 %7.1f peak=%s"%(k,v["p50_med_of_rounds"],v["p50_min"],v["p50_max"],v["p95_med_of_rounds"],v["peak_MB"]))
print({k:d[k] for k in d if k not in("rounds","components")}); print(d["components"])
if pipe: print(json.dumps(pipe,indent=0))
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ns=[1,2,4,8,16,32]; ax[0].errorbar(ns,[o["img_q%d"%n]["p50_med_of_rounds"] for n in ns],yerr=[[o["img_q%d"%n]["p50_med_of_rounds"]-o["img_q%d"%n]["p50_min"] for n in ns],[o["img_q%d"%n]["p50_max"]-o["img_q%d"%n]["p50_med_of_rounds"] for n in ns]],marker="o",label="image + n questions")
ax[0].plot(ns,[o["img_q%d"%n]["p95_med_of_rounds"] for n in ns],"--s",label="p95")
ax[0].set_xscale("log",base=2);ax[0].set_xlabel("questions per image");ax[0].set_ylabel("ms / call");ax[0].legend();ax[0].set_title("Latency vs #questions (1 image)")
Bs=[1,2,4,8,16]
for nq in (1,4):
    ax[1].plot(Bs,[o["batch%d_q%d"%(B,nq)]["p50_med_of_rounds"]/B for B in Bs],marker="o",label="%d q/img"%nq)
ax[1].set_xscale("log",base=2);ax[1].set_xlabel("batch size (images)");ax[1].set_ylabel("ms per image (p50)");ax[1].legend();ax[1].set_title("Batch throughput cost")
ks=["text_q8","img_q8","img_text_q8"]; ax[2].bar(range(3),[o[k]["p50_med_of_rounds"] for k in ks],yerr=[[o[k]["p50_med_of_rounds"]-o[k]["p50_min"] for k in ks],[o[k]["p50_max"]-o[k]["p50_med_of_rounds"] for k in ks]],capsize=4)
ax[2].set_xticks(range(3));ax[2].set_xticklabels(["text only","image only","image+text"]);ax[2].set_ylabel("ms (8 questions)");ax[2].set_title("Modality cost")
plt.tight_layout();plt.savefig("latency.png",dpi=120)
