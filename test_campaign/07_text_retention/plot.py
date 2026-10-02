import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=os.path.dirname(os.path.abspath(__file__)); r=json.load(open(D+"/results.json"))["models"]
names=["a_cal","b_unmerged_cal","b_wise080cal","final_c085cal"]; labs=["stage2_a","stage2_b","wise080","wise085"]
fig,ax=plt.subplots(1,3,figsize=(11,3.5))
for a,t in zip(ax,["ag_news","boolq","typed_decisions"]):
    o=r[names[0]][t]; xs=["stock"]+labs; ys=[o["stock_acc"]]+[r[n][t]["acc"] for n in names]
    ci=[o["stock_acc_ci"]]+[r[n][t]["acc_ci"] for n in names]
    a.bar(xs,ys,yerr=[[y-c[0] for y,c in zip(ys,ci)],[c[1]-y for y,c in zip(ys,ci)]],color=["gray"]+["C0"]*4,capsize=3)
    a.set_title(t); a.set_ylim(0.3 if t=="typed_decisions" else 0.6,1); a.tick_params(axis="x",rotation=30)
plt.tight_layout(); plt.savefig(D+"/accuracy.png",dpi=120)
