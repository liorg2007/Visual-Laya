import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
R = json.load(open("results.json" if os.path.exists("results.json") and json.load(open("results.json"))["n"] == 150 else "ckpt_n150.json"))
fams = ["blur", "noise", "jpeg", "brightness", "contrast", "zoom", "downscale", "occlusion"]
fig, ax = plt.subplots(2, 8, figsize=(30, 7))
for r, task in enumerate(["eurosat", "oxford_pets"]):
    T = R["tasks"][task]; c = T["clean@0"]
    for j, f in enumerate(fams):
        es = sorted([e for e in T.values() if e["family"] == f], key=lambda e: e["level"])
        if not es: continue
        x = [e["level"] for e in es]; a = ax[r, j]
        a.errorbar(x, [e["acc"] for e in es], yerr=[[e["acc"] - e["acc_ci"][0] for e in es], [e["acc_ci"][1] - e["acc"] for e in es]], marker="o", label="accuracy", capsize=3)
        a.plot(x, [e["mean_conf"] for e in es], marker="s", ls="--", label="mean confidence")
        a.axhline(c["acc"], color="gray", lw=.8); a.set_ylim(0, 1.02)
        a.set_title(f"{task}: {f}"); a.set_xlabel(es[0]["param"])
        if f in ("downscale",): a.set_xscale("log", base=2)
        if r == 0 and j == 0: a.legend()
plt.tight_layout(); plt.savefig("curves.png", dpi=80)
if False: pass
# table
for task in R["tasks"]:
    T = R["tasks"][task]; print("##", task)
    print("| condition | acc | 95% CI | mean conf | conf(correct) | conf(wrong) | agree w/ clean | top-pred share |\n|---|---|---|---|---|---|---|---|")
    for k, e in T.items():
        f = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"| {k} | {e['acc']:.3f} | {e['acc_ci'][0]:.2f}-{e['acc_ci'][1]:.2f} | {e['mean_conf']:.2f} | {f(e['conf_when_correct'])} | {f(e['conf_when_wrong'])} | {e['pred_agree_with_clean']:.2f} | {e['top_pred_share']:.2f} |")
