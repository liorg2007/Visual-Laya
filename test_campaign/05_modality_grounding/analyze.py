import json, sys, numpy as np, collections
TAG = sys.argv[1] if len(sys.argv) > 1 else "final"
rows = json.load(open("raw_%s.json" % TAG))
rng = np.random.RandomState(0)
CONDS = ["real","black","white","noise","patch4x4","shuffle_within","swap_cross_task","text_only"]
by = collections.defaultdict(list)
for r in rows: by[(r["cond"], r["task"])].append(r)
def stats(rs):
    p = np.array([np.array(r["probs"]) / sum(r["probs"]) for r in rs], dtype=object)
    corr = np.array([float(np.argmax(r["probs"]) == np.argmax(r["target"])) for r in rs])
    conf = np.array([max(r["probs"]) / sum(r["probs"]) for r in rs])
    return corr, conf
def boot(vals, groups, B=1000):
    ug = sorted(set(groups)); idx = {g: np.where(np.array(groups) == g)[0] for g in ug}
    m = []
    for _ in range(B):
        s = rng.choice(len(ug), len(ug)); ii = np.concatenate([idx[ug[i]] for i in s]); m.append(vals[ii].mean())
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def majority(rs):
    # majority-class accuracy within (qtype,k) strata of the real-image sample
    g = collections.defaultdict(collections.Counter)
    for r in rs: g[(r["qtype"], r["k"])][int(np.argmax(r["target"]))] += 1
    tot = sum(max(c.values()) for c in g.values()); return tot / len(rs)
out = {"checkpoint_tag": TAG, "per_task": {}, "pooled_image_tasks": {}}
real_by_id = {}
for t in sorted({r["task"] for r in rows}):
    real = by[("real", t)]; maj = majority(real)
    chance = float(np.mean([1.0 / r["k"] for r in real]))
    out["per_task"][t] = {"n": len(real), "n_images": len({r["image"] for r in real}), "majority_baseline": maj, "chance": chance, "conds": {}}
    rp = {r["id"]: r["probs"] for r in real}
    for c in CONDS:
        rs = by[(c, t)]; corr, conf = stats(rs); lo, hi = boot(corr, [r["image"] for r in rs])
        agree = float(np.mean([np.argmax(r["probs"]) == np.argmax(rp[r["id"]]) for r in rs]))
        out["per_task"][t]["conds"][c] = {"acc": float(corr.mean()), "ci95": [lo, hi], "mean_conf": float(conf.mean()), "agree_with_real_pred": agree}
# pooled: macro average over tasks (excluding none), and per-qtype breakdown
tasks = sorted(out["per_task"])
for c in CONDS:
    accs = [out["per_task"][t]["conds"][c]["acc"] for t in tasks]
    confs = [out["per_task"][t]["conds"][c]["mean_conf"] for t in tasks]
    out["pooled_image_tasks"][c] = {"macro_acc": float(np.mean(accs)), "macro_conf": float(np.mean(confs))}
out["pooled_image_tasks"]["majority_baseline_macro"] = float(np.mean([out["per_task"][t]["majority_baseline"] for t in tasks]))
# qtype-split
out["by_task_qtype"] = {}
for t in tasks:
    for qt in ("choice", "noul"):
        d = {}
        for c in CONDS:
            rs = [r for r in by[(c, t)] if r["qtype"] == qt]
            if rs:
                corr, conf = stats(rs); d[c] = {"n": len(rs), "acc": float(corr.mean()), "conf": float(conf.mean())}
        if d: out["by_task_qtype"][t + "/" + qt] = d
# vqav2 yes/no
vq = {}
for c in CONDS:
    rs = by[(c, "vqav2_yesno")]
    yes_pred = np.array([r["probs"][1] > 0.5 for r in rs])
    gold_yes = np.array([np.argmax(r["target"]) == 1 for r in rs])
    corr, conf = stats(rs)
    vq[c] = {"acc": float(corr.mean()), "ci95": out["per_task"]["vqav2_yesno"]["conds"][c]["ci95"], "pred_yes_rate": float(yes_pred.mean()),
             "mean_p_yes": float(np.mean([r["probs"][1] for r in rs])), "acc_on_gold_yes": float(corr[gold_yes].mean()), "acc_on_gold_no": float(corr[~gold_yes].mean())}
vq["always_yes_acc"] = float(gold_yes.mean()); vq["always_no_acc"] = float(1 - gold_yes.mean()); vq["n"] = len(rs)
out["vqav2"] = vq
json.dump(out, open("results_%s.json" % TAG, "w"), indent=1)
print("%-16s" % "task", *["%9s" % c[:9] for c in CONDS], "  maj")
for t in tasks:
    print("%-16s" % t, *["%9.3f" % out["per_task"][t]["conds"][c]["acc"] for c in CONDS], " %.3f" % out["per_task"][t]["majority_baseline"])
print("%-16s" % "MACRO", *["%9.3f" % out["pooled_image_tasks"][c]["macro_acc"] for c in CONDS])
print("%-16s" % "MACRO conf", *["%9.3f" % out["pooled_image_tasks"][c]["macro_conf"] for c in CONDS])
print(json.dumps(vq, indent=0))

# ---- extra: macro-accuracy CI (bootstrap images within task), paired drop vs real, plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng2 = np.random.RandomState(1)
imgs = {t: sorted({r["image"] for r in by[("real", t)]}) for t in tasks}
cor = {}  # (cond,task,image) -> mean correctness over that image's questions
for (c, t), rs in by.items():
    d = collections.defaultdict(list)
    for r in rs: d[r["image"]].append(float(np.argmax(r["probs"]) == np.argmax(r["target"])))
    for im, v in d.items(): cor[(c, t, im)] = np.mean(v)
B = 2000; boots = {c: [] for c in CONDS}; drops = {c: [] for c in CONDS}
for _ in range(B):
    acc = {c: [] for c in CONDS}
    for t in tasks:
        s = [imgs[t][i] for i in rng2.randint(0, len(imgs[t]), len(imgs[t]))]
        for c in CONDS: acc[c].append(np.mean([cor[(c, t, im)] for im in s]))
    for c in CONDS:
        boots[c].append(np.mean(acc[c])); drops[c].append(np.mean(acc["real"]) - np.mean(acc[c]))
out["macro_ci"] = {c: {"mean": float(np.mean([np.mean([cor[(c,t,im)] for im in imgs[t]]) for t in tasks])),
                       "ci95": [float(np.percentile(boots[c], 2.5)), float(np.percentile(boots[c], 97.5))],
                       "drop_vs_real": float(np.mean(drops[c])), "drop_ci95": [float(np.percentile(drops[c], 2.5)), float(np.percentile(drops[c], 97.5))]} for c in CONDS}
out["n_images_per_task"] = {t: len(imgs[t]) for t in tasks}
out["n_questions_per_task"] = {t: out["per_task"][t]["n"] for t in tasks}
json.dump(out, open("results_%s.json" % TAG, "w"), indent=1)
json.dump(out, open("results.json", "w"), indent=1)
for c in CONDS: print(c, out["macro_ci"][c])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.8, 1]})
w = 0.8 / len(CONDS); cols = plt.cm.tab10.colors
for k, c in enumerate(CONDS):
    ax[0].bar(np.arange(len(tasks)) + k * w, [out["per_task"][t]["conds"][c]["acc"] for t in tasks], w, label=c, color=cols[k])
ax[0].scatter(np.arange(len(tasks)) + 0.4, [out["per_task"][t]["majority_baseline"] for t in tasks], marker="_", s=400, color="k", label="majority", zorder=5)
ax[0].set_xticks(np.arange(len(tasks)) + 0.4); ax[0].set_xticklabels(tasks, rotation=30); ax[0].set_ylabel("accuracy"); ax[0].legend(fontsize=7, ncol=3); ax[0].set_title("Accuracy by task and image condition")
m = [out["macro_ci"][c]["mean"] for c in CONDS]; e = [[out["macro_ci"][c]["mean"] - out["macro_ci"][c]["ci95"][0] for c in CONDS], [out["macro_ci"][c]["ci95"][1] - out["macro_ci"][c]["mean"] for c in CONDS]]
ax[1].bar(range(len(CONDS)), m, yerr=e, color=cols[:len(CONDS)], capsize=3); ax[1].axhline(out["pooled_image_tasks"]["majority_baseline_macro"], ls="--", c="k", label="majority (macro)")
ax[1].set_xticks(range(len(CONDS))); ax[1].set_xticklabels(CONDS, rotation=45, ha="right"); ax[1].set_title("Macro accuracy (95% boot CI)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("grounding.png", dpi=130)
