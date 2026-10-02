import json, sys, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/home/zera/laya-vision/test_campaign/04_prompt_invariance"
TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
R = json.load(open("%s/raw_%s.json" % (OUT, TAG)))
rng = np.random.default_rng(0)
def boot(vals, B=2000):
    v = np.asarray(vals, float); n = len(v)
    if n == 0: return [None, None, None, 0]
    idx = rng.integers(0, n, (B, n)); m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), n]
TASKS = ["scienceqa", "aokvqa", "cifar10", "food101", "oxford_pets", "eurosat"]
perm_names = ["perm%d" % i for i in range(6)]
def recs(task=None): return [r for r in R if task is None or r["task"] == task]
def acc(r, v): return float(r["v"][v]["pred"] == r["gold"]) if v in r["v"] else None
res = {"n_records": {t: len(recs(t)) for t in TASKS}, "tag": TAG}
# ---- per-variant accuracy, paired diff vs base, flip rate vs base pred
variants = sorted({v for r in R for v in r["v"]})
tab = {}
for scope in TASKS + ["ALL"]:
    rs = recs(None if scope == "ALL" else scope)
    # ALL = macro average over tasks of per-task means is clumsy; pool records (equal N per task)
    row = {}
    for v in variants:
        sel = [r for r in rs if v in r["v"] and "base" in r["v"]]
        if not sel: continue
        a = [acc(r, v) for r in sel]; b = [acc(r, "base") for r in sel]
        flip = [float(r["v"][v]["pred"] != r["v"]["base"]["pred"]) for r in sel]
        row[v] = {"acc": boot(a), "base_acc": boot(b), "delta": boot(np.array(a) - np.array(b)), "flip": boot(flip),
                  "mean_k": float(np.mean([r["v"][v]["k"] for r in sel])), "chance": float(np.mean([1.0 / r["v"][v]["k"] for r in sel]))}
    tab[scope] = row
res["variants"] = tab
# ---- position bias / permutation consistency (base + 6 perms)
pb = {}
for scope in TASKS + ["ALL"]:
    rs = recs(None if scope == "ALL" else scope)
    cons, majacc, anyacc, bflip = [], [], [], []
    gp = collections.defaultdict(list); predpos = collections.Counter(); K_tot = collections.Counter()
    pair_flip = []
    for r in rs:
        vs = [v for v in ["base"] + perm_names if v in r["v"]]
        preds = [r["v"][v]["pred"] for v in vs]
        cons.append(float(len(set(preds)) == 1))
        c = collections.Counter(preds); top = c.most_common(1)[0][0]
        majacc.append(float(top == r["gold"]))
        for v in perm_names:
            if v in r["v"]:
                pair_flip.append(float(r["v"][v]["pred"] != r["v"]["base"]["pred"]))
                k = r["v"][v]["k"]
                gp[(k >= 4 and "late" or "any", ) ].append(0)
        # position accuracy: relative gold position (0..1) bucket first/mid/last under perms
        for v in perm_names:
            if v in r["v"]:
                x = r["v"][v]; k = x["k"]
                if k < 3: continue
                pos = "first" if x["gold_pos"] == 0 else ("last" if x["gold_pos"] == k - 1 else "middle")
                gp[pos].append(float(x["pred"] == r["gold"]))
                predpos["first" if x["pred_pos"] == 0 else ("last" if x["pred_pos"] == k - 1 else "middle")] += 1
                K_tot["n"] += 1
    ind = [np.mean([acc(r, v) for v in perm_names if v in r["v"]]) for r in rs]
    pb[scope] = {"all_orders_agree_rate": boot(cons), "pairwise_flip_vs_base_perm": boot(pair_flip),
                 "mean_acc_over_perms": boot(ind), "majority_vote_acc": boot(majacc),
                 "acc_by_gold_position": {k: boot(v) for k, v in gp.items() if k in ("first", "middle", "last")},
                 "pred_position_share": {k: predpos[k] / max(1, K_tot["n"]) for k in ("first", "middle", "last")},
                 "n_perm_instances": K_tot["n"]}
res["position"] = pb
# ---- count scaling
ks = {}
for scope in TASKS + ["ALL"]:
    rs = recs(None if scope == "ALL" else scope); row = {}
    for v in variants:
        if not v.startswith("k"): continue
        if not v[1:].isdigit(): continue
        sel = [r for r in rs if v in r["v"]]
        if sel: row[int(v[1:])] = {"acc": boot([acc(r, v) for r in sel]), "chance": 1.0 / int(v[1:])}
    ks[scope] = row
res["count_scaling"] = ks
json.dump(res, open("%s/results.json" % OUT, "w"), indent=1)
# ---- plots
fig, ax = plt.subplots(figsize=(7, 4))
for t in TASKS:
    row = ks[t]; xs = sorted(row)
    if not xs: continue
    ys = [row[k]["acc"][0] for k in xs]; lo = [row[k]["acc"][1] for k in xs]; hi = [row[k]["acc"][2] for k in xs]
    ax.plot(xs, ys, marker="o", label=t); ax.fill_between(xs, lo, hi, alpha=0.12)
xs = list(range(2, 21)); ax.plot(xs, [1 / k for k in xs], "k--", label="chance")
ax.set_xlabel("# options (gold + k-1 distractors)"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.set_title("Option-count scaling")
fig.tight_layout(); fig.savefig(OUT + "/count_scaling.png", dpi=130)
groups = [v for v in variants if v not in ("base", "drop_half") and not v.startswith("k") and not v.startswith("perm")] + perm_names[:1]
fig, ax = plt.subplots(figsize=(9, 5))
show = [v for v in variants if v not in ("base",) and not (v.startswith("k") and v[1:].isdigit())]
ys = [tab["ALL"][v]["flip"][0] for v in show if v in tab["ALL"]]; names = [v for v in show if v in tab["ALL"]]
lo = [tab["ALL"][v]["flip"][1] for v in names]; hi = [tab["ALL"][v]["flip"][2] for v in names]
ax.barh(names, ys, xerr=[np.array(ys) - lo, np.array(hi) - ys]); ax.set_xlabel("flip rate vs base prediction (pooled, 95% CI)")
fig.tight_layout(); fig.savefig(OUT + "/flip_rates.png", dpi=130)
# ---- print compact
def f(x): return "%.3f [%.3f,%.3f]" % tuple(x[:3]) if x[0] is not None else "NA"
for scope in ["ALL"] + TASKS:
    print("==", scope, res["n_records"].get(scope, ""))
    for v, d in tab[scope].items():
        print("%-20s acc %s  d %s  flip %s  k=%.1f ch=%.2f n=%d" % (v, f(d["acc"]), f(d["delta"]), f(d["flip"]), d["mean_k"], d["chance"], d["acc"][3]))
    p = pb[scope]; print("POS", {k: (f(v) if isinstance(v, list) else v) for k, v in p.items() if k != "acc_by_gold_position"}, {k: f(v) for k, v in p["acc_by_gold_position"].items()})
