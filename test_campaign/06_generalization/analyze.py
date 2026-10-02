import json, collections, math, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/home/zera/laya-vision/test_campaign/06_generalization"
rows = json.load(open(D + "/raw_rows.json"))
def wilson(c, n, z=1.96):
    if n == 0: return (float("nan"),)*2
    p = c/n; d = 1+z*z/n; m = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return m-h, m+h
g = collections.OrderedDict()
for r in rows:
    if r["task"]=="coco" and r["var"].startswith("orig("): r["var"]="orig(4-8 opt, varies)"
g_=None
for r in rows: g.setdefault((r["task"], r["var"]), []).append(r)
rng = np.random.default_rng(0)
res = []
for (t, v), rs in g.items():
    n = len(rs); c = sum(r["correct"] for r in rs); lo, hi = wilson(c, n)
    chance = float(np.mean([1/r["k"] for r in rs]))
    base = next(k for k in g if k[0] == t and k[1].startswith("orig"))
    d = dlo = dhi = None
    if (t, v) != base:
        bm = {r["base"]: r["correct"] for r in g[base]}
        pairs = [(r["correct"], bm[r["base"]]) for r in rs if r["base"] in bm]
        diffs = np.array([a-b for a, b in pairs], float)
        bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
        d, dlo, dhi = float(diffs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    res.append(dict(task=t, variant=v, status=rs[0]["status"], n=n, acc=c/n, ci_lo=lo, ci_hi=hi, chance=chance,
                    delta_vs_orig=d, delta_lo=dlo, delta_hi=dhi, mean_conf=float(np.mean([r["conf"] for r in rs])), chance_norm=(c/n-chance)/(1-chance)))
json.dump(res, open(D + "/results.json", "w"), indent=1)
lines = ["| task | variant | status | n | acc [95% CI] | chance | conf | (acc-ch)/(1-ch) | d vs orig [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
for r in res:
    dd = "" if r["delta_vs_orig"] is None else "%+.3f [%+.3f,%+.3f]%s" % (r["delta_vs_orig"], r["delta_lo"], r["delta_hi"], "" if r["delta_lo"] <= 0 <= r["delta_hi"] else " *")
    lines.append("| %s | %s | %s | %d | %.3f [%.3f,%.3f] | %.3f | %.3f | %.3f | %s |" % (r["task"], r["variant"], r["status"], r["n"], r["acc"], r["ci_lo"], r["ci_hi"], r["chance"], r["mean_conf"], r["chance_norm"], dd))
open(D + "/table.md", "w").write("\n".join(lines)); print("\n".join(lines))
fig, ax = plt.subplots(figsize=(9, 11)); ys = range(len(res))
ax.barh(list(ys), [r["acc"] for r in res], xerr=[[r["acc"]-r["ci_lo"] for r in res], [r["ci_hi"]-r["acc"] for r in res]], color="#4477aa")
ax.scatter([r["chance"] for r in res], list(ys), c="k", marker="|", s=120, label="chance")
ax.set_yticks(list(ys)); ax.set_yticklabels(["%s: %s" % (r["task"], r["variant"]) for r in res], fontsize=7); ax.invert_yaxis(); ax.legend(); ax.set_xlabel("accuracy")
plt.tight_layout(); plt.savefig(D + "/generalization.png", dpi=110)
