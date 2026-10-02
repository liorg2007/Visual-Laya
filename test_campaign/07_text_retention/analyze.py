import json, glob, os, sys, math
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
def load(name):
    rows = [json.loads(l) for l in open("%s/preds/%s.jsonl" % (D, name))]
    return {r["id"]: r for r in rows}
def arr(rows, ids):
    P, G, T = [], [], []
    for i in ids:
        r = rows[i]; P.append(np.array(r["probs"], float)); T.append(np.array(r["target"], float)); G.append(int(np.argmax(r["target"])))
    return P, np.array(G), T
def ece(conf, corr, nb=10):
    b = np.minimum((conf * nb).astype(int), nb - 1); e = 0
    for k in range(nb):
        m = b == k
        if m.any(): e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)
def stats(rows, ids):
    P, G, T = arr(rows, ids)
    pred = np.array([p.argmax() for p in P]); conf = np.array([p.max() for p in P]); corr = (pred == G).astype(float)
    nll = np.array([-np.log(max(p[g], 1e-9)) for p, g in zip(P, G)])
    brier = np.array([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(P, G)])
    return dict(pred=pred, conf=conf, corr=corr, nll=nll, brier=brier)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]
def mcnemar(a, b):
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum()); n = n01 + n10
    if n == 0: return n10, n01, 1.0
    k = min(n01, n10)
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p
def boot_diff(a, b, B=5000):
    n = len(a); idx = rng.integers(0, n, (B, n)); d = (b[idx] - a[idx]).mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
TASKS = ["ag_news", "boolq", "typed_decisions"]
def summarize(name, base="stock"):
    S, B = load(name), load(base)
    out = {}
    for t in TASKS + ["ALL"]:
        ids = [i for i in B if (t == "ALL" or B[i]["task"] == t) and i in S and S[i]["probs"] and B[i]["probs"]]
        s, b = stats(S, ids), stats(B, ids)
        n = len(ids); k = int(s["corr"].sum()); kb = int(b["corr"].sum())
        n10, n01, p = mcnemar(b["corr"], s["corr"])  # n10: stock right/ft wrong
        o = dict(n=n, acc=k / n, acc_ci=wilson(k, n), stock_acc=kb / n, stock_acc_ci=wilson(kb, n),
                 diff=(k - kb) / n, diff_ci_boot=boot_diff(b["corr"], s["corr"]),
                 stock_right_ft_wrong=n10, ft_right_stock_wrong=n01, mcnemar_p=p,
                 agreement=float((s["pred"] == b["pred"]).mean()),
                 mean_abs_prob_diff=float(np.mean([np.abs(arr(S, [i])[0][0] - arr(B, [i])[0][0]).max() for i in ids])),
                 ece10=ece(s["conf"], s["corr"]), stock_ece10=ece(b["conf"], b["corr"]),
                 nll=float(s["nll"].mean()), stock_nll=float(b["nll"].mean()),
                 brier=float(s["brier"].mean()), stock_brier=float(b["brier"].mean()),
                 mean_conf=float(s["conf"].mean()), stock_mean_conf=float(b["conf"].mean()))
        out[t] = o
    return out
if __name__ == "__main__":
    names = sorted(os.path.basename(f)[:-6] for f in glob.glob(D + "/preds/*.jsonl"))
    res = {n: summarize(n) for n in names if n != "stock"}
    # exact-identity check
    ident = {}
    if os.path.exists(D + "/preds/layaagent_c085cal.jsonl") and os.path.exists(D + "/preds/final_c085cal.jsonl"):
        A, B = load("final_c085cal"), load("layaagent_c085cal")
        d = max(np.abs(np.array(A[i]["probs"]) - np.array(B[i]["probs"])).max() for i in A if A[i]["probs"] and B[i]["probs"])
        ident = dict(n=len(A), max_abs_prob_diff_visionagent_vs_layaagent_same_ckpt=float(d),
                     identical_argmax=all(np.argmax(A[i]["probs"]) == np.argmax(B[i]["probs"]) for i in A))
    json.dump(dict(stock_n={t: sum(1 for r in load("stock").values() if r["task"] == t) for t in TASKS}, models=res, identity=ident), open(D + "/results.json", "w"), indent=1)
    for n, r in res.items():
        print(n, {t: (round(v["acc"], 3), round(v["diff"], 3), round(v["agreement"], 3), v["mcnemar_p"] if v["mcnemar_p"] > 1e-3 else "<1e-3") for t, v in r.items()})
    print(ident)
