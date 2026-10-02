"""Pre-fix vs post-fix comparison (all-caps normalisation in laya_vision/textnorm.py).

1. Replays run.py's variant construction (same seeds, drop_half driven by the recorded base preds)
   to find, for every (record, variant), whether any string the model sees would be rewritten by
   textnorm (question instructions, choice keys/descriptions, state text) -> "touched".
2. Paired comparison post - pre per variant (same task/id), bootstrap 95% CI, plus prediction
   change rate, split by touched / untouched and by exact-question pairing (first 72 records,
   where pre and post used identical random draws).
Writes compare.json and prints a table (-> compare.txt).
"""
import json, sys, collections, numpy as np
sys.argv = [sys.argv[0], "50"]
OUT = "/home/zera/laya-vision/test_campaign/04_prompt_invariance"
src = open(OUT + "/run.py").read().rsplit("\nmain()", 1)[0]
ns = {"__file__": OUT + "/run.py", "__name__": "replay"}
exec(compile(src, "run.py", "exec"), ns)
from laya_vision.textnorm import is_shouted, unshout_question, unshout_state

post = json.load(open(OUT + "/raw_main.json"))
pre = json.load(open(OUT + "/raw_main_prefix.json"))
postd = {(r["task"], r["id"]): r for r in post}
pred = {(r["task"], r["id"]): r for r in pre}

# ---------- replay (mirrors run.main exactly, without the model)
rng = ns["rng"]; build_variants = ns["build_variants"]; TASKS = ns["TASKS"]; LETTER = ns["LETTER"]; D = ns["D"]
per = {}
for task in TASKS:
    rows = ns["load_task"](task)
    pool = sorted({(v if task in LETTER else k) for r in rows for k, v in r["question"]["criteria"].items() if (v if task in LETTER else k)})
    seen = set(); uniq = []
    for r in rows:
        if r["image"] not in seen: seen.add(r["image"]); uniq.append(r)
    rr = ns["random"].Random(1); rr.shuffle(uniq); uniq = uniq[:50]
    per[task] = (uniq, pool)
order = [(t, per[t][0][i], per[t][1]) for i in range(50) for t in TASKS if i < len(per[t][0])]

def strings(q, state_text):
    out = [q["instructions"]]
    for k, d in q["criteria"].items():
        out.append(k)
        if isinstance(d, str): out.append(d)
    if state_text: out.append(state_text)
    return out

touched = {}  # (task,id) -> {variant: [shouted strings]}
mismatch = 0
examples = collections.defaultdict(list)
for task, r, pool in order:
    gold = int(np.argmax(r["target"]))
    V, items0, rep, text = build_variants(r, task, pool, rng)
    rec = postd[(task, r["id"])]
    bp = rec["v"]["base"]["pred"]
    keep = {gold, bp}
    others = [i for i in range(len(items0)) if i not in keep]
    rng.shuffle(others)
    keep |= set(others[:len(others) // 2])
    if len(keep) >= 2 and len(keep) < len(items0):
        qq, m = rep([items0[i] for i in sorted(keep)]); V["drop_half"] = (qq, m, gold)
    st = r.get("text")
    t = {}
    for nm, (q, m, g) in V.items():
        if nm in rec["v"] and len(q["criteria"]) != rec["v"][nm]["k"]: mismatch += 1
        sh = [s for s in strings(q, st) if is_shouted(s)]
        t[nm] = sh
        if sh and len(examples[nm]) < 3: examples[nm].append((task, r["id"], sh[:3]))
    touched[(task, r["id"])] = t
print("replay k-mismatches vs recorded:", mismatch)

# ---------- paired comparison
brng = np.random.default_rng(0)
def boot(v, B=4000):
    v = np.asarray(v, float); n = len(v)
    if n == 0: return [None, None, None, 0]
    m = v[brng.integers(0, n, (B, n))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), n]
first72 = {(r["task"], r["id"]) for r in pre[:72]}
variants = sorted({v for r in post for v in r["v"]})
RNG_VARS = {"perm0", "perm1", "perm2", "perm3", "k02", "k05", "k10", "k20", "add_dist", "drop_half"}

def compare(keys, v, task=None):
    a_pre, a_post, chg = [], [], []
    for key in keys:
        if task and key[0] != task: continue
        if key not in pred or v not in pred[key]["v"] or v not in postd[key]["v"]: continue
        g = postd[key]["gold"]
        p0 = pred[key]["v"][v]["pred"]; p1 = postd[key]["v"][v]["pred"]
        a_pre.append(float(p0 == g)); a_post.append(float(p1 == g)); chg.append(float(p0 != p1))
    if not a_pre: return None
    return {"pre": boot(a_pre), "post": boot(a_post), "delta": boot(np.array(a_post) - np.array(a_pre)),
            "pred_changed": boot(chg)}

res = {"replay_k_mismatches": mismatch, "variants": {}, "per_task_case_upper": {}, "touched_counts": {},
       "touched_examples": {k: v for k, v in examples.items()}}
allkeys = list(postd)
for v in variants:
    tk = [k for k in allkeys if touched[k].get(v)]
    un = [k for k in allkeys if v in touched[k] and not touched[k].get(v)]
    res["touched_counts"][v] = len(tk)
    d = {"all": compare(allkeys, v), "untouched": compare(un, v), "touched": compare(tk, v)}
    if v in RNG_VARS:
        d["exact_pairs_first72"] = compare([k for k in allkeys if k in first72], v)
        d["exact_pairs_first72_untouched"] = compare([k for k in un if k in first72], v)
    res["variants"][v] = d
for t in TASKS:
    res["per_task_case_upper"][t] = {"case_upper": compare(allkeys, "case_upper", t), "base": compare(allkeys, "base", t)}
    # post-fix: case_upper vs base (paired, same run)
    sel = [postd[k] for k in allkeys if k[0] == t]
    a = np.array([float(r["v"]["case_upper"]["pred"] == r["gold"]) for r in sel])
    b = np.array([float(r["v"]["base"]["pred"] == r["gold"]) for r in sel])
    fl = [float(r["v"]["case_upper"]["pred"] != r["v"]["base"]["pred"]) for r in sel]
    res["per_task_case_upper"][t]["post_upper_minus_base"] = boot(a - b)
    res["per_task_case_upper"][t]["post_flip_vs_base"] = boot(fl)
json.dump(res, open(OUT + "/compare.json", "w"), indent=1)

f = lambda x: "NA" if x is None or x[0] is None else "%.3f [%.3f,%.3f]" % tuple(x[:3])
print("\n== Pooled pre vs post per variant (paired by image; n=records)  touched = #records with an all-caps string")
print("%-19s %4s %3s  %-22s %-22s %-24s %-22s" % ("variant", "n", "tch", "pre acc", "post acc", "post-pre", "pred changed"))
for v in variants:
    d = res["variants"][v]["all"]
    print("%-19s %4d %3d  %-22s %-22s %-24s %-22s" % (v, d["pre"][3], res["touched_counts"][v], f(d["pre"]), f(d["post"]), f(d["delta"]), f(d["pred_changed"])))
print("\n== Untouched records only (no string rewritten -> model input identical pre/post except random draws)")
for v in variants:
    d = res["variants"][v]["untouched"]
    if d: print("%-19s n=%3d  post-pre %-24s pred changed %s" % (v, d["pre"][3], f(d["delta"]), f(d["pred_changed"])))
print("\n== Random-draw variants, first 72 records (identical draws pre/post)")
for v in sorted(RNG_VARS & set(variants)):
    d = res["variants"][v].get("exact_pairs_first72"); du = res["variants"][v].get("exact_pairs_first72_untouched")
    if d: print("%-10s n=%3d  post-pre %-24s pred changed %-22s | untouched n=%d changed %s" % (v, d["pre"][3], f(d["delta"]), f(d["pred_changed"]), du["pre"][3] if du else 0, f(du["pred_changed"]) if du else "NA"))
print("\n== case_upper per task")
for t in TASKS:
    d = res["per_task_case_upper"][t]
    print("%-12s base(post) %-22s upper pre %-22s upper post %-22s post-pre %-24s | post upper-base %-24s flip %s" % (
        t, f(d["base"]["post"]), f(d["case_upper"]["pre"]), f(d["case_upper"]["post"]), f(d["case_upper"]["delta"]),
        f(d["post_upper_minus_base"]), f(d["post_flip_vs_base"])))
print("\n== examples of touched strings in non-case_upper variants")
for v, ex in examples.items():
    if v != "case_upper": print(v, ex)

# ---------- merge everything into results.json (written by analyze.py) under "fix_verification"
import os
R = json.load(open(OUT + "/results.json"))
fv = dict(res); fv["note"] = ("pre = raw_main_prefix.json (records 73-300 used re-seeded random draws), post = raw_main.json "
                              "(clean single-seed run with laya_vision/textnorm.py all-caps normalisation)")
for nm in ("label_check", "drift_check", "residual_check"):
    p = "%s/%s.json" % (OUT, nm)
    if os.path.exists(p): fv[nm] = json.load(open(p))
R["fix_verification"] = fv
json.dump(R, open(OUT + "/results.json", "w"), indent=1, ensure_ascii=False)
print("merged fix_verification into results.json")
