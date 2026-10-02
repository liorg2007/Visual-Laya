"""Attribute the small confidence drift in unchanged variants (ScienceQA records 72+) by re-running
those records' exact predict() batches with the fix ON and with it monkey-patched OFF (identity),
and comparing to the stored pre-fix and post-fix confidences. No repo source is edited."""
import json, sys, numpy as np
sys.argv = [sys.argv[0], "50"]
OUT = "/home/zera/laya-vision/test_campaign/04_prompt_invariance"
src = open(OUT + "/run.py").read().rsplit("\nmain()", 1)[0]
ns = {"__file__": OUT + "/run.py", "__name__": "replay"}
exec(compile(src, "run.py", "exec"), ns)
import laya_vision, laya_vision.agent as LA
pre = {(r["task"], r["id"]): r for r in json.load(open(OUT + "/raw_main_prefix.json"))}
post = {(r["task"], r["id"]): r for r in json.load(open(OUT + "/raw_main.json"))}
WANT = {78, 90, 120, 156, 192, 6, 12}  # 5 drift records + 2 non-drift ScienceQA controls (first 72)
rng = ns["rng"]; TASKS = ns["TASKS"]; LETTER = ns["LETTER"]
per = {}
for task in TASKS:
    rows = ns["load_task"](task)
    pool = sorted({(v if task in LETTER else k) for r in rows for k, v in r["question"]["criteria"].items() if (v if task in LETTER else k)})
    seen = set(); uniq = []
    for r in rows:
        if r["image"] not in seen: seen.add(r["image"]); uniq.append(r)
    rr = ns["random"].Random(1); rr.shuffle(uniq); per[task] = (uniq[:50], pool)
order = [(t, per[t][0][i], per[t][1]) for i in range(50) for t in TASKS]
agent = laya_vision.load(ns["CKPT"], device="cuda")
ON = (LA.unshout_state, LA.unshout_question)
def predict(state, V, names, fix):
    LA.unshout_state, LA.unshout_question = ON if fix else ((lambda s: s), (lambda q: q))
    out = {}
    for c in range(0, len(names), 24):
        ch = names[c:c + 24]
        a = agent.predict(state, {nm: V[nm][0] for nm in ch})["answers"]
        out.update({nm: a[nm]["answer_confidence"] for nm in ch})
    return out
CMP = ["base", "case_lower", "opt_long", "para_ood0", "para_indist0", "rename_number", "punct_strip"]
res = []
for n, (task, r, pool) in enumerate(order):
    gold = int(np.argmax(r["target"]))
    V, items0, rep, text = ns["build_variants"](r, task, pool, rng)
    rec = post[(task, r["id"])]; bp = rec["v"]["base"]["pred"]
    others = [i for i in range(len(items0)) if i not in {gold, bp}]; rng.shuffle(others)
    if n not in WANT: continue
    state = {"image": ns["D"] + "images/" + r["image"]}
    if r.get("text"): state["text"] = r["text"]
    names = list(V)
    on1, on2, off = predict(state, V, names, True), predict(state, V, names, True), predict(state, V, names, False)
    p0 = pre[(task, r["id"])]["v"]; p1 = rec["v"]
    row = {"idx": n, "task": task,
           "max|on-post_stored|": max(abs(on1[v] - p1[v]["conf"]) for v in CMP if v in on1),
           "max|on-on_repeat|": max(abs(on1[v] - on2[v]) for v in CMP if v in on1),
           "max|off-post_stored|": max(abs(off[v] - p1[v]["conf"]) for v in CMP if v in off),
           "max|off-pre_stored|": max(abs(off[v] - p0[v]["conf"]) for v in CMP if v in off),
           "max|post-pre_stored|": max(abs(p1[v]["conf"] - p0[v]["conf"]) for v in CMP if v in p1)}
    res.append(row); print(json.dumps(row), flush=True)
json.dump(res, open(OUT + "/drift_check.json", "w"), indent=1)
