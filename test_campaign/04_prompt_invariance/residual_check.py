"""(a) case_upper strings left all-caps by textnorm (no word >= 4 letters), per task.
(b) Padding hypothesis for the tiny conf drift in unchanged variants: does the fix change the
longest question (token proxy) in the record's predict() batch?"""
import json, sys, collections, numpy as np
sys.argv = [sys.argv[0], "50"]
OUT = "/home/zera/laya-vision/test_campaign/04_prompt_invariance"
src = open(OUT + "/run.py").read().rsplit("\nmain()", 1)[0]
ns = {"__file__": OUT + "/run.py", "__name__": "replay"}
exec(compile(src, "run.py", "exec"), ns)
from laya_vision.textnorm import is_shouted, unshout
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ns["CKPT"] + "/tokenizer")
pre = {(r["task"], r["id"]): r for r in json.load(open(OUT + "/raw_main_prefix.json"))}
post = {(r["task"], r["id"]): r for r in json.load(open(OUT + "/raw_main.json"))}
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
def ntok(q, norm):
    f = unshout if norm else (lambda s: s)
    s = [f(q["instructions"])] + [f(k) + " " + (f(d) if isinstance(d, str) else "") for k, d in q["criteria"].items()]
    return sum(len(tok.tokenize(" " + x)) for x in s)
left = collections.defaultdict(lambda: [0, 0, 0]); ex = collections.defaultdict(list)
drift = []
for task, r, pool in order:
    V = ns["build_variants"](r, task, pool, rng)[0]
    rec = post[(task, r["id"])]
    gold = int(np.argmax(r["target"])); bp = rec["v"]["base"]["pred"]
    others = [i for i in range(rec["K"]) if i not in {gold, bp}]; rng.shuffle(others)  # keep rng in sync
    q = V["case_upper"][0]
    strs = [q["instructions"]] + [k for k in q["criteria"]] + [d for d in q["criteria"].values() if isinstance(d, str)]
    up = [s for s in strs if s == s.upper() and s != s.lower()]
    nl = [s for s in up if not is_shouted(s)]
    left[task][0] += len(nl); left[task][1] += len(up); left[task][2] += bool(nl)
    if nl and len(ex[task]) < 6: ex[task] += nl[:2]
    L_raw = max(ntok(v[0], False) for v in V.values()); L_norm = max(ntok(v[0], True) for v in V.values())
    dc = max(abs(pre[(task, r["id"])]["v"][v]["conf"] - rec["v"][v]["conf"]) for v in ("base", "case_lower", "opt_long", "para_ood0", "rename_number", "punct_strip"))
    drift.append((L_raw != L_norm, dc > 0))
out = {"case_upper_strings_not_rewritten": {t: {"n_left": a, "n_allcaps": b, "records_with_left": c, "examples": ex[t]} for t, (a, b, c) in left.items()}}
d = np.array(drift)
out["padding_proxy"] = {"records_longest_question_changes": int(d[:, 0].sum()),
                        "conf_drift_records": int(d[:, 1].sum()),
                        "conf_drift_and_longest_changes": int((d[:, 0] & d[:, 1]).sum()),
                        "conf_drift_but_longest_unchanged": int((~d[:, 0] & d[:, 1]).sum())}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(OUT + "/residual_check.json", "w"), indent=1, ensure_ascii=False)
