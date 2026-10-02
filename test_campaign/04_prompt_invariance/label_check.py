"""Verify that under case_upper the returned choice labels are the caller's upper-case labels,
and show what the tokenizer sees after textnorm. Writes label_check.json."""
import json, sys, random, numpy as np
sys.argv = [sys.argv[0], "50"]
OUT = "/home/zera/laya-vision/test_campaign/04_prompt_invariance"
src = open(OUT + "/run.py").read().rsplit("\nmain()", 1)[0]
ns = {"__file__": OUT + "/run.py", "__name__": "replay"}
exec(compile(src, "run.py", "exec"), ns)
import laya_vision
from laya_vision.textnorm import unshout_question, is_shouted

agent = laya_vision.load(ns["CKPT"], device="cuda")
tok = agent.tok
out = []
for task in ns["TASKS"]:
    rows = ns["load_task"](task)
    pool = sorted({(v if task in ns["LETTER"] else k) for r in rows for k, v in r["question"]["criteria"].items() if (v if task in ns["LETTER"] else k)})
    rr = random.Random(7); picks = rr.sample(rows, 2)
    for r in picks:
        V, items0, rep, text = ns["build_variants"](r, task, pool, random.Random(0))
        q, m, gold = V["case_upper"]
        state = {"image": ns["D"] + "images/" + r["image"]}
        if r.get("text"): state["text"] = r["text"]
        a = agent.predict(state, {"q": q, "b": V["base"][0]})["answers"]
        keys = list(q["criteria"])
        ch = a["q"]["choice"]
        internal_like = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
        norm = unshout_question(internal_like)
        first = keys[gold] if task not in ns["LETTER"] else q["criteria"][keys[gold]]
        first_n = list(norm["crit"])[gold] if task not in ns["LETTER"] else list(norm["crit"].values())[gold]
        rec = {"task": task, "id": r["id"], "caller_keys": keys[:6], "returned_choice": ch,
               "returned_in_caller_keys": ch in keys, "returned_is_upper": ch == ch.upper(),
               "correct": m[ch] == gold, "base_choice": a["b"]["choice"],
               "same_option_as_base": m[ch] == V["base"][1][a["b"]["choice"]],
               "ins_seen_by_model": norm["ins"][:90],
               "gold_text_tokens_raw": tok.tokenize(" " + first)[:10],
               "gold_text_tokens_normalised": tok.tokenize(" " + first_n)[:10]}
        out.append(rec); print(json.dumps(rec, ensure_ascii=False))
json.dump(out, open(OUT + "/label_check.json", "w"), indent=1, ensure_ascii=False)
print("all returned labels are caller keys:", all(x["returned_in_caller_keys"] for x in out),
      "| all upper:", all(x["returned_is_upper"] for x in out))
