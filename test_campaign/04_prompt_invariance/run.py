"""Prompt-invariance campaign. Usage: python run.py [N_per_task] [ckpt]"""
import json, random, sys, time, re, string, collections, os
import numpy as np, torch
sys.path.insert(0, "/home/zera/laya-vision")
import laya_vision
from laya_vision.data.augment import TEMPLATES, _choice_keys, KEY_STYLES

OUT = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
CKPT = sys.argv[2] if len(sys.argv) > 2 else "/home/zera/laya-vision/runs/stage2_c/wise085_calibrated"
TAG = sys.argv[3] if len(sys.argv) > 3 else "main"
TASKS = ["scienceqa", "aokvqa", "cifar10", "food101", "oxford_pets", "eurosat"]
LETTER = {"scienceqa", "aokvqa"}
D = "/home/zera/laya-vision/data/"
rng = random.Random(0)

def load_task(t):
    rows = [json.loads(l) for l in open(D + t + ".test.jsonl")]
    rows = [r for r in rows if r["question"]["type"] == "choice"]
    return rows

def Q(ins, opts):  # opts: list of (oid, key, desc)
    crit = {}
    for oid, k, d in opts:
        assert k not in crit, (k, opts)
        crit[k] = d
    return {"type": "choice", "instructions": ins, "criteria": crit}, {k: oid for oid, k, d in opts}

def letters(n):
    return _choice_keys(n, "upper")

OOD = [
    lambda s: "Look carefully at the image and answer the following. " + s,
    lambda s: "Q: " + s + " Select the single best option.",
    lambda s: "Please choose the correct answer. " + s,
]

def build_variants(r, task, pool, rng):
    """returns dict name -> (question, key->oid, gold_oid). oid = original option index or 'dN'."""
    q = r["question"]; crit = q["criteria"]; ins = q["instructions"]
    base_items = list(crit.items())
    K = len(base_items)
    gold = int(np.argmax(r["target"]))
    letter = task in LETTER
    # canonical option list: (oid, name/text, desc). For letter tasks text is desc; for classify text is key.
    text = [(d if letter else k) for k, d in base_items]
    V = {}
    def rep(items, ins_=ins):  # items: (oid, text) -> question in the task's native format
        if letter:
            ks = letters(len(items))
            return Q(ins_, [(oid, k, t) for k, (oid, t) in zip(ks, items)])
        return Q(ins_, [(oid, t, None) for oid, t in items])
    items0 = list(enumerate(text))
    # base: exactly the stored question
    V["base"] = (q, {k: i for i, (k, d) in enumerate(base_items)}, gold)
    # --- option order
    perms = []
    for j in range(4):
        p = list(range(K)); rng.shuffle(p); perms.append(p)
    perms.append(list(range(K))[::-1])
    if K > 1:
        perms.append(list(range(K))[1:] + [0])  # cyclic shift
    for j, p in enumerate(perms):
        qq, m = rep([items0[i] for i in p]); V["perm%d" % j] = (qq, m, gold)
    # --- label renaming (keys change, old key moves into description when there is none)
    for st in KEY_STYLES:
        ks = _choice_keys(K, st)
        opts = [(i, nk, (str(k) if (d is None or d == "") else d)) for nk, (i, (k, d)) in zip(ks, enumerate(base_items))]
        qq, m = Q(ins, opts); V["rename_" + st] = (qq, m, gold)
    # label = answer text, no description (letter tasks only)
    if letter:
        qq, m = Q(ins, [(i, t, None) for i, t in items0]); V["rename_text_as_key"] = (qq, m, gold)
    # --- paraphrase
    key = r.get("template") or r.get("task"); flds = r.get("fields") or {}
    tpl = TEMPLATES.get(key)
    if tpl:
        alts = []
        for i, tp in enumerate(tpl):
            try:
                s = tp.format(**flds)
            except (KeyError, IndexError):
                continue
            if s != ins: alts.append(s)
        for j, s in enumerate(alts[:2]):
            qq, m = rep(items0, s); V["para_indist%d" % j] = (qq, m, gold)
    for j, f in enumerate(OOD):
        qq, m = rep(items0, f(ins)); V["para_ood%d" % j] = (qq, m, gold)
    # --- casing / punctuation
    def tr(fi, ft):
        qq, m = rep([(i, ft(t)) for i, t in items0], fi(ins)); return qq, m, gold
    def nopunct(s): return re.sub(r"[%s]" % re.escape(string.punctuation), "", s)
    try:
        V["case_lower"] = tr(str.lower, str.lower)
        V["case_upper"] = tr(str.upper, str.upper)
        V["case_title"] = tr(lambda s: s, lambda s: s.title())
        V["punct_strip"] = tr(lambda s: nopunct(s).strip(), lambda s: nopunct(s).strip())
        V["punct_add"] = tr(lambda s: s.rstrip("?.") + "?!" if False else s.rstrip("?.") + " ?", lambda s: s.rstrip(".") + ".")
    except AssertionError:
        pass
    # --- long vs short option text
    if letter:
        qq, m = rep([(i, "It is %s, as can be determined from looking carefully at the picture" % t) for i, t in items0])
        V["opt_long"] = (qq, m, gold)
    else:
        opts = [(i, k, "a photograph in which the main subject is clearly %s, shown as the central object" % k) for i, (k, d) in enumerate(base_items)]
        qq, m = Q(ins, opts); V["opt_long"] = (qq, m, gold)
        if any(d for k, d in base_items):
            qq, m = Q(ins, [(i, k, None) for i, (k, d) in enumerate(base_items)]); V["opt_short"] = (qq, m, gold)
    # --- distractor add / count scaling
    avail = [t for t in pool if t not in set(text)]
    def with_k(k):  # gold + (k-1) distractors drawn from original wrong options, then pool
        wrong = [i for i in range(K) if i != gold]
        rng.shuffle(wrong)
        sel = [gold] + wrong[:k - 1]
        items = [items0[i] for i in sel]
        extra = k - len(items)
        if extra > 0:
            if extra > len(avail): return None
            items += [("d%d" % j, t) for j, t in enumerate(rng.sample(avail, extra))]
        rng.shuffle(items)
        return rep(items)
    for k in (2, 3, 4, 5, 8, 10, 15, 20):
        if letter and k > 8: continue
        v = with_k(k)
        if v: V["k%02d" % k] = (v[0], v[1], gold)
    # add distractors to base set: keep all base options, add up to 4 (cap 20)
    ad = min(4, 20 - K, len(avail))
    if ad > 0:
        items = list(items0) + [("d%d" % j, t) for j, t in enumerate(rng.sample(avail, ad))]
        rng.shuffle(items)
        qq, m = rep(items); V["add_dist"] = (qq, m, gold)
    drop = {'perm4','perm5','rename_lower','rename_paren','para_indist1','case_title','punct_add','k03','k04','k08','k15'}
    V = {k: v for k, v in V.items() if k not in drop}
    # remove-half distractors is built after base prediction (needs pred) -> handled in main
    return V, items0, rep, text

def main():
    t0 = time.time()
    agent = laya_vision.load(CKPT, device="cuda"); print("loaded %.0fs" % (time.time()-t0), flush=True)
    res = []
    rawp = "%s/raw_%s.json" % (OUT, TAG)
    if os.path.exists(rawp):
        res = json.load(open(rawp)); print("resumed", len(res), flush=True)
    done = {(x["task"], x["id"]) for x in res}
    per = {}
    for task in TASKS:
        rows = load_task(task)
        pool = sorted({(v if task in LETTER else k) for r in rows for k, v in r["question"]["criteria"].items() if (v if task in LETTER else k)})
        if task in LETTER: pool = pool  # answer texts from other questions
        # one question per image
        seen = set(); uniq = []
        for r in rows:
            if r["image"] not in seen: seen.add(r["image"]); uniq.append(r)
        rr = random.Random(1); rr.shuffle(uniq); uniq = uniq[:N]
        per[task] = (uniq, pool)
    order = [(t, per[t][0][i], per[t][1]) for i in range(N) for t in TASKS if i < len(per[t][0])]
    for n, (task, r, pool) in enumerate(order):
        if (task, r["id"]) in done: continue
        if True:
            gold = int(np.argmax(r["target"]))
            V, items0, rep, text = build_variants(r, task, pool, rng)
            state = {"image": D + "images/" + r["image"]}
            if r.get("text"): state["text"] = r["text"]
            names = list(V)
            out = {}
            def run(names_):
                for c in range(0, len(names_), 24):
                    ch = names_[c:c + 24]
                    qs = {nm: V[nm][0] for nm in ch}
                    try:
                        ans = agent.predict(state, qs)["answers"]
                    except Exception as e:
                        print("ERR", r["id"], repr(e)[:200]); continue
                    for nm in ch:
                        a = ans[nm]; pk = a["choice"]
                        out[nm] = {"pred": V[nm][1][pk], "conf": a["answer_confidence"], "k": len(V[nm][0]["criteria"]),
                                   "pred_pos": list(V[nm][0]["criteria"]).index(pk), "gold_pos": list(V[nm][0]["criteria"]).index([k for k, o in V[nm][1].items() if o == V[nm][2]][0])}
            print(task, n, len(names), "%.0fs" % (time.time()-t0), flush=True); run(names)
            # drop-half-of-distractors keeping the base prediction (+ gold)
            bp = out["base"]["pred"] if "base" in out else None
            if bp is not None:
                keep = {gold, bp}
                others = [i for i in range(len(items0)) if i not in keep]
                rng.shuffle(others)
                keep |= set(others[:len(others) // 2])
                if len(keep) >= 2 and len(keep) < len(items0):
                    items = [items0[i] for i in sorted(keep)]
                    qq, m = rep(items)
                    V["drop_half"] = (qq, m, gold); run(["drop_half"])
            res.append({"task": task, "id": r["id"], "K": len(items0), "gold": gold, "v": out})
            if len(res) % 6 == 0:
                print('prog', len(res), "%.0fs" % (time.time() - t0), flush=True)
                json.dump(res, open(rawp + ".tmp", "w")); os.replace(rawp + ".tmp", rawp)
    json.dump(res, open("%s/raw_%s.json" % (OUT, TAG), "w"))
    print("done %.0fs" % (time.time() - t0))

main()
