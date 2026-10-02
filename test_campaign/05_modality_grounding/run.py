import json, os, sys, random, time, collections
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/zera/laya-vision")
import torch
import laya_vision
from laya_vision.eval.metrics import answer_probs

ROOT = "/home/zera/laya-vision"
OUT = ROOT + "/test_campaign/05_modality_grounding"
CKPT = os.environ.get("CKPT", ROOT + "/runs/stage2_c/wise085_calibrated")
TAG = os.environ.get("TAG", "final")
TASKS = ["cifar10", "eurosat", "food101", "oxford_pets", "coco", "aokvqa", "scienceqa", "vqav2_yesno"]
NIMG = {"vqav2_yesno": 240}   # images per task (default 100)
MAXQ = 3                       # questions kept per image
rng = random.Random(0)

# ---- sample
data = {}
for t in TASKS:
    by = collections.OrderedDict()
    for l in open("%s/data/%s.test.jsonl" % (ROOT, t)):
        r = json.loads(l); by.setdefault(r["image"], []).append(r)
    imgs = sorted(by); rng.shuffle(imgs); imgs = imgs[:NIMG.get(t, 50)]
    data[t] = [(im, by[im][:MAXQ]) for im in imgs]
    print(t, len(imgs), sum(len(x[1]) for x in data[t]), flush=True)

def load(p): return Image.open(ROOT + "/data/images/" + p).convert("RGB")
def size(p): return load(p).size

nprng = np.random.RandomState(0)
def solid(v): return lambda im, s: Image.new("RGB", s, (v, v, v))
def noise(im, s): return Image.fromarray(nprng.randint(0, 256, (s[1], s[0], 3), dtype=np.uint8))
def patch_shuffle(g):
    def f(im, s):
        im = im.resize((g * (224 // g) if 224 % g == 0 else 224, 224)) if False else im.resize((224, 224))
        a = np.asarray(im); h = 224 // g * g
        a = a[:h, :h]; k = h // g
        tiles = [a[i*k:(i+1)*k, j*k:(j+1)*k] for i in range(g) for j in range(g)]
        perm = nprng.permutation(len(tiles))
        out = np.zeros_like(a)
        for n, pidx in enumerate(perm):
            i, j = divmod(n, g); out[i*k:(i+1)*k, j*k:(j+1)*k] = tiles[pidx]
        return Image.fromarray(out)
    return f

# mismatched image assignment: within-task derangement over images; cross-task = image from another task
partner = {}; xpartner = {}
for ti, t in enumerate(TASKS):
    ims = [x[0] for x in data[t]]
    n = len(ims); sh = list(range(n)); 
    while True:
        rng.shuffle(sh)
        if all(sh[i] != i for i in range(n)): break
    for i, im in enumerate(ims): partner[(t, im)] = ims[sh[i]]
    ot = TASKS[(ti + 1) % len(TASKS)]
    oims = [x[0] for x in data[ot]]
    for i, im in enumerate(ims): xpartner[(t, im)] = oims[i % len(oims)]

CONDS = ["real", "black", "white", "noise", "patch4x4", "shuffle_within", "swap_cross_task", "text_only"]
torch.set_num_threads(4)
DEADLINE = float(os.environ.get("BUDGET", 2400))
agent = laya_vision.load(CKPT, device="cuda")
print("loaded", flush=True)

def run_state(state, qs):
    qd = {"q%d" % j: r["question"] for j, r in enumerate(qs)}
    for attempt in range(2):
        try:
            ans = agent.predict(state, qd)["answers"]; break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); time.sleep(5)
    return [(answer_probs(ans["q%d" % j], r["question"]), ans["q%d" % j].get("answer_confidence")) for j, r in enumerate(qs)]

# ---- resume: keep only fully completed (task,image) groups from an earlier (killed) run
rows = []
if os.path.exists("%s/raw_%s.json" % (OUT, TAG)):
    old = json.load(open("%s/raw_%s.json" % (OUT, TAG)))
    cnt = collections.Counter((r["task"], r["image"], r["cond"]) for r in old)
    full = {(t, im) for (t, im, c) in cnt if all(cnt[(t, im, cc)] > 0 for cc in CONDS)}
    # a group is complete only if every cond has the same #rows (= #questions kept)
    full = {g for g in full if len({cnt[g + (cc,)] for cc in CONDS}) == 1}
    rows = [r for r in old if (r["task"], r["image"]) in full]
    print("resume: kept", len(rows), "rows from", len(full), "image groups", flush=True)
DONE = {(r["task"], r["image"]) for r in rows}
t0 = time.time()
order = []
for r in range(240):
    for t in TASKS:
        k = 4 if t == "vqav2_yesno" else 1
        for j in range(k):
            i = r * k + j
            if i < len(data[t]): order.append((t,) + data[t][i])
for t, im, qs in order:
    if (t, im) in DONE: continue
    if True:
        if time.time() - t0 > DEADLINE: print("deadline hit", flush=True); break
        nprng = np.random.RandomState(abs(hash(("%s" % im))) % (2**31) if False else sum(map(ord, im)) % (2**31))
        text = qs[0].get("text")
        real = load(im); s = real.size
        pending = []
        for c in CONDS:
            if c == "real": pic = im_path = ROOT + "/data/images/" + im
            elif c == "black": pic = solid(0)(real, s)
            elif c == "white": pic = solid(255)(real, s)
            elif c == "gray": pic = solid(128)(real, s)
            elif c == "noise": pic = noise(real, s)
            elif c == "patch4x4": pic = patch_shuffle(4)(real, s)
            elif c == "patch8x8": pic = patch_shuffle(8)(real, s)
            elif c == "shuffle_within": pic = load(partner[(t, im)])
            elif c == "swap_cross_task": pic = load(xpartner[(t, im)])
            if c == "text_only":
                state = text if text not in (None, "") else "No image is available."
            else:
                state = {"image": pic} if text in (None, "") else {"image": pic, "text": text}
                if c == "real": state = {"image": pic} if text in (None, "") else {"image": pic, "text": text}
            res = run_state(state, qs); nst = globals().get("nst", 0) + 1; globals()["nst"] = nst
            if nst % 40 == 0: print("states", nst, round(time.time() - t0), "s", len(rows), "rows", flush=True)
            for r, (p, conf) in zip(qs, res):
                pending.append({"cond": c, "task": t, "image": im, "id": r["id"], "qtype": r["question"]["type"],
                             "k": len(r["target"]), "target": r["target"], "probs": p, "conf_answer": conf})
        rows.extend(pending)
        json.dump(rows, open("%s/raw_%s.tmp" % (OUT, TAG), "w")); os.replace("%s/raw_%s.tmp" % (OUT, TAG), "%s/raw_%s.json" % (OUT, TAG))
print("finished", len(rows), flush=True)
