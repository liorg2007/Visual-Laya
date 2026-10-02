import json, random, math, os, sys, collections, time
import numpy as np, torch
sys.path.insert(0, "/home/zera/laya-vision")
import laya_vision
from laya_vision.eval.metrics import answer_probs
R = "/home/zera/laya-vision"; OUT = R + "/test_campaign/06_generalization"
N = int(os.environ.get("N", 300)); SEED = 0
def load(t): return [json.loads(l) for l in open(f"{R}/data/{t}.test.jsonl")]
def gold_of(r): return max(range(len(r["target"])), key=lambda i: r["target"][i])
def mk(keys, vals, gi, instr):  # choice question, target
    crit = {k: v for k, v in zip(keys, vals)}
    t = [0.0]*len(keys); t[gi] = 1.0
    return {"type": "choice", "instructions": instr, "criteria": crit}, t
rng = random.Random(SEED)
def sample_imgs(rows, n):
    ch = [r for r in rows if r["question"]["type"] == "choice"]
    rng.shuffle(ch); return ch[:n]
L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
items = []  # (image, variant_key, task, seen_status, k, question, target)
def add(r, var, task, status, q, t): items.append(dict(image=r["image"], var=var, task=task, status=status, q=q, t=t, k=len(t), base=r["id"]))

def labels_of(r): return list(r["question"]["criteria"].keys())
def distract(r, k, pool=None):
    labs = labels_of(r); g = labs[gold_of(r)]
    others = [l for l in (pool or labs) if l != g]; rng.shuffle(others)
    opts = [g] + others[:k-1]; rng.shuffle(opts); return opts, opts.index(g)

SYN = {"airplane": "aircraft", "automobile": "car", "ship": "boat", "truck": "lorry", "cat": "kitten", "dog": "puppy",
       "bird": "feathered flyer", "deer": "stag", "frog": "toad", "horse": "pony"}
cif = sample_imgs(load("cifar10"), N)
CPOOL = list(SYN)
for r in cif:
    labs = labels_of(r); g = labs[gold_of(r)]; I = r["question"]["instructions"]
    q, t = r["question"], r["target"]; add(r, "orig(10 opt)", "cifar10", "seen-task, held-out images", q, t)
    for k in (2, 3, 6, 8):
        o, gi = distract(r, k); q, t = mk(o, [None]*k, gi, I); add(r, f"{k} options", "cifar10", "unseen option count", q, t)
    o = [SYN[l] for l in labs]; q, t = mk(o, [None]*10, labs.index(g), I); add(r, "synonym labels", "cifar10", "novel vocabulary", q, t)
    o = [f"a photo of a {l}" for l in labs]; q, t = mk(o, [None]*10, labs.index(g), I); add(r, "'a photo of a X' labels", "cifar10", "novel vocabulary", q, t)
    q, t = mk(list(L[:10]), labs, labs.index(g), I); add(r, "letter MCQ keys A-J", "cifar10", "novel format", q, t)
    q, t = mk(list("1234567890"), labs, labs.index(g), I); add(r, "digit keys", "cifar10", "novel format", q, t)
    animal = g in ("bird", "cat", "deer", "dog", "frog", "horse")
    q, t = mk(["animal", "vehicle"], [None, None], 0 if animal else 1, "Is this an animal or a vehicle?"); add(r, "animal vs vehicle (new question)", "cifar10", "novel question", q, t)
    q, t = mk(["living thing", "man-made machine"], [None, None], 0 if animal else 1, "What kind of thing is shown?"); add(r, "living vs machine (new q, new words)", "cifar10", "novel question", q, t)

fd = load("food101"); fr = sample_imgs(fd, N)
allfood = sorted({labels_of(r)[gold_of(r)] for r in fd if r["question"]["type"] == "choice"})
for r in fr:
    labs = labels_of(r); g = labs[gold_of(r)]; I = r["question"]["instructions"]
    add(r, "orig(20 opt)", "food101", "seen-task, held-out images", r["question"], r["target"])
    for k in (3, 8, 12):
        o, gi = distract(r, k); q, t = mk(o, [None]*k, gi, I); add(r, f"{k} options", "food101", "unseen option count", q, t)
    for k in (30,):
        o, gi = distract(r, k, pool=allfood); q, t = mk(o, [None]*k, gi, I); add(r, f"{k} options", "food101", "unseen option count (>20)", q, t)
    o, gi = distract(r, 4); q, t = mk(list(L[:4]), o, gi, I); add(r, "letter MCQ, 4 opt", "food101", "novel format", q, t)
    o, gi = distract(r, 20); q, t = mk(list(L[:20]), o, gi, I); add(r, "letter MCQ, 20 opt", "food101", "novel format", q, t)
    o, gi = distract(r, 8); o2 = ["homemade " + x for x in o]; q, t = mk(o2, [None]*8, gi, I); add(r, "'homemade X' labels, 8 opt", "food101", "novel vocabulary", q, t)

pets = load("oxford_pets"); pr = sample_imgs(pets, N)
for r in pr:
    labs = labels_of(r); g = labs[gold_of(r)]; I = r["question"]["instructions"]
    add(r, "orig(20 opt)", "oxford_pets", "seen-task, held-out images", r["question"], r["target"])
    for k in (3, 10):
        o, gi = distract(r, k); q, t = mk(o, [None]*k, gi, I); add(r, f"{k} options", "oxford_pets", "unseen option count", q, t)
    cat = os.path.basename(r["image"])[0].isupper()
    q, t = mk(["cat", "dog"], [None, None], 0 if cat else 1, "Is this a cat or a dog?"); add(r, "cat vs dog (new question)", "oxford_pets", "novel question", q, t)

ao = sample_imgs(load("aokvqa"), N)
for r in ao:
    c = r["question"]["criteria"]; keys = list(c); vals = list(c.values()); gi = gold_of(r); I = r["question"]["instructions"]
    add(r, "orig(4 opt)", "aokvqa", "seen-task, held-out images", r["question"], r["target"])
    perm = list(range(4)); rng.shuffle(perm); q, t = mk(keys, [vals[i] for i in perm], perm.index(gi), I); add(r, "options shuffled", "aokvqa", "seen-task, perturbed", q, t)
    wrong = [i for i in range(4) if i != gi]; rng.shuffle(wrong)
    for k in (2, 3):
        idx = [gi] + wrong[:k-1]; rng.shuffle(idx); q, t = mk(list(L[:k]), [vals[i] for i in idx], idx.index(gi), I); add(r, f"{k} options", "aokvqa", "unseen option count", q, t)
    q, t = mk(list("wxyz"), vals, gi, I); add(r, "keys w/x/y/z", "aokvqa", "novel format", q, t)

sq = sample_imgs(load("scienceqa"), N)
for r in sq:
    c = r["question"]["criteria"]; keys = list(c); vals = list(c.values()); gi = gold_of(r); I = r["question"]["instructions"]
    add(r, "orig(2-5 opt)", "scienceqa", "seen-task, held-out images", r["question"], r["target"])
    q, t = mk(list("wxyz")[:len(keys)], vals, gi, I); add(r, "keys w/x/y/z", "scienceqa", "novel format", q, t)

es = sample_imgs(load("eurosat"), N)
ESYN = {"Annual Crop": "arable farmland", "Forest": "woodland", "Herbaceous Vegetation": "meadow and shrubs", "Highway": "motorway",
        "Industrial Buildings": "factories", "Pasture": "grazing land", "Permanent Crop": "orchards and vineyards",
        "Residential Buildings": "housing", "River": "stream", "Sea or Lake": "open water"}
for r in es:
    c = r["question"]["criteria"]; keys = list(c); gi = gold_of(r); I = r["question"]["instructions"]
    add(r, "orig(10 opt)", "eurosat", "seen-task, held-out images", r["question"], r["target"])
    q, t = mk(keys, [ESYN.get(k, v) for k, v in zip(keys, c.values())], gi, I); add(r, "synonym descriptions", "eurosat", "novel vocabulary", q, t)
    o, gi2 = distract(r, 3); q, t = mk(o, [c[x] for x in o], gi2, I); add(r, "3 options", "eurosat", "unseen option count", q, t)

co = sample_imgs(load("coco"), N)
cap_pool = [v for r in co for v in r["question"]["criteria"].values()]
for r in co:
    c = r["question"]["criteria"]; keys = list(c); vals = list(c.values()); gi = gold_of(r); I = r["question"]["instructions"]
    add(r, f"orig({len(keys)} opt)", "coco", "stage-1 only (not in stage-2 mix)", r["question"], r["target"])
    wrong = [i for i in range(len(keys)) if i != gi]; rng.shuffle(wrong)
    for k in (2, 3):
        idx = [gi] + wrong[:k-1]; rng.shuffle(idx); q, t = mk(list(L[:k]), [vals[i] for i in idx], idx.index(gi), I); add(r, f"{k} options", "coco", "stage-1 only; unseen option count", q, t)
    extra = rng.sample([x for x in cap_pool if x not in vals], 10 - len(keys))
    allv = vals + extra; idx = list(range(len(allv))); rng.shuffle(idx); q, t = mk(list(L[:10]), [allv[i] for i in idx], idx.index(gi), I); add(r, "10 options", "coco", "stage-1 only; unseen option count", q, t)
print("items", len(items), flush=True)

dev = "cuda" if torch.cuda.is_available() and os.environ.get("CPU") != "1" else "cpu"
agent = laya_vision.load(R + "/runs/stage2_c/wise085_calibrated", device=dev)
groups = collections.OrderedDict()
for it in items: groups.setdefault((it["image"], it["task"]), []).append(it)
print("loaded", flush=True)
t0 = time.time(); rows = []
PART = OUT + "/partial.json"; done = set()
if os.path.exists(PART):
    st = json.load(open(PART)); rows = st["rows"]; done = set(map(tuple, st["done"]))
for gi_, ((img, task), its) in enumerate(groups.items()):
    if (img, task) in done: continue
    state = {"image": f"{R}/data/images/{img}"}
    for c in range(0, len(its), 16):
        ch = its[c:c+16]
        try:
            with torch.no_grad(): ans = agent.predict(state, {f"q{j}": it["q"] for j, it in enumerate(ch)})["answers"]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print("OOM", img); continue
        for j, it in enumerate(ch):
            a = ans[f"q{j}"]; p = answer_probs(a, it["q"])
            rows.append(dict(task=it["task"], var=it["var"], status=it["status"], k=it["k"], base=it["base"],
                             correct=int(int(np.argmax(p)) == int(np.argmax(it["t"]))), conf=float(max(p)), gold=int(np.argmax(it["t"])), pred=int(np.argmax(p))))
    done.add((img, task))
    if gi_ % 20 == 0: json.dump({'rows': rows, 'done': sorted(done)}, open(PART, 'w'))
    if gi_ % 10 == 0: print(gi_, len(groups), time.time()-t0, flush=True)
json.dump(rows, open(OUT + "/raw_rows.json", "w"))
