import json, os, sys, time, io, random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
sys.path.insert(0, "/home/zera/laya-vision")
import laya_vision
from laya_vision.eval.metrics import answer_probs

ROOT = "/home/zera/laya-vision"
N = int(os.environ.get("N", 300))
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = os.environ.get("DEV", "cuda")

def load_recs(task):
    rs = [json.loads(l) for l in open(f"{ROOT}/data/{task}.test.jsonl")]
    random.Random(0).shuffle(rs)
    return rs[:N]

def rng(i, salt): return np.random.default_rng(1000 * salt + i)

def blur(r):
    return lambda im, i: im.filter(ImageFilter.GaussianBlur(r))
def noise(s):
    def f(im, i):
        a = np.asarray(im).astype(np.float32) + rng(i, 1).normal(0, s, (im.size[1], im.size[0], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f
def jpeg(q):
    def f(im, i):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f
def bright(k): return lambda im, i: ImageEnhance.Brightness(im).enhance(k)
def contrast(k): return lambda im, i: ImageEnhance.Contrast(im).enhance(k)
def gray(im, i): return ImageOps.grayscale(im).convert("RGB")
def zoom(frac):  # center crop keeping frac of side, resized back
    def f(im, i):
        w, h = im.size; cw, ch = int(w * frac), int(h * frac)
        l, t = (w - cw) // 2, (h - ch) // 2
        return im.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC)
    return f
def down(fac):
    def f(im, i):
        w, h = im.size
        return im.resize((max(1, int(w / fac)), max(1, int(h / fac))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return f
def occl(frac):  # black square covering frac of the area at random position
    def f(im, i):
        w, h = im.size; s = int(round((frac * w * h) ** 0.5)); s = min(s, w, h)
        r = rng(i, 2); x = int(r.integers(0, w - s + 1)); y = int(r.integers(0, h - s + 1))
        a = np.array(im); a[y:y + s, x:x + s] = 0; return Image.fromarray(a)
    return f

C = [("clean", "none", 0, lambda im, i: im)]
C.append(("rot90", "rot90", 1, lambda im, i: im.rotate(90, expand=True)))
C.append(("rot180", "rot180", 1, lambda im, i: im.rotate(180)))
C.append(("hflip", "hflip", 1, lambda im, i: ImageOps.mirror(im)))
C.append(("vflip", "vflip", 1, lambda im, i: ImageOps.flip(im)))
C.append(("grayscale", "grayscale", 1, gray))
for lv in (0, 1):
    C.append(("blur", "gaussian_blur_radius", [2, 6][lv], blur([2, 6][lv])))
    C.append(("noise", "gaussian_noise_sigma", [20, 70][lv], noise([20, 70][lv])))
    C.append(("jpeg", "jpeg_quality", [30, 5][lv], jpeg([30, 5][lv])))
    C.append(("brightness", "brightness_factor", [0.6, 0.25][lv], bright([0.6, 0.25][lv])))
    C.append(("contrast", "contrast_factor", [0.5, 0.2][lv], contrast([0.5, 0.2][lv])))
    C.append(("zoom", "center_crop_keep_frac", [0.7, 0.4][lv], zoom([0.7, 0.4][lv])))
    C.append(("downscale", "downscale_factor", [4, 12][lv], down([4, 12][lv])))
    C.append(("occlusion", "occluded_area_frac", [0.15, 0.4][lv], occl([0.15, 0.4][lv])))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]

def main():
    ck_path = f"{OUT}/ckpt_n{N}.json"
    ck = json.load(open(ck_path)) if os.path.exists(ck_path) else {"n": N, "seed": 0, "tasks": {"eurosat": {}, "oxford_pets": {}}, "preds": {}}
    res = ck
    t0 = time.time()
    todo = [(fam, param, lvl, fn, task) for fam, param, lvl, fn in C for task in res["tasks"] if f"{fam}@{lvl}" not in res["tasks"][task]]
    print("todo", len(todo), flush=True)
    agent = laya_vision.load(f"{ROOT}/runs/stage2_c/wise085_calibrated", device=DEV)
    data = {}
    for task in res["tasks"]:
        recs = load_recs(task)
        data[task] = (recs, [Image.open(f"{ROOT}/data/images/{r['image']}").convert("RGB") for r in recs])
    for fam, param, lvl, fn, task in todo:
        recs, imgs = data[task]
        correct, conf, goldp, preds = [], [], [], []
        for i, (r, im) in enumerate(zip(recs, imgs)):
            x = fn(im, i)
            with torch.no_grad():
                a = agent.predict({"image": x}, {"q0": r["question"]})["answers"]["q0"]
            p = np.array(answer_probs(a, r["question"])); y = int(np.argmax(r["target"]))
            preds.append(int(p.argmax())); correct.append(int(p.argmax() == y))
            conf.append(float(p.max())); goldp.append(float(p[y]))
        correct = np.array(correct); conf = np.array(conf); preds = np.array(preds)
        key = f"{fam}@{lvl}"
        if fam == "clean": res["preds"][task] = preds.tolist()
        cp = np.array(res["preds"][task])
        e = dict(family=fam, param=param, level=lvl, acc=float(correct.mean()), acc_ci=wilson(int(correct.sum()), len(correct)),
                 mean_conf=float(conf.mean()), conf_when_correct=float(conf[correct == 1].mean()) if correct.any() else None,
                 conf_when_wrong=float(conf[correct == 0].mean()) if (correct == 0).any() else None,
                 mean_gold_prob=float(np.mean(goldp)), pred_agree_with_clean=float((preds == cp).mean()),
                 n=len(correct), top_pred_share=float(np.bincount(preds).max() / len(preds)))
        res["tasks"][task][key] = e
        print(task, key, round(e["acc"], 3), round(e["mean_conf"], 3), round(time.time() - t0), flush=True)
        json.dump(res, open(ck_path + ".tmp", "w")); os.replace(ck_path + ".tmp", ck_path)
    out = {k: v for k, v in res.items() if k != "preds"}
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
