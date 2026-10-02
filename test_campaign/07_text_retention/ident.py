"""Text-only identity check: VisionAgent vs laya.Agent loaded from the SAME checkpoint dir, 60 records, batched path + predict()."""
import sys, os, json, random
sys.path.insert(0, "/home/zera/laya-vision"); os.chdir("/home/zera/laya-vision")
import torch, numpy as np, laya, laya_vision
from laya_vision.eval.run_eval import read_jsonl
from laya_vision.eval.metrics import answer_probs
ck = sys.argv[1]
recs = []
for t in ["ag_news", "boolq", "typed_decisions"]:
    r = read_jsonl(["data/%s.test.jsonl" % t]); random.Random(1).shuffle(r); recs += r[:20]
va = laya_vision.load(ck, device="cuda")
def go(ag):
    out = []
    with torch.no_grad():
        for r in recs: out.append(answer_probs(ag.predict(r["text"], {"q": r["question"]})["answers"]["q"], r["question"]))
    return out
A = go(va); del va; torch.cuda.empty_cache()
la = laya.Agent(ck, device="cuda"); B = go(la)
d = max(np.abs(np.array(a) - np.array(b)).max() for a, b in zip(A, B))
res = dict(checkpoint=ck, n=len(recs), max_abs_prob_diff=float(d), identical_argmax=all(np.argmax(a) == np.argmax(b) for a, b in zip(A, B)),
           bit_identical=all(list(a) == list(b) for a, b in zip(A, B)))
print(res); json.dump(res, open("test_campaign/07_text_retention/identity.json", "w"), indent=1)
