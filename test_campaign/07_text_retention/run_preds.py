"""Run text tasks (fixed-seed n<=1000 each) for one model spec; write preds/<name>.jsonl.
usage: run_preds.py NAME KIND SPEC [alpha...]   KIND in stock|vision|layaagent|wise (wise: SPEC=finetuned ckpt dir, alphas in-memory)"""
import sys, os, json, random, time, gc
sys.path.insert(0, "/home/zera/laya-vision")
os.chdir("/home/zera/laya-vision")
import torch
from laya_vision.eval.run_eval import read_jsonl
from laya_vision.eval.metrics import answer_probs
from laya.agent import collate_items
BS = int(os.environ.get("BS", 16))
@torch.no_grad()
def run_records(ag, recs, *_):
    rows = []
    for s in range(0, len(recs), BS):
        chunk = recs[s:s + BS]; ids = ["q"]
        internals = [{"q": ag._to_internal(r["question"])} for r in chunk]
        enc = [ag._encode_state(r["text"], ids, internals[j]) for j, r in enumerate(chunk)]
        b = collate_items(enc, ag.tok.pad_token_id)
        logits, act = ag._forward(b)
        row = 0
        for j, r in enumerate(chunk):
            a = ag._decode_answers(logits, act, enc[j], ids, internals[j], row)["q"]; row += len(enc[j])
            rows.append({"id": r["id"], "task": r["task"], "qtype": r["question"]["type"], "target": r["target"],
                         "probs": answer_probs(a, r["question"])})
        if (s // BS) % 10 == 0: print("[batch] %d/%d" % (s, len(recs)), flush=True)
    return rows
SEED, N = 0, int(os.environ.get("NN", 400))
def sample():
    recs = []
    for t in ["ag_news", "boolq", "typed_decisions"]:
        r = read_jsonl(["data/%s.test.jsonl" % t])
        random.Random(SEED).shuffle(r)
        recs += r[:N]
    return recs
STOCK_REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
def dump(name, rows, extra=None):
    with open("test_campaign/07_text_retention/preds/%s.jsonl" % name, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print("done", name, len(rows), sum(r["probs"] is None for r in rows), "failed", flush=True)
def main():
    name, kind, spec = sys.argv[1:4]
    recs = sample(); dev = "cuda"
    if kind == "stock":
        import laya; ag = laya.Agent("convaiinnovations/laya", device=dev, revision=STOCK_REV)
    elif kind == "layaagent":
        import laya; ag = laya.Agent(spec, device=dev)
    elif kind == "vision":
        import laya_vision; ag = laya_vision.load(spec, device=dev)
    if kind != "wise":
        print(name, "temps", ag.temperature, ag.temperature_by_options)
        dump(name, run_records(ag, recs, None, 32, 100)); return
    import laya_vision
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    stock = load_file(os.path.join(snapshot_download("convaiinnovations/laya", revision=STOCK_REV), "model.safetensors"))
    ft = load_file(os.path.join(spec, "model.safetensors"))
    ag = laya_vision.load(spec, device=dev)
    print("ft cfg temps", ag.temperature, ag.temperature_by_options)
    for a in [float(x) for x in sys.argv[4:]]:
        mix = {k: (stock[k].float() + a * (ft[k].float() - stock[k].float())).to(ft[k].dtype) if ft[k].is_floating_point() else ft[k] for k in ft}
        ag.model.load_state_dict({k: v.to(dev) for k, v in mix.items()}, strict=True)
        dump("%s_a%.2f" % (name, a), run_records(ag, recs, None, 32, 100))
main()
