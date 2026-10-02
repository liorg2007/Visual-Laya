s = open("run_preds.py").read()
s = s.replace("from laya_vision.eval.run_eval import read_jsonl, run_records", '''from laya_vision.eval.run_eval import read_jsonl
from laya_vision.eval.metrics import answer_probs
from laya.agent import collate_items
BS = int(os.environ.get("BS", 16))
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
    return rows''')
open("run_preds.py", "w").write(s)
