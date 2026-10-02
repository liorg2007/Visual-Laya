"""Run a checkpoint on subsampled states of extra tasks (cifar10, food101, aokvqa, scienceqa, coco)."""
import sys, json, random, time
sys.path.insert(0, '/home/zera/laya-vision')
import torch
from laya_vision.eval.run_eval import read_jsonl, group_records, run_records
ck, out = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 150
recs = []
for t in ['cifar10', 'food101', 'aokvqa', 'scienceqa', 'coco']:
    rs = read_jsonl([f'/home/zera/laya-vision/data/{t}.test.jsonl'])
    keys = list(group_records(rs).keys())
    random.Random(0).shuffle(keys)
    keep = set(keys[:NS])
    g = group_records(rs)
    for k in keys[:NS]:
        recs += g[k]
print(len(recs), 'records', file=sys.stderr)
import laya_vision
agent = laya_vision.load(ck, device='cuda')
t0 = time.time()
with torch.autocast('cuda', dtype=torch.float16):
    rows = run_records(agent, recs, '/home/zera/laya-vision/data/images', 32, 50)
with open(out, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('done', time.time() - t0, 'skipped', sum(r['probs'] is None for r in rows), file=sys.stderr)
