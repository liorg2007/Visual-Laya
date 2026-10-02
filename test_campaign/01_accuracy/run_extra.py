import json, random, sys, gc, torch
sys.path.insert(0,'/home/zera/laya-vision')
from laya_vision.eval import run_eval as R
from laya_vision.eval import baselines as B
which = sys.argv[1]
root='/home/zera/laya-vision/'
recs=[]
rng=random.Random(0)
for t in ['cifar10','food101','aokvqa','coco']:
    rs=[json.loads(l) for l in open(root+f'data/{t}.test.jsonl')]
    imgs=sorted({r['image'] for r in rs}); rng.shuffle(imgs); keep=set(imgs[:300])
    recs+=[r for r in rs if r['image'] in keep]
# repro subset of oxford_pets
rs=[json.loads(l) for l in open(root+'data/oxford_pets.test.jsonl')]
imgs=sorted({r['image'] for r in rs}); rng.shuffle(imgs); keep=set(imgs[:60])
recs+=[r for r in rs if r['image'] in keep]
print(len(recs),flush=True)
dev='cuda'
if which=='final':
    import laya_vision
    runner=laya_vision.load(root+'runs/stage2_c/wise085_calibrated',device=dev)
else:
    import laya
    a=laya.Agent('convaiinnovations/laya',device=dev,revision='55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851')
    runner=B.CaptionLaya(a,cache_dir=root+'eval/cache/captions',captioner_id=B.DEFAULT_CAPTIONER,captioner_revision='82a37760796d32b1411fe092ab5d4e227313294b',device=dev)
res=R.evaluate(runner,recs,f'preds/{which}.json',which,image_root=root+'data/images',max_questions=32)
print(json.dumps(res['metrics']['by_task'],indent=0)[:1500])
