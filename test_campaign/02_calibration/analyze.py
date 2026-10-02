import json, math, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
R = '/home/zera/laya-vision/'
rng = np.random.default_rng(0)

def load(p):
    out = []
    for l in open(p):
        r = json.loads(l)
        if r.get('probs') is None: continue
        out.append(r)
    return out

def arrs(rows):
    P = [np.asarray(r['probs'], float) for r in rows]
    P = [np.clip(p, 0, None) for p in P]; P = [p / p.sum() for p in P]
    Y = np.array([int(np.argmax(r['target'])) for r in rows])
    K = np.array([len(p) for p in P])
    return P, Y, K

def conf_corr(P, Y):
    c = np.array([p.max() for p in P]); ok = np.array([int(p.argmax()) == y for p, y in zip(P, Y)], float)
    return c, ok

def ent_conf(P):
    return np.array([1 - (-(p * np.log(np.clip(p, 1e-12, 1))).sum()) / math.log(len(p)) for p in P])

def ece_fixed(c, ok, b):
    e = np.linspace(0, 1, b + 1); t = 0
    for i in range(b):
        s = ((c >= e[i]) if i == 0 else (c > e[i])) & (c <= e[i + 1])
        if s.any(): t += s.sum() * abs(c[s].mean() - ok[s].mean())
    return t / len(c)

def ece_adapt(c, ok, b):
    o = np.argsort(c, kind='stable'); parts = np.array_split(o, b); t = 0
    for s in parts:
        if len(s): t += len(s) * abs(c[s].mean() - ok[s].mean())
    return t / len(c)

def auroc(score, ok):
    pos, neg = ok == 1, ok == 0
    if pos.sum() == 0 or neg.sum() == 0: return float('nan')
    r = rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))

def selective(score, ok, covs=(1, .8, .6, .5, .4, .3, .2)):
    o = np.argsort(-score, kind='stable'); out = {}
    for cv in covs:
        k = max(1, int(round(cv * len(ok)))); out[str(int(cv * 100))] = float(ok[o[:k]].mean())
    return out

def wilson(k, n, z=1.96):
    if n == 0: return [float('nan')] * 2
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [m - h, m + h]

def nll_brier(P, Y):
    nll = np.mean([-math.log(max(p[y], 1e-12)) for p, y in zip(P, Y)])
    br = np.mean([((p - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(P, Y)])
    return float(nll), float(br)

def boot(fn, n, B=300):
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(fn(i))
    v = np.array(v); return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

def metrics(rows, ci=True):
    P, Y, K = arrs(rows); c, ok = conf_corr(P, Y); ec = ent_conf(P); n = len(rows)
    nll, br = nll_brier(P, Y)
    m = dict(n=n, acc=float(ok.mean()), acc_ci=wilson(int(ok.sum()), n), chance=float(np.mean(1 / K)),
             mean_conf=float(c.mean()), signed_gap=float(c.mean() - ok.mean()),
             ece10=ece_fixed(c, ok, 10), ece15=ece_fixed(c, ok, 15),
             aece10=ece_adapt(c, ok, 10), aece15=ece_adapt(c, ok, 15),
             auroc_conf=auroc(c, ok), auroc_entropy=auroc(ec, ok), nll=nll, brier=br,
             sel_conf=selective(c, ok), sel_entropy=selective(ec, ok))
    if ci and n >= 30:
        m['ece10_ci'] = boot(lambda i: ece_fixed(c[i], ok[i], 10), n)
        m['aece10_ci'] = boot(lambda i: ece_adapt(c[i], ok[i], 10), n)
        m['auroc_conf_ci'] = boot(lambda i: auroc(c[i], ok[i]), n)
    return m

def rescale(P, tau):  # p^(1/tau) renormalised == logit temperature scaling
    out = []
    for p in P:
        q = np.clip(p, 1e-9, 1) ** (1 / tau); out.append(q / q.sum())
    return out

def fit_tau(P, Y):
    f = lambda lt: nll_brier(rescale(P, math.exp(lt)), Y)[0]
    return math.exp(minimize_scalar(f, bounds=(-1.5, 1.5), method='bounded').x)

# ---- data
old = load(R + 'eval/results/stage2_c_wise085.predictions.jsonl')
newc = load('pred_cal.jsonl')
try: newu = load('pred_uncal.jsonl')
except Exception: newu = []
ID_TASKS = sorted({r['task'] for r in old})
seen = ['cifar10', 'food101', 'aokvqa', 'scienceqa']
allrows = old + newc
by = lambda rows, t: [r for r in rows if r['task'] == t]
res = {'meta': {'checkpoint': 'runs/stage2_c/wise085_calibrated', 'seed': 0,
                'old_source': 'eval/results/stage2_c_wise085.predictions.jsonl (full test splits; probs rounded to 4dp)',
                'new_source': 'pred_cal.jsonl: 150 random states/task, fp16 autocast', 'bootstrap': 'row-level, 300 resamples (rows from same image are correlated -> CIs optimistic)'}}
res['per_task'] = {t: metrics(by(allrows, t)) for t in ID_TASKS + seen + ['coco']}
# per modality/qtype
res['per_modality'] = {}
for name, rs in [('image_old(oxford,vqav2,eurosat)', [r for r in old if r['modality'] == 'image']),
                 ('text_old(ag_news,boolq,typed)', [r for r in old if r['modality'] == 'text']),
                 ('image_trained_new(cifar10,food101,aokvqa,scienceqa)', [r for t in seen for r in by(newc, t)]),
                 ('image_unseen(coco)', by(newc, 'coco'))]:
    res['per_modality'][name] = metrics(rs)
# per option count (image, all tasks) ; split choice vs noul
img = [r for r in allrows if r['modality'] == 'image']
res['per_option_count'] = {}
for k in sorted({len(r['probs']) for r in img}):
    rs = [r for r in img if len(r['probs']) == k]
    res['per_option_count'][f'K={k}'] = metrics(rs)
res['per_option_count_text'] = {f'K={k}': metrics([r for r in allrows if r['modality'] == 'text' and len(r['probs']) == k])
                                for k in sorted({len(r['probs']) for r in allrows if r['modality'] == 'text'})}
# OOD shift: cifar/food/aokvqa vs in-dist (oxford/vqav2/eurosat) by option-matched
res['groups'] = {'ID_old_image': metrics([r for r in old if r['modality'] == 'image']),
                 'cifar10+food101+aokvqa': metrics([r for t in ['cifar10', 'food101', 'aokvqa'] for r in by(newc, t)]),
                 'coco_unseen': metrics(by(newc, 'coco'))}
# temperature analysis (power scaling of final probs). fit tau on ID old image, apply elsewhere
Pi, Yi, _ = arrs([r for r in old if r['modality'] == 'image'])
tau_id = fit_tau(Pi, Yi)
res['temperature'] = {'tau_fit_on_old_image_tasks(NLL)': tau_id, 'per_task': {}}
for t in ID_TASKS + seen + ['coco']:
    rs = by(allrows, t); P, Y, _ = arrs(rs); c0, ok = conf_corr(P, Y)
    to = fit_tau(P, Y)
    ent = {}
    for nm, tau in [('transfer_tau', tau_id), ('oracle_tau', to)]:
        Q = rescale(P, tau); c, ok = conf_corr(Q, Y); nl, br = nll_brier(Q, Y)
        ent[nm] = dict(tau=tau, ece10=ece_fixed(c, ok, 10), aece10=ece_adapt(c, ok, 10), nll=nl)
    P0 = metrics(rs, ci=False)
    ent['as_is'] = dict(ece10=P0['ece10'], aece10=P0['aece10'], nll=P0['nll'])
    res['temperature']['per_task'][t] = ent
# calibrated vs uncalibrated checkpoint on new tasks
if newu:
    key = {r['id']: r for r in newu}
    res['cal_vs_uncal'] = {}
    for t in seen + ['coco']:
        a = by(newc, t); b = [key[r['id']] for r in a if r['id'] in key]
        a = [r for r in a if r['id'] in key]
        res['cal_vs_uncal'][t] = {'calibrated': metrics(a, False), 'uncalibrated': metrics(b, False)}
json.dump(res, open('results.json', 'w'), indent=1, default=float)

# ---- plots
def rel(ax, rows, title, b=10):
    P, Y, _ = arrs(rows); c, ok = conf_corr(P, Y); e = np.linspace(0, 1, b + 1)
    xs, ys, ns = [], [], []
    for i in range(b):
        s = ((c >= e[i]) if i == 0 else (c > e[i])) & (c <= e[i + 1])
        if s.sum() >= 5: xs.append(c[s].mean()); ys.append(ok[s].mean()); ns.append(s.sum())
    ax.plot([0, 1], [0, 1], 'k--', lw=1); ax.plot(xs, ys, 'o-'); ax.set_title(f'{title}\nn={len(rows)} ECE10={ece_fixed(c, ok, b):.3f}', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
tasks = ID_TASKS + seen + ['coco']
fig, axs = plt.subplots(3, 4, figsize=(14, 10))
for ax, t in zip(axs.ravel(), tasks): rel(ax, by(allrows, t), t)
for ax in axs.ravel()[len(tasks):]: ax.axis('off')
fig.supxlabel('confidence (max p)'); fig.supylabel('accuracy'); plt.tight_layout(); plt.savefig('reliability_by_task.png', dpi=110); plt.close()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for nm, rs in [('ID image (old)', [r for r in old if r['modality'] == 'image']), ('cifar/food/aokvqa', [r for t in ['cifar10', 'food101', 'aokvqa'] for r in by(newc, t)]), ('coco (unseen)', by(newc, 'coco'))]:
    P, Y, _ = arrs(rs); c, ok = conf_corr(P, Y); ec = ent_conf(P)
    for ax, s, lb in [(axs[0], c, 'max-p ranking'), (axs[1], ec, 'entropy ranking')]:
        sel = selective(s, ok, [i / 20 for i in range(20, 3, -1)]); ax.plot([int(k) / 100 for k in sel], list(sel.values()), label=nm); ax.set_title('selective accuracy, ' + lb); ax.set_xlabel('coverage')
axs[0].legend(fontsize=7); axs[0].set_ylabel('accuracy'); plt.tight_layout(); plt.savefig('selective.png', dpi=110); plt.close()
print(json.dumps({k: v for k, v in res.items() if k in ('groups',)}, indent=1, default=float)[:3000])
