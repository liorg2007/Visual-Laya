!nvidia-smi
import os, subprocess, torch

n_gpu = torch.cuda.device_count()
print(f"CUDA Available: {torch.cuda.is_available()} | Visible GPUs: {n_gpu}")
for i in range(n_gpu):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f} GB)")

assert n_gpu >= 2, (
    f"Expected 2 GPUs, but detected {n_gpu}!\n"
    "Please switch your Kaggle Accelerator: on the right sidebar, go to Notebook options -> "
    "Accelerator -> select GPU T4 x2."
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("Both T4 GPUs verified and ready for DDP training!")

!pip install -q -U "laya>=0.1.6" "transformers>=4.48.0" "datasets>=3.0.0" safetensors huggingface_hub pyarrow pandas scipy accelerate tabulate
import laya, transformers, datasets, torch
print("Laya version        :", laya.__version__)
print("Transformers version:", transformers.__version__)
print("PyTorch version     :", torch.__version__)

import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES

MODEL_ID = "convaiinnovations/laya"
print(f"Fetching tokenizer and config from {MODEL_ID}...")
model_dir = snapshot_download(MODEL_ID)
_fix_tokenizer_config(model_dir)

tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
    cfg = json.load(f)

print("Downloading LocalLLaMA/typed-decisions (train split)...")
ds_train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")

def build_training_item(state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))
    
    seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label
    }

items = []
for row in ds_train:
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    for qid, q in questions.items():
        if qid in gold:
            it = build_training_item(state, q, gold[qid])
            if it:
                items.append(it)

print(f"Preprocessed {len(items)} training sequences across 1,200 cases.")
torch.save(items, "/kaggle/working/train_items.pt")
print("Saved preprocessed items to /kaggle/working/train_items.pt")

%%writefile /kaggle/working/train_ddp.py
import os, sys, time, json, random, math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, proper_reward, QTYPES

def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items])
    }

def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    all_items = torch.load("/kaggle/working/train_items.pt", weights_only=False)

    # Hold the calibration slice out of training before sharding. Temperatures fitted on
    # items the run has already trained on measure the fit rather than the calibration: the
    # model is near-certain and near-correct on them, so the optimiser has nothing to soften
    # and returns a degenerate scale. The seed is fixed and rank-independent, so every rank
    # withholds exactly the same items and none of them reaches a training batch.
    CALIB_MAX = 400
    order = list(range(len(all_items)))
    random.Random(20260922).shuffle(order)
    n_calib = min(CALIB_MAX, len(all_items) // 10)
    calib_items = [all_items[i] for i in sorted(order[:n_calib])]
    train_items = [all_items[i] for i in sorted(order[n_calib:])]
    my_items = train_items[rank::world_size]
    
    EPOCHS = 4
    MICRO_BATCH = 8      # 8 sequences per forward pass per GPU
    GRAD_ACCUM = 4       # Effective batch across 2 GPUs = 64 sequences (8 * 2 * 4)
    GROUP_SIZE = 4       # GRPO baseline samples
    LR_ENCODER = 2.5e-5  # Encoder adaptation rate
    LR_HEAD = 1.0e-4     # Head adaptation rate
    SIGMA_START = 0.4    # Exploration noise
    SIGMA_END = 0.1

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)
    
    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    if rank == 0:
        print(f"Starting 2xT4 DDP training: {len(train_items)} train items ({len(calib_items)} held out for calibration) | {len(my_items)} per rank | {EPOCHS} epochs")
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        
        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress
        
        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            
            batch = collate_train_batch(chunk, tok.pad_token_id)
            
            with torch.autocast("cuda", dtype=torch.float16):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device)
                )
            
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            
            # 1. Sample G noisy logit distributions with zero-mean projection
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            
            # 2. Evaluate proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            
            # 3. Policy gradient loss + full 1.0 soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            
            scaler.scale(loss).backward()
            accum_step += 1
            
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            
            if rank == 0 and (n_batches % 50) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}")

        if rank == 0:
            print(f"=== Epoch {epoch+1}/{EPOCHS} Completed in {time.time()-t0:.1f}s | Avg Loss: {epoch_loss/max(1, n_batches):.4f} ===")

        dist.barrier()

        # Overwrite a single rolling checkpoint after each epoch so a crash,
        # OOM, or Kaggle session timeout doesn't lose all prior training.
        if rank == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoint_latest")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, os.path.join(ckpt_dir, "model.safetensors"))
            model.encoder.config.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            tok.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "total_epochs": EPOCHS,
                    "avg_loss": epoch_loss / max(1, n_batches)
                }, f, indent=2)
            print(f"  Saved rolling checkpoint (epoch {epoch+1}/{EPOCHS}) to {ckpt_dir}")

    dist.barrier()
    
    # Post-training temperature calibration on rank 0 (micro-batched in chunks of 16 to prevent OOM)
    if rank == 0:
        print("\nFitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        # calib_items was held out above and never entered a training batch
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device)
                    )
                l_np = l_sub.float().cpu().numpy()
                for r, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r, :k], it["target"]))
        
        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted calibration temperatures (choice, score, noul):", [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fitting fallback:", e)
        os.makedirs(output_dir, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(output_dir, "model.safetensors"))
        model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
        tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
        
        cfg["fine_tuned"] = True
        cfg["model_name"] = "laya-typed-decisions"
        cfg["temperature"] = fitted_temps
        # This fit is per type; inherited bucket overrides would hide the new values.
        cfg.pop("temperature_by_options", None)
        with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Model successfully saved to {output_dir}!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

OUTPUT_DIR = "/kaggle/working/laya_finetuned_typed_decisions"
MODEL_DIR = model_dir

cmd = f"torchrun --standalone --nproc_per_node=2 /kaggle/working/train_ddp.py {MODEL_DIR} {OUTPUT_DIR}"
print("Executing DDP training:", cmd)
!{cmd}

import time, json
import numpy as np
import pandas as pd
from datasets import load_dataset
import laya
from laya.common import ece_score

print("Loading test split for evaluation...")
ds_test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")

agent_ft = laya.Agent(OUTPUT_DIR, device="cuda")

predictions = []
latencies_ms = []

print(f"Evaluating {len(ds_test)} test cases on GPU...")
t0_eval = time.time()

for i, row in enumerate(ds_test):
    case_id = row["id"]
    workflow = row["workflow"]
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    
    t0 = time.perf_counter()
    res = agent_ft.predict(state, questions)
    dt_ms = (time.perf_counter() - t0) * 1000
    latencies_ms.append(dt_ms)
    
    predictions.append({
        "id": case_id,
        "workflow": workflow,
        "pred": res["answers"],
        "gold": gold,
        "questions": questions,
        "latency_ms": dt_ms
    })

print(f"Evaluated all {len(ds_test)} cases in {time.time() - t0_eval:.1f}s!")

accuracies = []
soft_accuracies = []
brier_scores = []
kl_divs = []
tv_distances = []
score_maes = []
within_one = []
all_confs = []
all_corrects = []

for item in predictions:
    pred_answers = item["pred"]
    gold_answers = item["gold"]
    questions = item["questions"]
    
    for qid, qdef in questions.items():
        p_ans = pred_answers[qid]
        g_ans = gold_answers[qid]
        q_type = qdef["type"]
        
        # 1. Choice
        if q_type == "choice":
            keys = list(qdef["criteria"].keys())
            pred_choice = p_ans["choice"]
            gold_label = str(g_ans["label"])
            
            is_corr = float(pred_choice == gold_label)
            accuracies.append(is_corr)
            all_corrects.append(is_corr)
            
            p_probs = np.array([p_ans["probabilities"].get(k, 1e-6) for k in keys])
            g_probs = np.array([g_ans["probabilities"].get(k, 1e-6) for k in keys])
            p_probs /= p_probs.sum()
            g_probs /= g_probs.sum()
            
            all_confs.append(float(p_probs.max()))
            soft_accuracies.append(float((p_probs * g_probs).sum()))
            brier_scores.append(float(((p_probs - g_probs) ** 2).sum()))
            tv_distances.append(float(0.5 * np.abs(p_probs - g_probs).sum()))
            kl_divs.append(float((g_probs * np.log(np.clip(g_probs / p_probs, 1e-12, 1e4))).sum()))
            
        # 2. Noul
        elif q_type == "noul":
            p_val = p_ans["noul"]
            g_val = g_ans.get("noul", g_ans.get("probabilities", {}).get("true", 0.5))
            gold_label = str(g_ans["label"]).lower()
            
            pred_label = "true" if p_val >= 0.5 else "false"
            is_corr = float(pred_label == gold_label)
            accuracies.append(is_corr)
            all_corrects.append(is_corr)
            all_confs.append(float(max(p_val, 1.0 - p_val)))
            
            p_dist = np.array([1.0 - p_val, p_val])
            g_dist = np.array([1.0 - g_val, g_val])
            
            soft_accuracies.append(float((p_dist * g_dist).sum()))
            brier_scores.append(float(((p_dist - g_dist) ** 2).sum()))
            tv_distances.append(float(0.5 * np.abs(p_dist - g_dist).sum()))
            kl_divs.append(float((g_dist * np.log(np.clip(g_dist / p_dist, 1e-12, 1e4))).sum()))
            
        # 3. Score
        elif q_type == "score":
            p_score = p_ans["score"]
            g_score = g_ans.get("score", 0.0)
            score_maes.append(abs(p_score - g_score))
            within_one.append(float(abs(p_score - g_score) <= 1.0))
            
            # Use argmax of probabilities for discrete prediction
            n_levels = len(qdef.get("criteria", []))
            p_probs_score = np.array([p_ans["probabilities"].get(str(i), 0.0) for i in range(n_levels)])
            if p_probs_score.sum() > 0:
                p_probs_score /= p_probs_score.sum()
                p_lvl = int(np.argmax(p_probs_score))
                all_confs.append(float(p_probs_score.max()))
            else:
                p_lvl = int(round(p_score))
                all_confs.append(0.5)
                
            g_lvl = int(g_ans.get("label", int(round(g_score))))
            is_corr = float(p_lvl == g_lvl)
            accuracies.append(is_corr)
            all_corrects.append(is_corr)

laya_acc = np.mean(accuracies)
laya_soft_acc = np.mean(soft_accuracies)
laya_brier = np.mean(brier_scores)
laya_kl = np.mean(kl_divs)
laya_tv = np.mean(tv_distances)
laya_ece = ece_score(np.array(all_confs), np.array(all_corrects))
laya_mae = np.mean(score_maes) if score_maes else 0.0
laya_within1 = np.mean(within_one) if within_one else 0.0
laya_latency = np.percentile(latencies_ms, 50)

comparison_data = [
    {
        "Model": "TypeSafe Jev 1.13.0",
        "Kind": "general",
        "Accuracy": 0.727,
        "Soft Acc": 0.580,
        "Brier": 0.148,
        "ECE": 0.144,
        "Score MAE": 0.391,
        "Within 1 Level": "0.952",
        "ms/case": 710,
        "Cost/Case": "$0.0004 (API)"
    },
    {
        "Model": "Laya (Fine-Tuned 2xT4)",
        "Kind": "fine-tuned",
        "Accuracy": round(laya_acc, 3),
        "Soft Acc": round(laya_soft_acc, 3),
        "Brier": round(laya_brier, 3),
        "ECE": round(laya_ece, 3),
        "Score MAE": round(laya_mae, 3),
        "Within 1 Level": f"{laya_within1:.3f}",
        "ms/case": round(laya_latency, 1),
        "Cost/Case": "$0.00 (Self-Hosted)"
    },
    {
        "Model": "ModernBERT-base (149M)",
        "Kind": "specialist",
        "Accuracy": 0.646,
        "Soft Acc": 0.542,
        "Brier": 0.119,
        "ECE": 0.179,
        "Score MAE": 0.444,
        "Within 1 Level": "0.931",
        "ms/case": 349,
        "Cost/Case": "$0.00"
    },
    {
        "Model": "Teacher Self-Agreement",
        "Kind": "ceiling",
        "Accuracy": 0.735,
        "Soft Acc": "-",
        "Brier": "-",
        "ECE": "-",
        "Score MAE": "-",
        "Within 1 Level": "-",
        "ms/case": "-",
        "Cost/Case": "-"
    }
]

df_comp = pd.DataFrame(comparison_data)
print("=== HEAD-TO-HEAD BENCHMARK TABLE ===\n")
print(df_comp.to_markdown(index=False))

import os
from huggingface_hub import HfApi

# Make sure you have added HF_TOKEN to Kaggle Secrets (Add-ons -> Secrets -> HF_TOKEN)
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    token = None

if not token:
    raise RuntimeError(
        "No HF_TOKEN found. Add it via Kaggle's Add-ons -> Secrets -> HF_TOKEN "
        "before running this cell, or set token = \"<your-write-token>\" manually above."
    )

NEW_REPO = "convaiinnovations/laya-typed-decisions"
print(f"Publishing {laya_acc:.3f} accuracy fine-tuned model to {NEW_REPO}...")

# Reference baselines pulled from the comparison table above, so the model
# card's narrative always matches the numbers actually computed this run.
JEV_ACC = next(r["Accuracy"] for r in comparison_data if r["Model"] == "TypeSafe Jev 1.13.0")
CEILING_ACC = next(r["Accuracy"] for r in comparison_data if r["Model"] == "Teacher Self-Agreement")

def describe_vs_baseline(acc, jev_acc=JEV_ACC, ceiling_acc=CEILING_ACC):
    """Builds an accuracy comparison sentence that reflects this run's actual result,
    instead of assuming the fine-tuned model always wins."""
    if acc > ceiling_acc:
        return (f"outperforming **TypeSafe Jev 1.13.0 ({jev_acc:.3f})** and surpassing the "
                f"benchmark's **Teacher Self-Agreement ceiling ({ceiling_acc:.3f})**")
    elif acc > jev_acc:
        return (f"outperforming **TypeSafe Jev 1.13.0 ({jev_acc:.3f})**, though still below the "
                f"benchmark's Teacher Self-Agreement ceiling ({ceiling_acc:.3f})")
    else:
        return (f"trailing **TypeSafe Jev 1.13.0 ({jev_acc:.3f})** and the benchmark's "
                f"Teacher Self-Agreement ceiling ({ceiling_acc:.3f})")

comparison_sentence = describe_vs_baseline(laya_acc)

# 1. Write comprehensive model card README.md into OUTPUT_DIR before upload
readme_content = f"""---
license: apache-2.0
library_name: transformers
tags:
- laya
- system-one
- calibrated-decisions
- rlcd
- structured-decisions
- typed-decisions
- benchmark
metrics:
- accuracy
- brier_score
model-index:
- name: laya-typed-decisions
  results:
  - task:
      type: text-classification
      name: System One Decision Benchmark
    dataset:
      type: LocalLLaMA/typed-decisions
      name: Typed Decisions
    metrics:
    - type: accuracy
      value: {laya_acc:.3f}
    - type: brier_score
      value: {laya_brier:.3f}
---

# Laya (Fine-Tuned on Typed-Decisions Benchmark)

This is **Laya** fine-tuned on the 1,200 training cases (6,000 decisions) of the independent [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark.

On the official 400-case test set (2,000 decisions across Agent Trace Observability, Customer Service, Invoice Processing, and Security Incidents), it achieves **{laya_acc:.3f} Accuracy**, {comparison_sentence}.

## Head-to-Head Benchmark Results

| Model | Kind | Accuracy | Soft Acc | Brier Score | ECE | Score MAE | Within 1 Level | Latency (p50) | Cost/Case |
|---|---|---|---|---|---|---|---|---|---|
| **Laya (Ours)** | **fine-tuned** | **{laya_acc:.3f}** | **{laya_soft_acc:.3f}** | **{laya_brier:.3f}** | **{laya_ece:.3f}** | **{laya_mae:.3f}** | **{laya_within1:.3f}** | **{laya_latency:.1f} ms** | **$0.00 (Self-Hosted)** |
| TypeSafe Jev 1.13.0 | general | 0.727 | 0.580 | 0.148 | 0.144 | 0.391 | 0.952 | 710 ms | $0.0004 (API) |
| ModernBERT-base (149M) | specialist | 0.646 | 0.542 | 0.119 | 0.179 | 0.444 | 0.931 | 349 ms | $0.00 |
| Teacher Self-Agreement | ceiling | 0.735 | - | - | - | - | - | - | - |

## Installation & Quickstart

```bash
pip install laya
```

```python
import laya

# Load the fine-tuned model directly from Hugging Face
agent = laya.load("convaiinnovations/laya-typed-decisions")

# Evaluate any workflow state and typed questions in a single forward pass
result = agent.predict(state, questions)
print(result["answers"])
```

## License
Apache 2.0. Developed by [Convai Innovations](https://huggingface.co/convaiinnovations).
"""

with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme_content)

print(f"Generated model card at {os.path.join(OUTPUT_DIR, 'README.md')}")

api = HfApi(token=token)
api.create_repo(NEW_REPO, repo_type="model", private=False, exist_ok=True)

# Upload the fine-tuned weights, config, tokenizer, and README
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=NEW_REPO,
    repo_type="model",
    commit_message=f"Laya fine-tuned on typed-decisions: Accuracy {laya_acc:.3f} vs Jev {JEV_ACC:.3f} / Ceiling {CEILING_ACC:.3f}"
)

print(f"\nModel successfully published to: https://huggingface.co/{NEW_REPO}")

import json, os

report = {
    "benchmark": "LocalLLaMA/typed-decisions",
    "model": "Laya (Fine-Tuned 2xT4)",
    "n_cases": len(ds_test),
    "n_decisions": len(ds_test) * 5,
    "metrics": {
        "accuracy": round(float(laya_acc), 4),
        "soft_accuracy": round(float(laya_soft_acc), 4),
        "brier_score": round(float(laya_brier), 4),
        "ece": round(float(laya_ece), 4),
        "score_mae": round(float(laya_mae), 4),
        "within_1_level": round(float(laya_within1), 4),
        "latency_p50_ms": round(float(laya_latency), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "kl_divergence": round(float(laya_kl), 4),
        "total_variation": round(float(laya_tv), 4)
    },
    "comparison": comparison_data,
    "per_workflow": {}
}

# Calculate per-workflow accuracy
for wf in ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"]:
    wf_items = [item for item in predictions if item["workflow"] == wf]
    wf_corr = []
    for item in wf_items:
        for qid, qdef in item["questions"].items():
            p_ans = item["pred"][qid]
            g_ans = item["gold"][qid]
            if qdef["type"] == "choice":
                wf_corr.append(float(p_ans["choice"] == str(g_ans["label"])))
            elif qdef["type"] == "noul":
                pred_label = "true" if p_ans["noul"] >= 0.5 else "false"
                wf_corr.append(float(pred_label == str(g_ans["label"]).lower()))
            elif qdef["type"] == "score":
                p_probs = [p_ans["probabilities"].get(str(i), 0.0) for i in range(len(qdef.get("criteria", [])))]
                p_lvl = int(np.argmax(p_probs)) if sum(p_probs) > 0 else int(round(p_ans["score"]))
                g_lvl = int(g_ans.get("label", int(round(g_ans.get("score", 0.0)))))
                wf_corr.append(float(p_lvl == g_lvl))
    if wf_corr:
        report["per_workflow"][wf] = {
            "n_decisions": len(wf_corr),
            "accuracy": round(float(np.mean(wf_corr)), 4)
        }

# Save in working directory and model directory
out_file1 = "/kaggle/working/laya_typed_decisions_benchmark_report.json"
out_file2 = os.path.join(OUTPUT_DIR, "benchmark_report.json")

with open(out_file1, "w") as f:
    json.dump(report, f, indent=2)

with open(out_file2, "w") as f:
    json.dump(report, f, indent=2)

print(f"Benchmark results successfully saved to:")
print(f"  - {out_file1}")
print(f"  - {out_file2}")
print("\nSummary:")
print(json.dumps(report["metrics"], indent=2))
print("\nPer-Workflow Accuracy:")
print(json.dumps(report["per_workflow"], indent=2))
