# 08 Latency and resources: Laya-Vision `stage2_c/wise085_calibrated`

**Question.** What do latency, throughput, VRAM and CPU/GPU cost look like for the final checkpoint, with and without an image, across batch sizes, and versus the caption-then-text pipeline?

**Method.** `bench.py` (3 rounds x 8 timed iterations per config after 3 warm-ups, distinct COCO images, seed 0, `torch.cuda.synchronize` around each call, end-to-end `agent.predict`/`predict_batch` including decode, preprocess, vision tower and heads). `pipe.py` (8 iters) compares against stock text Laya and BLIP-caption -> stock Laya. `cpu_small.py` (6 iters, 8 threads, fp32) gives a CPU reference. Tables report the median over rounds of the per-round p50, with the min-max range across rounds; p95 is per-round p95 (n=8, so it is essentially the worst call). **Caveat: the RTX 4060 Ti was shared with about 11 other agents (GPU util 63-88% before/after our rounds, 7.5 GB used), so all numbers are contended; CUDA init took 84 s, model load 239 s and the first call 18 s purely because of contention.** Treat p50 as an upper bound on dedicated-GPU latency; p95 is dominated by multi-second stalls from other processes (many configs show p95 about 2.4 s while p50 is about 50 ms).

## Numbers (GPU, ms per call, p50 [range over 3 rounds])
| config | p50 | p95 (median of rounds) |
|---|---|---|
| text only, 1 q | 25.6 [24.7-43.3] | 56 |
| text only, 8 q | 35.2 [32.8-44.2] | 40 |
| image, 1 q | 44.3 [43.3-82.6] | 2374 (stall) |
| image, 4 q | 49.5 [47.4-57.5] | 65 |
| image, 8 q | 57.5 [53.9-87.9] | 2437 |
| image, 16 q | 84.2 [79.5-94.4] | 2363 |
| image, 32 q | 148.8 [148.3-178.4] | 3735 |
| image + text, 8 q | 63.1 [61.3-81.7] | 2343 |
| image, 1 q, 2..12 options | 41-45 (flat) | - |
| batch 1 / 4 / 8 / 16 images, 1 q (total) | 48 / 59 / 92 / 146 | - |
| batch 1 / 4 / 8 / 16 images, 4 q (total) | 49 / 93 / 172 / 357 | - |
| 8 images, 4 q, all the same image | 164 (no gain vs distinct: 172) | - |

Per image in a batch of 16 with 1 q: 9.1 ms (about 110 images/s), versus 44 ms unbatched (about 23/s): roughly 4.8x throughput. With 4 q the batch-16 rate is 22 ms/image (about 45/s).

Components (p50, ms): image decode+hash from path 1.8, from PIL 0.6, preprocess 2.4, vision-tower `encode_images` 17.3 (single image, fp32 path without autocast).

**Memory.** VLM 516.0 M params (Laya 421.3 M incl. 394.8 M encoder, SigLIP-base vision tower 92.9 M, projector 1.8 M), stored fp32: 1969 MB allocated after load. Peak torch allocation 2.9 GB for image + 1-32 q, 3.0 GB for batch 8, 3.1 GB for batch 16 x 4 q. So about 3.1 GB peak, fitting an 8 GB card with headroom (activations add only 0.9-1.1 GB). Stock Laya text: 1607 MB, BLIP-base captioner: 2484 MB; the Laya-Vision model (1977 MB) is smaller than BLIP alone.

**Pipeline comparison (GPU, 8 iters, p50 ms).**
| system | 1 q | 8 q | 32 q | VRAM |
|---|---|---|---|---|
| Laya-Vision (image) | 52 | 58 | 150 | 1.98 GB (peak 3.0) |
| BLIP caption -> stock Laya, cache miss | 230 | 218 | 317 | 1.6 + 2.5 GB |
| same, caption cache hit | - | 32 | - | text path only |
| stock Laya on caption text | 25 | 35 | 87 | 1.6 GB |
BLIP captioning alone is about 244 ms p50, so the cascade is about 4x slower per fresh image than Laya-Vision (52 ms), needs about 2x the weights in VRAM, and only wins if captions are cached (32 ms vs 58 ms). Pipeline p95/mean were dominated by contention stalls (e.g. BLIP mean 2.1 s vs p50 244 ms).

**CPU reference (8 threads of 16, fp32, other agents also use CPU, n=6).** image 1 q 369 ms (3.1 CPU-s per call), image 8 q 1718 ms (14.0 CPU-s), text 8 q 773 ms, batch of 8 images 1 q 2179 ms; RSS 3.2 GB, load 2.9 s. CPU is about 8x slower than the contended GPU, still usable at about 3 images/s for 1 question.

## Findings
- Cost is dominated by the question count, not the image: image 1 q = 44 ms, 8 q = 58 ms, 32 q = 149 ms (about 3.3 ms per extra question beyond the first few); option count (2 to 12) has no measurable effect; the image adds about 20-25 ms over text only (decode+preprocess about 3 ms, vision tower about 17 ms).
- Batching helps images: about 4.8x throughput at B=16 (1 q). No image dedup benefit was found (same-image batch is not faster than distinct images).
- Memory is small: 2.0 GB weights, about 3.1 GB peak even at B=16 x 4 q; fp32 weights, so fp16/bf16 storage would roughly halve the weight footprint (not tested, no accuracy check).

## Caveats
Shared, heavily loaded GPU (inflated and noisy p50, round-to-round range up to 2x on some configs, and p95 reflects stalls rather than the model); only 3 rounds x 8 iters (p95 from n=8 is weak); one run on one machine; CPU run is a small subset; GPU power/CPU-utilization of the GPU runs not measured; `first_call_ms` and `load_s` are contention artifacts and should not be quoted.

## Take-aways
1. About 45-60 ms per image with up to 8 questions on a shared 4060 Ti (about 25-35 ms text only); about 150 ms for 32 questions.
2. Peak VRAM about 3.1 GB (2.0 GB weights fp32), so the model fits comfortably; batch 16 gives about 110 images/s at 1 q.
3. The caption->text cascade is about 4x slower and uses more VRAM per uncached image than Laya-Vision; CPU-only is feasible at about 0.4 s/image.

Files: `results.json`, `latency.png`, `gpu_bench.json`, `pipe_cuda.json`, `cpu_small.json`, scripts `bench.py pipe.py cpu_small.py analyze.py make_report.py`.
