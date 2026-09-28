# Handoff: data pipeline (`laya_vision/data/`) — DONE

## Files (all done)

| File | Content |
|---|---|
| laya_vision/data/__init__.py | re-exports the schema helpers (no heavy imports) |
| laya_vision/data/schema.py | TrainRecord, validate_record, read/write_jsonl, normalize, split_for (image-hash split) |
| laya_vision/data/dataset.py | DecisionDataset, make_collate/VisionCollate, LRUCache, MixtureSampler, build_mixture/MixtureDataset |
| laya_vision/data/augment.py | Augmenter, shuffle_options (choice only), rename_labels, paraphrase, TEMPLATES per task |
| laya_vision/data/text_mix.py | typed-decisions to records, with the notebook target rules |
| laya_vision/data/caption_align.py | stage-1 COCO caption choice/noul, hard negatives (TfidfNeighbors, or SiglipNeighbors behind a flag) |
| laya_vision/data/vision_tasks.py | stage-2 converters (see below), SOURCES config, save_image, hf_rows/csv_rows |
| laya_vision/data/prepare.py | CLI |
| laya_vision/data/LICENSES.md | licence table and every VERIFY item |
| configs/data_stage1.yaml, configs/data_stage2.yaml | sources, ratios, mixture/calib/eval file lists |
| tests/test_data_{schema,augment,dataset,converters}.py | 29 fast tests + 1 slow |

The stage-2 converters are classify (cifar10, oxford_pets, food101, eurosat), mc (aokvqa, scienceqa), vqa_yesno (vqav2_yesno) and score (koniq, ava).

## Test status

- `.venv/bin/python -m pytest tests/test_data*.py -q` gives `29 passed, 1 deselected`.
- `-m slow` (real typed-decisions test split) gives `1 passed`.
- ruff with the CI select passes.

## Contract changes

These are recorded in docs/interfaces.md, Data section.

- Records may carry the optional keys `template` and `fields`, which paraphrase uses.
- **Items:** also carry `id`, and `__getitem__` returns None for a record that does not fit. make_collate drops Nones and returns None for an empty batch. `prescan=True` filters those records up front.
- **Collate:** make_collate returns a picklable `VisionCollate`. It takes an optional `loader`.
- **Mixing:** `MixtureSampler(sizes, ratios, num_samples, seed, rank, world_size, drop_last)`, plus `build_mixture` and `MixtureDataset`.

## How to run

```
# stage 1 (COCO caption alignment, 500k records ~ 125k images)
python -m laya_vision.data.prepare --config configs/data_stage1.yaml
python -m laya_vision.data.prepare --stage1 coco --n 500000 --out data --image-root data/images
# stage 2 (+ text mix)
python -m laya_vision.data.prepare --config configs/data_stage2.yaml
python -m laya_vision.data.prepare --tasks cifar10,oxford_pets,food101,eurosat,aokvqa,scienceqa,vqav2_yesno \
    --out data --image-root data/images --max-per-task 60000 --n-distractors 1 --text-mix
```

KonIQ needs the official release unpacked to `data/raw/koniq10k/`. AVA is disabled.

## Gotchas

- The train configs (configs/stage*_a.yaml, owned by train) point at `data/stage2/train.jsonl` and similar. prepare instead writes per-task files `data/{task}.{split}.jsonl`. Use the `mixture`/`calib`/`eval` lists in configs/data_stage*.yaml.
- HF sources are shuffled before `max_per_task` is applied, because several are sorted by class. `max_per_task` counts records, including distractors.
- For the classification converters, the test split always uses the maximum option count on q0. Train and calib vary K across Laya's temperature buckets.
- VQAv2 balancing: per split, the yes-majority and no-majority counts never differ by more than `balance_slack` (50).
- The text rows are not augmented (Augmenter `augment_text=False`), which keeps the original typed-decisions distribution.

## Real-data smoke check (2026-09-28)

I streamed the first rows of `lmms-lab-encoder/VQAv2`, `blanchon/EuroSAT_RGB` and `HuggingFaceM4/A-OKVQA` through `convert_task` (3 records each). All the resulting records passed `validate_record`, and the column names are confirmed. COCO, CIFAR-10, Pets, Food-101 and ScienceQA columns were checked only through the datasets-server metadata.
