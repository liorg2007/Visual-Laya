# Dataset licences (plan.md Phase 3, item 1)

**Nothing here is legal advice.** The HF licence tags were read from the HF hub API on 2026-09-28. The rest comes from memory of the original dataset pages. Any row not marked `OK` must be checked by a human against the original source before its data goes into a released model.

Status legend:
- **OK**: licence clear for the stated use.
- **VERIFY**: unclear or unchecked; do not ship weights trained on it until resolved.
- **BLOCKED**: off by default.

| Dataset | HF id (default in code) | Licence (as far as known) | Allowed use | Used for | Status |
|---|---|---|---|---|---|
| typed-decisions (text mix) | `LocalLLaMA/typed-decisions` (config `all`) | Apache-2.0 (HF tag) | commercial OK with attribution | train (HF train split), text calib (5% of train cases), eval (HF test split) | OK |
| AG News (text regression) | `fancyzhx/ag_news` (test split) | unclear (academic, non-commercial per the original AG corpus page) | eval only | eval (`ag_news.test.jsonl`) | VERIFY |
| BoolQ (text regression) | `google/boolq` (validation split) | CC BY-SA 3.0 | eval only | eval (`boolq.test.jsonl`) | OK for eval |
| COCO Captions | `jxie/coco_captions` (repackaged COCO, Karpathy splits) | Annotations CC BY 4.0. Images come from Flickr under their individual licences, per the COCO terms of use. HF card has no licence tag. | research is common practice; commercial use of the images is unclear | stage-1 train/calib/test | VERIFY (repackaged mirror; image licences) |
| CIFAR-10 | `uoft-cs/cifar10` | No explicit licence (HF tag `unknown`). Images from the 80M Tiny Images collection, which was withdrawn by its authors in 2020. | research | stage-2 train/calib/test | VERIFY |
| Oxford-IIIT Pets | `timm/oxford-iiit-pet` | CC BY-SA 4.0 (HF tag and original page) | commercial OK, share-alike on derived *datasets* | stage-2 train + eval task (choice) | OK (check share-alike implications for released data) |
| Food-101 | `ethz/food101` | HF tag `unknown`. Images from foodspotting.com; the original page cites fair use. | research | stage-2 train/calib/test | VERIFY |
| EuroSAT (RGB) | `blanchon/EuroSAT_RGB` | EuroSAT is MIT per its GitHub repo, with Sentinel-2 data under the Copernicus open licence. The HF mirror is tagged `unknown`. | commercial OK if MIT confirmed | stage-2 train + eval fallback (choice) | VERIFY (mirror) |
| A-OKVQA | `HuggingFaceM4/A-OKVQA` | Annotations Apache-2.0 per the AI2 repo; no HF tag. Images are COCO 2017 (Flickr terms). | research; commercial for annotations if Apache confirmed | stage-2 train/calib/test (MC) | VERIFY |
| ScienceQA (image subset) | `derek-thomas/ScienceQA` | HF tag CC BY-SA 4.0. The original ScienceQA release states **CC BY-NC-SA 4.0**, which is non-commercial. | **non-commercial** if NC applies | stage-2 train/calib/test (MC) | VERIFY; exclude for a commercial release |
| VQAv2 (yes/no) | `lmms-lab-encoder/VQAv2` (validation, 214k questions with 10 answers each) | Annotations CC BY 4.0 (HF tag). Images are COCO (Flickr terms). The official `HuggingFaceM4/VQAv2` is a loading script and cannot be used with datasets>=4. | research; commercial unclear because of the images | stage-2 train + eval task (noul) | VERIFY (mirror id, columns, licence) |
| KonIQ-10k | none (local csv: `koniq10k_scores_and_distributions.csv` + `1024x768/`). `chaofengc/IQA-PyTorch-Datasets` hosts archives, tagged CC BY-SA 4.0. | Images are from YFCC100M under CC licences; the KonIQ page grants use for research. | research; commercial unclear | stage-2 train + eval task (score) | VERIFY |
| AVA | none (local `AVA.txt` + images) | Research-only; images from dpchallenge.com | research only | not used (`enabled: false`) | BLOCKED |
| Moderation / NSFW | none | licence-dependent (plan.md §7 decision 3) | none | not used | BLOCKED |

## VERIFY items in code

These are `laya_vision/data/vision_tasks.py` `SOURCES` and `caption_align.py` `COCO`.

### Columns checked with the HF datasets-server

- `uoft-cs/cifar10`: `img`, `label`
- `timm/oxford-iiit-pet`: `image`, `label`, `image_id`
- `ethz/food101`: `image`, `label`
- `blanchon/EuroSAT_RGB`: `image`, `label`, `filename`
- `HuggingFaceM4/A-OKVQA`: `image`, `question_id`, `question`, `choices`, `correct_choice_idx`
- `derek-thomas/ScienceQA`: `image`, `question`, `choices`, `answer`, `hint`
- `lmms-lab-encoder/VQAv2`: `question`, `answers[{answer}]`, `answer_type`, `image_id`, `question_id`, `image`
- `jxie/coco_captions`: `image`, `cocoid`, `caption`

### Not verified

- Whether those mirrors are faithful copies of the originals.
- KonIQ-10k csv column names: assumed `image_name`, `c1..c5`, `MOS`, `SD`.
- AVA.txt layout.
- The VQAv2 mirror's `answer_type` values: assumed `"yes/no"`.

### No revision pinned

Set `revision` per source in `configs/data_stage*.yaml` before the production run.
