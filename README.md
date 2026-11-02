# Disentangle to Create: Architecture-Level Disentangled Diffusion Model for Few-Shot Chinese Font Generation

This repository is the official implementation of **“Disentangle to Create:
Architecture-Level Disentangled Diffusion Model for Few-Shot Chinese Font
Generation”** (ALD-Font). 

## Environment

Python 3.8 is recommended because the project uses PyTorch Lightning 1.6.5.

```bash
conda create -n ald-font python=3.8.5 -y
conda activate ald-font

# This is the combination used for the end-to-end smoke test (CUDA 12.1).
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e .
```

Use the matching PyTorch wheel when the machine has a different CUDA runtime.
Run all commands below from the repository root so relative model and data paths
resolve correctly.

The default config expects these pretrained assets outside the repository:

```text
PATH/v2-1_512-ema-pruned.ckpt
PATH/chinese-vit-large-patch14/
```

Change `configs/train.yaml` when the assets are stored elsewhere.

## Data layout

The checked-in fonts and JSON files are small samples only. For real training,
place your own fonts in the same layout; a `.txt` character map must accompany
each `.ttf` with the same stem.

```text
train_data/
  simsun.ttf
  train_chars_eg.json
  train_ttf/
    example.ttf
    example.txt
eval_data/
  eval_chars_eg.json
  ref_chars.json
  eval_ttf/
    example.ttf
    example.txt
```

Generate the `.txt` maps with:

```bash
python process_ttf.py train_data/train_ttf
python process_ttf.py eval_data/eval_ttf
```

## 1. Train the diffusion model

```bash
python main.py -b configs/train.yaml --gpus 0,1,2,3 -t --no-test true
```

The run directory is printed at startup and checkpoints are saved below
`RUN_DIR/checkpoints/`. The sample config uses a small batch size; tune it for
your GPUs and dataset.

Set the actual directory printed by the training command before running the
remaining commands, for example:

```bash
RUN_DIR=logs/2026-11-02T08-27-39_train
```

## 2. Train BNR

BNR consumes a completed diffusion run:

```bash
python train_bnr.py \
  -l "$RUN_DIR" \
  --batch_size 16 \
  --epochs 100 \
  --name bnr \
  --tqdm
```

This creates `$RUN_DIR/bnr/config.json` and `$RUN_DIR/bnr/last.ckpt`.

## 3. Generate from reference images

Put a few glyph images in one directory. Each filename must identify the glyph:

```text
references/
  永.png
  字.png
  体.png
```

The images may be PNG, JPEG, or WebP. Use one Chinese character per filename;
an optional suffix after `_` is allowed, for example `智_01.png`. Generate a
target character list without fine-tuning:

```bash
python predict.py \
  --logdir "$RUN_DIR" \
  --bnr "$RUN_DIR/bnr" \
  --ref-img-dir references \
  --chars eval_data/eval_chars_eg.json \
  --output outputs/example
```

`--chars` also accepts a literal string such as `--chars 人工智能`.

## 4. PEFT then generate

In this repository, `peft` means training the diffusion transformer together
with the Chinese-CLIP image projection, matching `transformer+clip` in the
reference project. The named reference images are both the few-shot supervision
and the style references.

```bash
python predict.py \
  --logdir "$RUN_DIR" \
  --bnr "$RUN_DIR/bnr" \
  --ref-img-dir references \
  --chars 人工智能 \
  --finetune peft \
  --finetune-epochs 80 \
  --finetune-lr 8.192e-4 \
  --save-peft outputs/example/peft_transformer_clip.pt \
  --output outputs/example
```

Reuse a saved adapter without fine-tuning again:

```bash
python predict.py \
  --logdir "$RUN_DIR" \
  --bnr "$RUN_DIR/bnr" \
  --ref-img-dir references \
  --chars 人工智能 \
  --peft-ckpt outputs/example/peft_transformer_clip.pt \
  --output outputs/example_reuse
```

Omit `--bnr` to save the raw diffusion output. With BNR enabled, the final image
is saved as `CHAR.png` and the pre-BNR image as `CHAR_before_bnr.png`.
