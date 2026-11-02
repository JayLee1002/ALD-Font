"""Few-shot font generation from reference images, with optional PEFT and BNR."""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldm.data.ald_font import ReferenceImageFinetune, ReferenceImagePredict
from ldm.util import instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Training YAML; inferred from LOGDIR/configs when omitted")
    parser.add_argument("-l", "--logdir", default="", help="Training log directory")
    parser.add_argument("--ckpt", default="", help="Diffusion checkpoint; defaults to LOGDIR/checkpoints/last.ckpt")
    parser.add_argument("--bnr", default="", help="BNR directory containing config.json and last.ckpt")
    parser.add_argument("--ref-img-dir", required=True, help="Reference images named CHAR.png, e.g. 永.png")
    parser.add_argument("--chars", required=True, help="Target character JSON, or a literal character string")
    parser.add_argument("--source-font", default="train_data/simsun.ttf")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--font-name", default="reference")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--finetune",
        choices=("", "peft", "transformer+clip"),
        default="",
        help="peft is an alias of transformer+clip",
    )
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=8.192e-4)
    parser.add_argument("--finetune-repeat", type=int, default=1)
    parser.add_argument("--peft-ckpt", default="", help="Load an existing PEFT adapter state dict")
    parser.add_argument("--save-peft", default="", help="Where to save the trained PEFT adapter")
    return parser.parse_args()


def load_config(args):
    paths = [args.config] if args.config else sorted(glob.glob(os.path.join(args.logdir, "configs", "*.yaml")))
    if not paths:
        raise FileNotFoundError("No config found. Pass --config or a --logdir containing configs/*.yaml")
    configs = [OmegaConf.load(path) for path in paths]
    config = OmegaConf.merge(*configs)
    if "model" not in config:
        raise ValueError(f"No model section found in configs: {paths}")
    return config


def resolve_checkpoint(args):
    path = args.ckpt or (os.path.join(args.logdir, "checkpoints", "last.ckpt") if args.logdir else "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("Diffusion checkpoint not found; pass --ckpt or --logdir")
    return path


def configure_model(config, diffusion_ckpt, bnr_dir):
    # Avoid loading the diffusion prior twice. The first-stage VAE still uses its configured checkpoint.
    config.model.params.ckpt_path = None
    if bnr_dir:
        config.model.target = "ldm.models.diffusion.ald_font_bnr.ALDFont_with_BNR"
        config.model.params.ckpt_path = diffusion_ckpt
        config_path = os.path.join(bnr_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"BNR config not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            post_config = json.load(f)
        bnr_ckpt = os.path.join(bnr_dir, "last.ckpt")
        if not os.path.isfile(bnr_ckpt):
            raise FileNotFoundError(f"BNR checkpoint not found: {bnr_ckpt}")
        post_config.setdefault("params", {})["ckpt_path"] = bnr_ckpt
        config.model.params.post_stage_config = OmegaConf.create(post_config)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(diffusion_ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded diffusion checkpoint: {diffusion_ckpt}")
    print(f"State dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model


def peft_parameter_names(model):
    names = []
    for name, parameter in model.named_parameters():
        trainable = "transformer" in name or "style_stage_model.image_embedder.projection" in name
        parameter.requires_grad = trainable
        if trainable:
            names.append(name)
    if not names:
        raise RuntimeError("No transformer+clip parameters were found")
    return names


def load_peft(model, path):
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded PEFT adapter {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")


def finetune(model, args, device):
    names = peft_parameter_names(model)
    dataset = ReferenceImageFinetune(
        source_path=args.source_font,
        ref_img_dir=args.ref_img_dir,
        num_ref=-1,
        repeat=args.finetune_repeat,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.finetune_lr)
    model.train()
    for epoch in range(args.finetune_epochs):
        progress = tqdm(loader, desc=f"PEFT {epoch + 1}/{args.finetune_epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss = model.shared_step(batch)[0]
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
    adapter = {name: model.state_dict()[name].detach().cpu() for name in names}
    save_path = args.save_peft or os.path.join(args.output, "peft_transformer_clip.pt")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({"state_dict": adapter, "method": "transformer+clip"}, save_path)
    print(f"Saved PEFT adapter: {save_path}")
    model.requires_grad_(False)
    model.eval()


def tensor_to_image(tensor):
    array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8)).convert("RGB")


def generate(model, args):
    dataset = ReferenceImagePredict(
        source_path=args.source_font,
        ref_img_dir=args.ref_img_dir,
        target_chars=args.chars,
        font_name=args.font_name,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            generated, _ = model.predict(batch)
            middle = None
            if isinstance(generated, tuple):
                generated, middle = generated
            for index, (char, image) in enumerate(zip(batch["chars"], generated)):
                tensor_to_image(image).save(output_dir / f"{char}.png")
                if middle is not None:
                    tensor_to_image(middle[index]).save(output_dir / f"{char}_before_bnr.png")
    print(f"Generated {len(dataset)} characters in {output_dir}")


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    seed_everything(args.seed)
    config = load_config(args)
    checkpoint = resolve_checkpoint(args)
    device = torch.device(args.device)
    model = configure_model(config, checkpoint, args.bnr).to(device)
    if args.peft_ckpt:
        load_peft(model, args.peft_ckpt)
    if args.finetune:
        finetune(model, args, device)
    else:
        model.requires_grad_(False)
        model.eval()
    generate(model, args)


if __name__ == "__main__":
    main()
