import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import json

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.modules.bnr_module import BNR
from ldm.util import instantiate_from_config
import tqdm


def sobel_edges(img):
    Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    if img.shape[1] == 3:  # RGB
        img = img.mean(dim=1, keepdim=True)  # 灰度化
    
    edge_x = F.conv2d(img, Gx, padding=1)
    edge_y = F.conv2d(img, Gy, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
    return edge

def sobel_loss(output, target):
    return F.l1_loss(sobel_edges(output), sobel_edges(target))


class LightVGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.slice = nn.Sequential(*vgg[:9]).eval()  # up to relu2_2
        for p in self.slice.parameters():
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)
        x_feat = self.slice(x)
        y_feat = self.slice(y)
        return F.l1_loss(x_feat, y_feat)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="bnr module",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_ref",
        type=int,
        default=8
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
    )
    return parser


def tensor2image(x):
    x = x.detach().cpu()
    x = torch.clamp(x, 0., 1.)
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M ({total_params} parameters)")
    return total_params



if __name__ == "__main__":
    result_save_root = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    
    opt, unknown = parser.parse_known_args()
    base_configs = sorted(glob.glob(os.path.join(opt.logdir, "configs/*.yaml")))
    opt.base = base_configs + opt.base
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    config.data.params.batch_size = opt.batch_size
    config.data.params.predict.params.num_ref = opt.num_ref
    
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    print(config.data)
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    
    device = torch.device("cuda")
    model = instantiate_from_config(config.model)
    ckpt_path = opt.ckpt_path or f"{opt.logdir}/checkpoints/last.ckpt"
    save_dir = f"{opt.logdir}/{opt.name}"
    if os.path.exists(save_dir) and os.path.exists(f"{save_dir}/last.ckpt"):
        print("Continue training from:", save_dir)
        config_path = f"{save_dir}/config.json"
        bnr_model = BNR.from_config(config_path).to(device)
        print(bnr_model.hyper_parameters)
    else:
        print("Training new model")
        os.makedirs(save_dir, exist_ok=True)
        bnr_model = BNR(out_channels=3, in_channels=4).to(device)
        hyper = dict(target=f"{bnr_model.__class__.__module__}.{bnr_model.__class__.__name__}", 
                 params=bnr_model.hyper_parameters)
        hyper['params']['ckpt_path'] = f"{save_dir}/last.ckpt"
        print(hyper)
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(hyper, f, indent=4)
    count_parameters(bnr_model)
    sd = torch.load(ckpt_path, map_location='cpu')
    print("Diffusion Epoch:", sd['epoch'])
    print("Post Unet will be saved in:", save_dir)
    log_loss = open(f"{save_dir}/log_loss.txt", "w")
    model.load_state_dict(sd['state_dict'], strict=False)
    count_parameters(model)
    
    model = model.to(device)
    model.eval()
    light_vgg_loss = LightVGGPerceptualLoss().to(device)

    bnr_model.train()
    optimizer = torch.optim.AdamW(bnr_model.parameters(), lr=1e-5)
    
    for ep in range(opt.epochs):
        pbar = tqdm.tqdm(train_loader, dynamic_ncols=False, ascii=True) if opt.tqdm else train_loader
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            with torch.no_grad():
                x, c, latent_raw_image = model.get_input(batch, model.first_stage_key)
                t = torch.randint(0, 500, (x.shape[0],), device=model.device).long()
                noise = torch.randn_like(x)
                x_noisy = model.q_sample(x_start=x, c=c, t=t, noise=noise)
                model_output = model.apply_model(x_noisy, t, c)
                model_output_raw = model.decode_first_stage(model_output).detach()
                        
            target_raw = batch['image'].permute(0, 3, 1, 2).to(model.device)
            source_raw = batch['char_img'].permute(0, 3, 1, 2).to(model.device)
            
            output = bnr_model(model_output_raw, source_raw)  # B, 3, 128, 128
            loss_l1 = torch.nn.functional.l1_loss(output, target_raw)
            loss_edge = sobel_loss(output, target_raw)
            vgg_loss = light_vgg_loss(output, target_raw)
            mse = torch.nn.functional.mse_loss(output, target_raw)
            loss = loss_l1 + loss_edge + 0.05 * vgg_loss
            if torch.isnan(loss).any():
                print("Loss is NaN, skipping this batch")
                continue
            loss_info = f"LOSS:{round(loss.item(), 3)},L_l1={round(loss_l1.item(), 3)},L_edge={round(loss_edge.item(), 3)},vgg_loss={round(vgg_loss.item(),3)},MSE={round(mse.item(), 3)}"
            if opt.tqdm:
                pbar.set_description(loss_info)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bnr_model.parameters(), max_norm=1.0)
            optimizer.step()
            if i % 50 == 0:
                log_loss.write(f"Epoch {ep} step {i}: {loss_info}\n")
                if not opt.tqdm:
                    print(f"Epoch {ep}, Step {i}, {loss_info}")
        ckpt_dict = dict(state_dict=bnr_model.state_dict(), epoch=ep)
        if ep % opt.save_freq == 0:
            torch.save(ckpt_dict, f"{save_dir}/epoch={ep}.ckpt")
        torch.save(ckpt_dict, f"{save_dir}/last.ckpt")
        print(f"Epoch {ep}, Loss: {loss_info}")
