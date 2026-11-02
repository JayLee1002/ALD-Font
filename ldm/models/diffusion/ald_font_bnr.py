from .ddpm import LatentDiffusion
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig
from PIL import Image
import torch.nn.functional as F
import time
import cv2

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.models.diffusion.ald_font import ALDFont
from ldm.models.diffusion.ald_font import ALDFont_Warper


class ALDFont_with_BNR(ALDFont):
    def __init__(self, style_stage_config, post_stage_config, *args, **kwargs):
        super().__init__(style_stage_config, *args, **kwargs)
        self.model = ALDFont_Warper(kwargs['unet_config'], kwargs['conditioning_key'])
        self.instantiate_style_stage(style_stage_config)
        self.init_from_ckpt(kwargs['ckpt_path'])
        self.instantiate_post_stage(post_stage_config)
    
    def instantiate_post_stage(self, config):
        model = instantiate_from_config(config)
        self.post_stage_model = model
        
    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        ## reload from pretrained prior model <model>
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())

        # #################################################
        # # ignore attn2 parameters, load pretrained diffuser model (e.g., SD released model)
        # ignore_keys = []
        # for k in keys:
        #     if 'attn2' in k:
        #         ignore_keys.append(k)

        # # ignore unet parameters, training from scratch for diffuser
        # ignore_keys = []
        # for k in keys:
        #     if 'diffusion_model' in k:
        #         ignore_keys.append(k)
        # print('ignore_keys', ignore_keys)
        # #################################################

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        
    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, return_x=False):
        x = self.get_input_ori(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        raw_image = self.get_input_ori(batch, "char_img").to(self.device) 
        latent_raw_image = self.encode_first_stage(raw_image)
        latent_raw_image = self.get_first_stage_encoding(latent_raw_image).detach()
        
        
        style_imgs = batch['style_imgs']
        style_imgs = [img.permute(0, 3, 1, 2) for img in style_imgs]  # bn, 128, 128, 3 -> bn, 3, 128, 128
        encode_style_images = self.style_stage_model.encode_image(style_imgs, batch['style_imgs_name'])


        c = dict(encode_style_image=encode_style_images, 
                 latent_raw_image=latent_raw_image, 
                 raw_image=raw_image,  # Source Font in pixel space
                 raw_trg=x)  # Target Font in pixel space

        out = [z, c, latent_raw_image]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_x:
            out.extend([x])
        if return_original_cond:
            out.append(raw_image)
        return out
    
    def _progressive_denoising_impl(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        img, intermediates = super(ALDFont_with_BNR, self).progressive_denoising(
            cond, shape, verbose=verbose, callback=callback, quantize_denoised=quantize_denoised,
            img_callback=img_callback, mask=mask, x0=x0, temperature=temperature, noise_dropout=noise_dropout,
            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, batch_size=batch_size,
            x_T=x_T, start_T=start_T, log_every_t=log_every_t
        )
        raw_image_source = cond['raw_image']  # B, 3, 128, 128
        out = self.post_stage_model(img, raw_image_source) # B, 3, 128, 128
        return (out, img), intermediates
    
    @torch.no_grad()
    def progressive_denoising(self, *args, **kwargs):
        return self._progressive_denoising_impl(*args, **kwargs)
    
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, c=cond, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # logvar_t = self.logvar[t].to(self.device)
        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(self.style_stage_model.parameters()) + list(self.post_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
