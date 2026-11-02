import torch
import torch.nn as nn
import kornia
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, ChineseCLIPModel, ChineseCLIPProcessor

try:
    import open_clip
except ImportError:
    open_clip = None
from ldm.util import default, count_params, autocast
import os

import time
import functools

def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"[Timing] Function `{func.__name__}` took {duration:.4f} seconds.")
        return result
    return wrapper


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77,
                 freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)

    
    
class ChineseCLIPEmbedder(nn.Module):
    def __init__(
        self,
        version, 
        device='cuda',
        output_dim=512,
        freeze=True,
    ):
        super().__init__()
        self.text_embedder = ChineseCLIPTextEmbedder(version, device, output_dim, freeze)
        self.image_embedder = ChineseCLIPImageEmbedder(version, device, output_dim, freeze)
        
    def encode_text(self, text, sep=False):
        if isinstance(text, list) and not isinstance(text[0], str):
            # return torch.stack([self.text_embedder(t) for t in text], dim=1)
            return torch.cat([self.text_embedder(t, sep) for t in text], dim=1)
        if isinstance(text, str):
            text = [text]
        return self.text_embedder(text, sep)
    
    def encode_image(self, image, *args, **kwargs):
        if isinstance(image, list):
            # image: List[bn x 3 x H x W, len=n]
            # return bn x n x output_dim
            n = len(image)
            bn, c, h, w = image[0].shape
            image_cat = torch.cat(image, dim=0)
            result = self.image_embedder(image_cat)
            result = result.view(n, bn, -1).permute(1, 0, 2)
            return result
        return self.image_embedder(image)
    
    
class ChineseCLIPEmbedderV2(ChineseCLIPEmbedder):
    def __init__(
        self,
        version, 
        device='cuda',
        output_dim=512,
        freeze=True,
        img_encoding_path="all_image_encoding",
        prompt_encoding_path="all_prompt_encoding",
    ):
        super().__init__(version, device, output_dim, freeze)
        self.img_encoding_path = img_encoding_path
        self.name2encoding = {}
        self.name2path = {}
        for name in os.listdir(img_encoding_path):
            name = name.split(".")[0]
            self.name2path[name] = os.path.join(self.img_encoding_path, name + ".pth")
    
    def encode_image(self, image, font_name_list):
        n, bn = len(font_name_list), len(font_name_list[0])
        projection = self.image_embedder.projection
        device = projection.weight.device
        result = []
        image_cat = torch.cat(image, dim=0)
        font_name_cat = [name for sublist in font_name_list for name in sublist]
        for img, font_name in zip(image_cat, font_name_cat):
            encoding_path = os.path.join(self.img_encoding_path, font_name + ".pth")
            flag = False
            if font_name in self.name2encoding:
                encoding = self.name2encoding[font_name].to(device)
                encoding = projection(encoding)
                encoding = encoding / encoding.norm(p=2, dim=-1, keepdim=True)
                flag = True
            elif font_name in self.name2path:
                try:
                    encoding = torch.load(self.name2path[font_name]).to(device)
                    flag = True
                    if torch.isnan(encoding).any():
                        print(f"Encoding {font_name} contains NaN, encode again")
                        flag = False
                    self.name2encoding[font_name] = encoding
                    encoding = projection(encoding)
                    encoding = encoding / encoding.norm(p=2, dim=-1, keepdim=True)
                    
                except Exception as e:
                    # print(f"Error loading encoding for {font_name}, encode again")
                    flag = False 
                
            if not flag:
                encoding, mid = self.image_embedder.encode(img, mid=True)
                self.name2encoding[font_name] = mid
                if torch.isnan(encoding).any():
                    print(f"Error encoding for {font_name}")
                    continue
                torch.save(mid, encoding_path)
                # print(f"Encoding {font_name} and save to {encoding_path}")
            result.append(encoding)
        result = torch.cat(result, dim=0).view(n, bn, -1).permute(1, 0, 2)
        return result
    
class ChineseCLIPTextEmbedder(AbstractEncoder):
    def __init__(
        self,
        version, 
        device='cuda',
        output_dim=512,
        freeze=True,
    ):
        super().__init__()
        self.processor = ChineseCLIPProcessor.from_pretrained(version)
        model = ChineseCLIPModel.from_pretrained(version)
        self.processor = ChineseCLIPProcessor.from_pretrained(version)
        self.model = model.text_model.to(device).train()
        if freeze:
            self.model = self.freeze(self.model)
        self.projection = nn.Linear(model.text_projection.in_features, output_dim).to(device)
            
        self.device = device
        
    def freeze(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
        return module.eval()
        
    def encode(self, text, sep=False):
        device = next(self.model.parameters()).device
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs).last_hidden_state
        if sep:
            outputs = outputs[:, 0].unsqueeze(1)
        outputs =  self.projection(outputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs
    
    def forward(self, text, sep=False):
        return self.encode(text, sep)
    
    
class ChineseCLIPImageEmbedder(AbstractEncoder):
    def __init__(
        self,
        version, 
        device='cuda',
        output_dim=512,
        freeze=True,
    ):
        super().__init__()
        self.processor = ChineseCLIPProcessor.from_pretrained(version)
        model = ChineseCLIPModel.from_pretrained(version)
        self.processor = ChineseCLIPProcessor.from_pretrained(version)
        self.model = model.vision_model.to(device).train()
        if freeze:
            self.model = self.freeze(self.model)
        self.projection = nn.Linear(model.visual_projection.in_features, output_dim).to(device)
            
        self.device = device
        
    def freeze(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
        return module.eval()
    
    def rescale(self, image):
        img_max = image.max()
        img_min = image.min()
        if not (img_max <= 1 and img_min >= 0):
            image = (image - img_min) / (img_max - img_min + 1e-6)
        return image
        
    def encode(self, image, mid=False):
        device = next(self.model.parameters()).device
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw_outputs = self.model(**inputs)[1]
        outputs =  self.projection(raw_outputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        if mid:
            return outputs, raw_outputs
        return outputs
    
    def forward(self, image):
        return self.encode(image)

class ClipImageEmbedder(nn.Module):
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=True,
            ucg_rate=0.
    ):
        super().__init__()
        from clip import load as load_clip
        self.model, _ = load_clip(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # re-normalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, no_dropout=False):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0. and not no_dropout:
            out = torch.bernoulli((1. - self.ucg_rate) * torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="pooled", antialias=True, ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'),
                                                            pretrained=version, )
        del model.transformer
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli((1. - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
              f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params.")

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


from ldm.modules.diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from ldm.modules.diffusionmodules.openaimodel import Timestep


class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=256, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_stats_path is None:
            clip_mean, clip_std = torch.zeros(timestep_dim), torch.ones(timestep_dim)
        else:
            clip_mean, clip_std = torch.load(clip_stats_path, map_location="cpu")
        self.register_buffer("data_mean", clip_mean[None, :], persistent=False)
        self.register_buffer("data_std", clip_std[None, :], persistent=False)
        self.time_embed = Timestep(timestep_dim)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean) * 1. / self.data_std
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std) + self.data_mean
        return x

    def forward(self, x, noise_level=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        else:
            assert isinstance(noise_level, torch.Tensor)
        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        return z, noise_level
