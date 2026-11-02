# modified from font_ttf.py
import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from collections import defaultdict

from ldm.data.util import read_font, load_ttf_data, load_img_data, sample, render
from pathlib import Path
from itertools import chain
import random
from PIL import Image, ImageOps
import cv2
import torch
import numpy as np
from torchvision import transforms
import json

class BaseDataset(Dataset):
    def __init__(self, size, data_dirs, train_chars, extension="ttf", n_font=None):
        
        chars = json.load(open(train_chars))

        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        elif isinstance(data_dirs, list):
            self.data_dirs = data_dirs
        else:
            raise TypeError(f"The type of data_dirs is invalid: {type(data_dirs)}")

        self.use_ttf = (extension == "ttf")
        if self.use_ttf:
            self.load_ttf_data(chars, extension, n_font)
        else:
            self.load_img_data(chars, extension, n_font)

        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.n_fonts = len(self.keys)
        self.n_chars = len(self.chars)

        self.transform = transforms.Compose([
                            transforms.Resize((size, size)),
                            transforms.ToTensor(),
                        ])

    def load_ttf_data(self, chars, extension, n_font):
        self.key_font_dict, self.key_char_dict = load_ttf_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.get_img = self.render_from_ttf
        self.get_RGBimg = self.render_from_ttf
        self.get_Greyimg = self.render_from_ttf_Grey


    def load_img_data(self, chars, extension, n_font):
        self.key_dir_dict, self.key_char_dict = load_img_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.extension = extension
        self.get_img = self.load_img

    def render_from_ttf(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = img.convert("RGB") # current version only support H X W X 3 img, due to the vae_encoder
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def render_from_ttf_save(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img_raw = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
        img = self.transform(img_raw)
        img = img.permute(1,2,0)
        return img, img_raw
    
    def render_from_ttf_Grey(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img

    def load_img(self, key, char):
        img_dir = self.key_dir_dict[key][char]
        img = Image.open(str(img_dir / f"{char}.{self.extension}"))
        img = self.transform(img)
        return img
    
    def render_from_image(self, img_path):
        img = Image.open(img_path)
        img = self.transform(img).convert("RGB")
        img = img.permute(1,2,0)
        return img


class BaseTrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)

class FontTTFTrain(BaseTrainDataset):
    def __init__(self, source_path, keys=None, num_ref=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_char_dict, self.char_key_dict = self.filter_chars()
        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.n_chars = len(self.chars)  
        self.n_fonts = len(self.keys)
        self.num_ref = num_ref
        print("Training num_ref: ", self.num_ref)
        self.source = read_font(source_path)


    def render_from_source(self, char):
        img = render(self.source, char)
        img = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def render_from_source_Grey(self, char):
        img = render(self.source, char)
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys >= 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = set(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        index = i
        key, char = self.data_list[index]

        trg_img = self.get_RGBimg(key, char) # 128, 128, 3 | <key> Font

        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.num_ref)
        style_imgs = [self.get_img(key, c) for c in style_chars] # 3, 128, 128, 1  | 3 * <key> Font other chars

        char_img = self.render_from_source(char) # 128, 128, 3 | SimSun Font
        pad_img = torch.zeros_like(style_imgs[0])
        style_imgs += [pad_img] * (3 - len(style_imgs))
        
        example = dict()
        example['key_char'] = f"{key}_{char}"
        example['style_imgs_name'] = [f"{key}_{c}" for c in style_chars]
        example["image"] = trg_img # 128, 128, 3
        example["style_imgs"] = style_imgs # 128, 128, 3
        example["char_img"] = char_img # 128, 128, 3

        return example
    
    
class FontTTFPredict(BaseTrainDataset):
    def __init__(self, source_path, ref_json_path=None, ref_img_dir=None, keys=None, num_ref=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_char_dict, self.char_key_dict = self.filter_chars()
        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.style_imgs = []
        self.style_chars = []
        if ref_json_path and ref_img_dir:
            raise ValueError("Please provide either ref_json_path or ref_img_dir, not both.")
        if ref_json_path:
            self.style_chars = json.load(open(ref_json_path))
            print("Predict reference strategy: ", f"style chars {self.style_chars}")
        elif ref_img_dir:
            self.ref_img_dir = ref_img_dir
            for name in os.listdir(ref_img_dir):
                if name.endswith(".png"):
                    img = Image.open(os.path.join(ref_img_dir, name)).convert("RGB")
                    img = self.transform(img)
                    img = img.permute(1,2,0)
                    self.style_imgs.append(img)
                    self.style_chars.append(name.split(".")[0])
            print("Predict reference strategy: ", f"{len(self.style_imgs)} style imgs in {ref_img_dir}")
        else:
            if num_ref >= 0:
                print("Predict reference strategy: ", f"random {num_ref} chars")
            else:
                print("Predict reference strategy: ", f"all chars")
        
        self.n_chars = len(self.chars)  
        self.n_fonts = len(self.keys)   
        self.source = read_font(source_path)
        self.num_ref = num_ref
        
        
    def __len__(self):
        return len(self.data_list)
    
    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys >= 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = set(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict
        
    
    def render_from_source(self, char):
        img = render(self.source, char)
        img = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
        
    def __getitem__(self, i):
        index = i
        key, char = self.data_list[index]
        trg_img = self.get_RGBimg(key, char)
        
        if self.style_imgs:  # style images is given
            style_imgs = self.style_imgs
            style_chars = self.style_chars
        elif not self.style_chars:  # nothing is given, use random reference
            style_chars = set(self.key_char_dict[key]).difference({char})
            if self.num_ref >= 0:
                style_chars = sample(list(style_chars), self.num_ref)
            else:
                style_chars = list(style_chars)
            style_imgs = [self.get_img(key, c) for c in style_chars]
        else:  # style chars is given
            style_chars = self.style_chars
            style_imgs = [self.get_img(key, c) for c in style_chars]
        char_img = self.render_from_source(char)
        example = dict()
        example["chars"] = char
        example["font"] = key
        example['style_imgs_name'] = [f"{key}_{c}" for c in style_chars]
        example["image"] = trg_img # 128, 128, 3
        example["style_imgs"] = style_imgs # 128, 128, 1
        example["char_img"] = char_img # 128, 128, 3


        return example


def _load_char_list(chars):
    if isinstance(chars, (list, tuple)):
        return list(chars)
    if os.path.isfile(chars):
        with open(chars, encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, list):
            raise ValueError(f"Character JSON must contain a list: {chars}")
        return value
    return list(chars)


def _reference_char(path):
    """Use the filename stem as the character label (for example: 永.png)."""
    stem = Path(path).stem
    if not stem:
        raise ValueError(f"Reference image has an empty filename: {path}")
    return stem.split("_")[0]


def _prepare_reference_image(path, size):
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return transforms.ToTensor()(canvas).permute(1, 2, 0)


class ReferenceImagePredict(Dataset):
    """Inference dataset that needs only reference images and target chars."""

    def __init__(self, source_path, ref_img_dir, target_chars, size=128, font_name="reference"):
        self.font_name = font_name
        self.chars = _load_char_list(target_chars)
        if not self.chars:
            raise ValueError("At least one target character is required")
        paths = sorted(
            p for p in Path(ref_img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not paths:
            raise ValueError(f"No reference images found in {ref_img_dir}")
        self.style_imgs = [_prepare_reference_image(p, size) for p in paths]
        self.style_names = [f"{font_name}_{_reference_char(p)}" for p in paths]
        self.source = read_font(source_path)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.chars)

    def _source_image(self, char):
        image = render(self.source, char).convert("RGB")
        return self.transform(image).permute(1, 2, 0)

    def __getitem__(self, index):
        char = self.chars[index]
        char_img = self._source_image(char)
        return {
            "image": char_img,
            "char_img": char_img,
            "style_imgs": self.style_imgs,
            "style_imgs_name": self.style_names,
            "chars": char,
            "font": self.font_name,
        }


class ReferenceImageFinetune(Dataset):
    """Supervised few-shot dataset built directly from named reference images."""

    def __init__(self, source_path, ref_img_dir, size=128, num_ref=-1, repeat=1):
        paths = sorted(
            p for p in Path(ref_img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not paths:
            raise ValueError(f"No reference images found in {ref_img_dir}")
        self.items = [(_reference_char(p), _prepare_reference_image(p, size)) for p in paths]
        self.source = read_font(source_path)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.num_ref = num_ref
        self.repeat = max(1, int(repeat))

    def __len__(self):
        return len(self.items) * self.repeat

    def _source_image(self, char):
        image = render(self.source, char).convert("RGB")
        return self.transform(image).permute(1, 2, 0)

    def __getitem__(self, index):
        char, target = self.items[index % len(self.items)]
        refs = [(c, image) for c, image in self.items if c != char]
        if not refs:
            refs = [(char, target)]
        if self.num_ref > 0:
            refs = sample(refs, self.num_ref)
        return {
            "key_char": f"reference_{char}",
            "image": target,
            "char_img": self._source_image(char),
            "style_imgs": [image for _, image in refs],
            "style_imgs_name": [f"reference_{c}" for c, _ in refs],
        }


class FontTTFFinetune(FontTTFTrain):
    """TTF-backed finetuning dataset kept for older saved configs."""

    pass
    
