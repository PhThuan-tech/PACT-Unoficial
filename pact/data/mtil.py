"""The 11 MTIL datasets, built the same way as the ZSCL / MoE-Adapters benchmark.

- torchvision datasets with their official splits;
- Caltech101, EuroSAT and SUN397 have no official split: 80/20 `random_split` with
  `torch.Generator().manual_seed(42)`, as in ZSCL;
- class names: torchvision names with "_" -> " ", except Caltech101 / Flowers102 / EuroSAT
  which use CLIP's prompt names (classnames.py);
- templates: CLIP prompts.md; first one for training, ensemble of all for evaluation.

Note: StanfordCars and (sometimes) Caltech101 cannot be downloaded automatically by
torchvision any more; place them under `data_root` manually (see README).
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets

from . import classnames as cn

_CIFAR = ["a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
          "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
          "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.",
          "a photo of the {}.", "a blurry photo of the {}.", "a black and white photo of the {}.",
          "a low contrast photo of the {}.", "a high contrast photo of the {}.",
          "a bad photo of the {}.", "a good photo of the {}.", "a photo of the small {}.",
          "a photo of the big {}."]
_CALTECH = ["a photo of a {}.", "a painting of a {}.", "a plastic {}.", "a sculpture of a {}.",
            "a sketch of a {}.", "a tattoo of a {}.", "a toy {}.", "a rendition of a {}.",
            "a embroidered {}.", "a cartoon {}.", "a {} in a video game.", "a plushie {}.",
            "a origami {}.", "art of a {}.", "graffiti of a {}.", "a drawing of a {}.",
            "a doodle of a {}.", "a photo of the {}.", "a painting of the {}.", "the plastic {}.",
            "a sculpture of the {}.", "a sketch of the {}.", "a tattoo of the {}.", "the toy {}.",
            "a rendition of the {}.", "the embroidered {}.", "the cartoon {}.",
            "the {} in a video game.", "the plushie {}.", "the origami {}.", "art of the {}.",
            "graffiti of the {}.", "a drawing of the {}.", "a doodle of the {}."]

# Prompt templates of OpenAI CLIP `data/prompts.md` (also used by ZSCL / MoE-Adapters).
# Training uses the first one (train.template_mode=first); evaluation averages all of them
# (eval.template_mode=ensemble), as ZSCL's zeroshot_classifier(classnames, templates).
TEMPLATES = {
    "Aircraft": ["a photo of a {}, a type of aircraft.", "a photo of the {}, a type of aircraft."],
    "Caltech101": _CALTECH,
    "CIFAR100": _CIFAR,
    "DTD": ["a photo of a {} texture.", "a photo of a {} pattern.", "a photo of a {} thing.",
            "a photo of a {} object.", "a photo of the {} texture.", "a photo of the {} pattern.",
            "a photo of the {} thing.", "a photo of the {} object."],
    "EuroSAT": ["a centered satellite photo of {}.", "a centered satellite photo of a {}.",
                "a centered satellite photo of the {}."],
    "Flowers": ["a photo of a {}, a type of flower."],
    "Food": ["a photo of {}, a type of food."],
    "MNIST": ['a photo of the number: "{}".'],
    "OxfordPet": ["a photo of a {}, a type of pet."],
    "StanfordCars": ["a photo of a {}.", "a photo of the {}.", "a photo of my {}.", "i love my {}!",
                     "a photo of my dirty {}.", "a photo of my clean {}.", "a photo of my new {}.",
                     "a photo of my old {}."],
    "SUN397": ["a photo of a {}.", "a photo of the {}."],
}


def _clean(names):
    return [n.replace("_", " ") for n in names]


def _split(dataset, ratio=0.8):
    train_size = int(ratio * len(dataset))
    return random_split(dataset, [train_size, len(dataset) - train_size],
                        generator=torch.Generator().manual_seed(42))


def build_mtil_dataset(name, root, transform, download=True):
    """Returns (train_set, test_set, classnames, templates)."""
    if name == "Aircraft":
        train = datasets.FGVCAircraft(root, split="train", transform=transform, download=download)
        test = datasets.FGVCAircraft(root, split="test", transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "Caltech101":
        full = datasets.Caltech101(root, transform=transform, download=download)
        train, test = _split(full)
        names = list(cn.CALTECH101)
        assert len(names) == len(full.categories)
    elif name == "CIFAR100":
        train = datasets.CIFAR100(root, train=True, transform=transform, download=download)
        test = datasets.CIFAR100(root, train=False, transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "DTD":
        train = datasets.DTD(root, split="train", transform=transform, download=download)
        test = datasets.DTD(root, split="test", transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "EuroSAT":
        full = datasets.EuroSAT(root, transform=transform, download=download)
        train, test = _split(full)
        names = list(cn.EUROSAT)
    elif name == "Flowers":
        train = datasets.Flowers102(root, split="train", transform=transform, download=download)
        test = datasets.Flowers102(root, split="test", transform=transform, download=download)
        names = list(cn.FLOWERS102)
    elif name == "Food":
        train = datasets.Food101(root, split="train", transform=transform, download=download)
        test = datasets.Food101(root, split="test", transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "MNIST":
        train = datasets.MNIST(root, train=True, transform=transform, download=download)
        test = datasets.MNIST(root, train=False, transform=transform, download=download)
        names = _clean(train.classes)  # "0 - zero", kept as in ZSCL's MNIST class names
    elif name == "OxfordPet":
        train = datasets.OxfordIIITPet(root, split="trainval", transform=transform, download=download)
        test = datasets.OxfordIIITPet(root, split="test", transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "StanfordCars":
        train = datasets.StanfordCars(root, split="train", transform=transform, download=download)
        test = datasets.StanfordCars(root, split="test", transform=transform, download=download)
        names = _clean(train.classes)
    elif name == "SUN397":
        full = datasets.SUN397(root, transform=transform, download=download)
        train, test = _split(full)
        names = _clean(full.classes)
    else:
        raise ValueError(f"unknown MTIL dataset {name}")
    return train, test, names, TEMPLATES[name]


def get_targets(dataset):
    """Integer labels of a dataset without decoding images."""
    if isinstance(dataset, Subset):
        parent = get_targets(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    for attr in ("targets", "_labels", "y"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            return [int(v) for v in (values.tolist() if torch.is_tensor(values) else values)]
    if hasattr(dataset, "_samples"):  # StanfordCars
        return [int(label) for _, label in dataset._samples]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def few_shot_subset(dataset, shots, seed):
    """`shots` training samples per class (few-shot MTIL)."""
    by_class = defaultdict(list)
    for idx, label in enumerate(get_targets(dataset)):
        by_class[label].append(idx)
    rng = random.Random(seed)
    chosen = []
    for label in sorted(by_class):
        indices = by_class[label]
        chosen.extend(rng.sample(indices, min(shots, len(indices))))
    return Subset(dataset, sorted(chosen))


def eval_subset(dataset, max_samples, seed=0):
    """Fixed random subset of a test set, for quick evaluations."""
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    indices = np.random.RandomState(seed).choice(len(dataset), max_samples, replace=False)
    return Subset(dataset, sorted(indices.tolist()))
