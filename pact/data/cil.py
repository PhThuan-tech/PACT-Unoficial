"""Class-incremental splits of CIFAR-100 and TinyImageNet."""

import os

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from .mtil import get_targets


class TinyImageNet(Dataset):
    """`root/tiny-imagenet-200` with the official train / val folders (val = test set).

    Labels follow the sorted wnids (ImageFolder convention), which the standard class
    order file refers to.
    """

    def __init__(self, root, train=True, transform=None):
        base = os.path.join(root, "tiny-imagenet-200")
        with open(os.path.join(base, "wnids.txt")) as f:
            wnids = sorted(line.strip() for line in f if line.strip())
        words = {}
        with open(os.path.join(base, "words.txt"), encoding="utf-8") as f:
            for line in f:
                wnid, _, text = line.rstrip("\n").partition("\t")
                words[wnid] = text.split(",")[0].strip()
        self.classes = [words.get(w, w) for w in wnids]
        index = {w: i for i, w in enumerate(wnids)}
        self.samples = []
        if train:
            for wnid in wnids:
                folder = os.path.join(base, "train", wnid, "images")
                for name in sorted(os.listdir(folder)):
                    self.samples.append((os.path.join(folder, name), index[wnid]))
        else:
            with open(os.path.join(base, "val", "val_annotations.txt")) as f:
                for line in f:
                    parts = line.split("\t")
                    self.samples.append((os.path.join(base, "val", "images", parts[0]),
                                         index[parts[1]]))
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        return (self.transform(image) if self.transform else image), label


def build_cil_base(name, root, transform, download=True):
    if name == "CIFAR100":
        train = datasets.CIFAR100(root, train=True, transform=transform, download=download)
        test = datasets.CIFAR100(root, train=False, transform=transform, download=download)
        names = [c.replace("_", " ") for c in train.classes]
    elif name == "TinyImageNet":
        train = TinyImageNet(root, train=True, transform=transform)
        test = TinyImageNet(root, train=False, transform=transform)
        names = train.classes
    else:
        raise ValueError(f"unknown CIL dataset {name}")
    return train, test, names


ORDER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "class_orders")


def class_order(num_classes, seed):
    """Seeded random order (PyCIL style), used when cil.class_order == "seed"."""
    return np.random.RandomState(seed).permutation(num_classes).tolist()


def resolve_class_order(cfg, num_classes):
    spec = cfg["class_order"]
    if spec == "seed":
        order = class_order(num_classes, cfg["class_order_seed"])
    elif isinstance(spec, list):
        order = list(spec)
    else:
        path = spec if spec != "auto" else os.path.join(ORDER_DIR, cfg["dataset"].lower() + ".yaml")
        with open(path, encoding="utf-8") as f:
            order = yaml.safe_load(f)["class_order"]
    if sorted(order) != list(range(num_classes)):
        raise ValueError(f"class order must be a permutation of 0..{num_classes - 1}")
    return order


def task_splits(num_classes, initial, increment):
    tasks, start = [], 0
    sizes = [initial] + [increment] * ((num_classes - initial) // increment)
    for size in sizes:
        tasks.append(list(range(start, start + size)))
        start += size
    return tasks


class RemappedSubset(Dataset):
    """Samples of the given original classes, relabelled to their position in `order`."""

    def __init__(self, dataset, original_classes, order_index):
        wanted = set(original_classes)
        targets = get_targets(dataset)
        self.indices = [i for i, t in enumerate(targets) if t in wanted]
        self.dataset = dataset
        self.order_index = order_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.dataset[self.indices[i]]
        return image, self.order_index[int(label)]


def build_cil_tasks(cfg, root, transform, download=True):
    """Returns (task list, ordered class names). Labels are positions in the class order."""
    train, test, names = build_cil_base(cfg["dataset"], root, transform, download)
    order = resolve_class_order(cfg, len(names))
    order_index = {c: pos for pos, c in enumerate(order)}
    ordered_names = [names[c] for c in order]
    tasks = []
    for positions in task_splits(len(names), cfg["initial_classes"], cfg["increment"]):
        originals = [order[p] for p in positions]
        seen = list(range(positions[-1] + 1))
        tasks.append({
            "classes": positions,
            "seen": seen,
            "train": RemappedSubset(train, originals, order_index),
            "test_seen": RemappedSubset(test, [order[p] for p in seen], order_index),
        })
    return tasks, ordered_names

