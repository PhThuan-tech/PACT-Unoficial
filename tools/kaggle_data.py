"""Find the PACT datasets inside attached Kaggle inputs and lay them out for torchvision.

Kaggle copies of the datasets use many different folder layouts, and /kaggle/input is
read-only. This script walks the input folder once, recognises each dataset by its
characteristic files, then builds `data_root` with the exact layout torchvision expects,
using symlinks (no copy, no extra disk). Small archives (MNIST .gz, CIFAR .tar.gz) are
extracted into data_root when only the archive is present.

    python tools/kaggle_data.py --input /kaggle/input --data-root /kaggle/working/data
    python tools/kaggle_data.py ... --datasets DTD EuroSAT --check
    python tools/kaggle_data.py ... --map SUN397=/kaggle/input/my-sun397   # restrict a search

Exit code 1 if a requested dataset is missing or fails the check.
"""

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tarfile

MTIL = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers", "Food", "MNIST",
        "OxfordPet", "StanfordCars", "SUN397"]
ALL = MTIL + ["TinyImageNet"]

# (train, test) sizes with the splits used by the repo (80/20 seed-42 split where needed).
# Reference only: a mismatch is reported as a warning.
EXPECTED = {
    "Aircraft": (3334, 3333), "Caltech101": (6941, 1736), "CIFAR100": (50000, 10000),
    "DTD": (1880, 1880), "EuroSAT": (21600, 5400), "Flowers": (1020, 6149),
    "Food": (75750, 25250), "MNIST": (60000, 10000), "OxfordPet": (3680, 3669),
    "StanfordCars": (8144, 8041), "SUN397": (87003, 21751), "TinyImageNet": (100000, 10000),
}

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
MNIST_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
MNIST_EXCLUDE = ("fashion", "kuzushiji", "kmnist", "emnist", "qmnist", "overhead")


# ----------------------------------------------------------------------------- scanning
class Index:
    """One os.walk over the input; detectors read from these tables."""

    def __init__(self):
        self.dirs = []          # (path, set(subdir names), set(file names))
        self.files = {}         # file name -> [paths] (only names some detector cares about)

    WANTED_FILE = re.compile(
        r"^(images_variant_(train|test)\.txt|imagelabels\.mat|setid\.mat|cars_meta\.mat|"
        r"cars_train_annos\.mat|cars_test_annos_withlabels.*\.mat|ClassName\.txt|"
        r"cifar-100-python\.tar\.gz|(train|t10k)-(images|labels)[-.]idx[13]-ubyte(\.gz)?)$")

    def scan(self, root, max_depth=8):
        root = os.path.abspath(root)
        base_depth = root.rstrip(os.sep).count(os.sep)
        for path, dirnames, filenames in os.walk(root):
            if "__MACOSX" in path:
                dirnames[:] = []
                continue
            if path.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []
            names = set(filenames)
            self.dirs.append((path, set(dirnames), names))
            for name in filenames:
                if self.WANTED_FILE.match(name):
                    self.files.setdefault(name, []).append(os.path.join(path, name))
        return self

    def find_dirs(self, predicate):
        return sorted((p for p, d, f in self.dirs if predicate(p, d, f)), key=_rank)

    def find_files(self, pattern):
        rx = re.compile(pattern)
        hits = [p for name, paths in self.files.items() if rx.match(name) for p in paths]
        return sorted(hits, key=_rank)


def _rank(path):
    """Prefer short, plain paths."""
    return (path.count(os.sep), len(path), path)


def _has(path, *parts):
    return os.path.exists(os.path.join(path, *parts))


def _first(items):
    return items[0] if items else None


def _closest(paths, anchor):
    """Path among `paths` sharing the longest prefix with `anchor` (same dataset copy)."""
    if not paths:
        return None
    if not anchor:
        return paths[0]  # already sorted by _rank
    return max(paths, key=lambda p: (len(os.path.commonpath([p, anchor])), -len(p)))


# ----------------------------------------------------------------------------- detectors
# Each returns (plan, note): plan = list of (relative target in data_root, source path) or
# special actions; None if the dataset is not found.

def detect_aircraft(ix):
    data = _first(ix.find_dirs(lambda p, d, f: {"images_variant_train.txt",
                                                "images_variant_test.txt",
                                                "variants.txt"} <= f and "images" in d))
    if not data:
        return None, "need a folder with images/, variants.txt, images_variant_{train,test}.txt"
    return [("fgvc-aircraft-2013b/data", data)], data


def detect_caltech101(ix):
    cats = _first(ix.find_dirs(lambda p, d, f: {"accordion", "airplanes", "Faces_easy"} <= d))
    if not cats:
        return None, "need 101_ObjectCategories/ (folders accordion, airplanes, Faces_easy, ...)"
    plan = [(f"caltech101/101_ObjectCategories/{c}", os.path.join(cats, c))
            for c in sorted(os.listdir(cats)) if os.path.isdir(os.path.join(cats, c))]
    if not _has(cats, "BACKGROUND_Google"):  # torchvision removes it from the class list
        plan.append(("caltech101/101_ObjectCategories/BACKGROUND_Google", "@emptydir"))
    return plan, cats


def detect_cifar100(ix):
    found = ix.find_dirs(lambda p, d, f: {"train", "test", "meta"} <= f
                         and "data_batch_1" not in f)
    named = [p for p in found if os.path.basename(p) == "cifar-100-python"]
    folder = _first(named or found)
    if folder:
        return [("cifar-100-python", folder)], folder
    archive = _first(ix.find_files(r"^cifar-100-python\.tar\.gz$"))
    if archive:
        return [(".", "@untar:" + archive)], archive
    return None, "need cifar-100-python/ (train, test, meta pickles) or its .tar.gz"


def detect_dtd(ix):
    folder = _first(ix.find_dirs(lambda p, d, f: {"images", "labels"} <= d
                                 and _has(p, "labels", "train1.txt")))
    if not folder:
        return None, "need dtd/ with images/ and labels/train1.txt"
    return [("dtd/dtd", folder)], folder


def detect_eurosat(ix):
    folder = _first(ix.find_dirs(lambda p, d, f: set(EUROSAT_CLASSES) <= d))
    if not folder:
        return None, "need the 10 EuroSAT RGB class folders (AnnualCrop, Forest, ...)"
    sample = os.listdir(os.path.join(folder, "Forest"))[:20]
    if not any(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in sample):
        return None, f"{folder}: class folders hold no .jpg (multispectral .tif copy?); use EuroSAT RGB"
    return [(f"eurosat/2750/{c}", os.path.join(folder, c)) for c in EUROSAT_CLASSES], folder


def detect_flowers(ix):
    jpg = _first(ix.find_dirs(lambda p, d, f: {"image_00001.jpg", "image_08189.jpg"} <= f))
    labels = _closest(ix.find_files(r"^imagelabels\.mat$"), jpg or "")
    setid = _closest(ix.find_files(r"^setid\.mat$"), jpg or "")
    if not (jpg and labels and setid):
        return None, "need jpg/ (image_00001.jpg ... image_08189.jpg), imagelabels.mat, setid.mat"
    return [("flowers-102/jpg", jpg), ("flowers-102/imagelabels.mat", labels),
            ("flowers-102/setid.mat", setid)], jpg


def detect_food(ix):
    images = _first(ix.find_dirs(lambda p, d, f: {"apple_pie", "baby_back_ribs", "waffles"} <= d))
    meta = _closest(ix.find_dirs(lambda p, d, f: {"train.json", "test.json", "classes.txt"} <= f),
                    images or "")
    if not (images and meta):
        return None, "need food-101 images/ (101 class folders) and meta/ (train.json, test.json)"
    return [("food-101/images", images), ("food-101/meta", meta)], images


def detect_mnist(ix):
    plan, srcs = [], []
    for name in MNIST_FILES:
        stem = name.replace("-idx", "[-.]idx")
        cands = [p for p in ix.find_files(rf"^{stem}(\.gz)?$")
                 if not any(x in p.lower() for x in MNIST_EXCLUDE)]
        cands.sort(key=lambda p: (p.endswith(".gz"), "mnist" not in p.lower(), _rank(p)))
        if not cands:
            return None, "need the 4 MNIST idx files (train-images-idx3-ubyte, ...), plain or .gz"
        src = cands[0]
        plan.append((f"MNIST/raw/{name}", ("@gunzip:" + src) if src.endswith(".gz") else src))
        srcs.append(src)
    return plan, os.path.dirname(srcs[0])


def detect_pets(ix):
    anns = _first(ix.find_dirs(lambda p, d, f: {"trainval.txt", "test.txt", "list.txt"} <= f))
    images = _closest(ix.find_dirs(lambda p, d, f: {"Abyssinian_1.jpg",
                                                    "staffordshire_bull_terrier_1.jpg"} <= f),
                      anns or "")
    if not (anns and images):
        return None, "need images/ (Abyssinian_1.jpg, ...) and annotations/ (trainval.txt, test.txt)"
    return [("oxford-iiit-pet/images", images), ("oxford-iiit-pet/annotations", anns)], images


def detect_cars(ix):
    devkit = _first(ix.find_dirs(lambda p, d, f: {"cars_meta.mat", "cars_train_annos.mat"} <= f))
    anchor = devkit or ""
    test_labels = _closest(ix.find_files(r"^cars_test_annos_withlabels.*\.mat$"), anchor)
    train = _closest(ix.find_dirs(lambda p, d, f: {"00001.jpg", "08144.jpg"} <= f), anchor)
    test = _closest(ix.find_dirs(lambda p, d, f: "08041.jpg" in f and "08042.jpg" not in f),
                    anchor)
    missing = [n for n, v in [("devkit (cars_meta.mat, cars_train_annos.mat)", devkit),
                              ("cars_test_annos_withlabels.mat", test_labels),
                              ("cars_train/ (00001.jpg..08144.jpg)", train),
                              ("cars_test/ (00001.jpg..08041.jpg)", test)] if not v]
    if missing:
        return None, "missing " + ", ".join(missing)
    return [("stanford_cars/devkit", devkit), ("stanford_cars/cars_train", train),
            ("stanford_cars/cars_test", test),
            ("stanford_cars/cars_test_annos_withlabels.mat", test_labels)], devkit


def detect_sun397(ix):
    folder = _first(ix.find_dirs(lambda p, d, f: "ClassName.txt" in f and {"a", "b"} <= d))
    if folder:
        return [("SUN397", folder)], folder
    if ix.find_dirs(lambda p, d, f: {"a", "b", "c"} <= d and _has(p, "a", "abbey")):
        return None, ("found the SUN397 image tree but not ClassName.txt next to a/, b/, ...: "
                      "torchvision needs the original layout (SUN397/ClassName.txt + SUN397/a/...)")
    return None, "need SUN397/ with ClassName.txt and the a/ ... z/ image folders"


def detect_tiny(ix):
    folder = _first(ix.find_dirs(lambda p, d, f: {"wnids.txt", "words.txt"} <= f
                                 and {"train", "val"} <= d))
    if not folder:
        return None, "need tiny-imagenet-200/ (wnids.txt, words.txt, train/, val/)"
    if not _has(folder, "val", "val_annotations.txt"):
        return None, f"{folder}: val/val_annotations.txt missing (re-organised copy?)"
    return [("tiny-imagenet-200", folder)], folder


DETECTORS = {
    "Aircraft": detect_aircraft, "Caltech101": detect_caltech101, "CIFAR100": detect_cifar100,
    "DTD": detect_dtd, "EuroSAT": detect_eurosat, "Flowers": detect_flowers, "Food": detect_food,
    "MNIST": detect_mnist, "OxfordPet": detect_pets, "StanfordCars": detect_cars,
    "SUN397": detect_sun397, "TinyImageNet": detect_tiny,
}


# ----------------------------------------------------------------------------- layout
def _remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def apply_plan(plan, data_root, link_mode):
    for rel, src in plan:
        dst = os.path.normpath(os.path.join(data_root, rel))
        if rel != ".":
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _remove(dst)
        if src == "@emptydir":
            os.makedirs(dst, exist_ok=True)
        elif src.startswith("@gunzip:"):
            with gzip.open(src[len("@gunzip:"):], "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        elif src.startswith("@untar:"):
            with tarfile.open(src[len("@untar:"):]) as tar:
                try:
                    tar.extractall(data_root, filter="data")  # Python >= 3.11.4
                except TypeError:
                    tar.extractall(data_root)
        elif link_mode == "symlink":
            os.symlink(src, dst, target_is_directory=os.path.isdir(src))
        elif os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# ----------------------------------------------------------------------------- check
def check_dataset(name, data_root):
    """Load with the repo's own builders (download=False) and read one sample."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if name == "TinyImageNet":
        from pact.data.cil import TinyImageNet
        train, test = TinyImageNet(data_root, True), TinyImageNet(data_root, False)
        classes = len(train.classes)
    else:
        from pact.data.mtil import build_mtil_dataset
        train, test, classnames, _ = build_mtil_dataset(name, data_root, None, download=False)
        classes = len(classnames)
    train[0], test[len(test) - 1]  # decode one image from each split
    return len(train), len(test), classes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--data-root", default="/kaggle/working/data")
    ap.add_argument("--datasets", nargs="*", default=["mtil"],
                    help="names, or 'mtil' (11 MTIL datasets) / 'all' (+ TinyImageNet)")
    ap.add_argument("--map", nargs="*", default=[], help="Name=/path: search only there")
    ap.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--check", action="store_true", help="load each dataset after linking")
    args = ap.parse_args()

    wanted = []
    for item in args.datasets:
        wanted += MTIL if item == "mtil" else ALL if item == "all" else [item]
    unknown = [n for n in wanted if n not in DETECTORS]
    if unknown:
        ap.error(f"unknown dataset(s) {unknown}; choose from {ALL}")
    overrides = dict(item.split("=", 1) for item in args.map)

    print(f"scanning {args.input} ...", flush=True)
    main_index = Index().scan(args.input)
    os.makedirs(args.data_root, exist_ok=True)
    manifest, failed = {}, []
    for name in wanted:
        ix = Index().scan(overrides[name]) if name in overrides else main_index
        plan, info = DETECTORS[name](ix)
        if plan is None:
            print(f"  [MISSING] {name:12s} {info}")
            failed.append(name)
            continue
        apply_plan(plan, args.data_root, args.link_mode)
        manifest[name] = info
        line = f"  [linked ] {name:12s} <- {info}"
        if args.check:
            try:
                n_train, n_test, n_cls = check_dataset(name, args.data_root)
                exp = EXPECTED[name]
                flag = "" if (n_train, n_test) == exp else f"   (WARN: expected {exp[0]}/{exp[1]})"
                line = (f"  [OK     ] {name:12s} train={n_train} test={n_test} "
                        f"classes={n_cls}{flag}\n             <- {info}")
            except Exception as exc:  # report and continue with the other datasets
                line = f"  [ERROR  ] {name:12s} {type(exc).__name__}: {exc}\n             <- {info}"
                failed.append(name)
        print(line, flush=True)

    with open(os.path.join(args.data_root, "_sources.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ok = [n for n in wanted if n not in failed]
    print(f"\nready: {ok}\nnot ready: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
