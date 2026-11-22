"""Test tools/kaggle_data.py on a fake /kaggle/input with messy, Kaggle-like layouts.

    python tests/kaggle_data_test.py

Uses --link-mode copy (symlinks need admin rights on Windows); on Kaggle the default
symlink mode creates the same tree. Datasets whose torchvision loader checks MD5 sums or
needs real .mat files (CIFAR100, Flowers, StanfordCars) and Caltech101 (needs all 101
classes) are tested for detection and layout only; the others are also loaded (--check).
"""
import gzip
import io
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOOL = os.path.join(ROOT, "tools", "kaggle_data.py")


def jpg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), (120, 30, 200)).save(path)


def text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def blob(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def idx_images(n):
    return struct.pack(">IIII", 2051, n, 28, 28) + bytes(n * 28 * 28)


def idx_labels(n):
    return struct.pack(">II", 2049, n) + bytes(range(n))


def build(inp):
    # Aircraft: original archive one level down
    d = f"{inp}/fgvc-aircraft/fgvc-aircraft-2013b/data"
    for i in range(1, 5):
        jpg(f"{d}/images/{i:07d}.jpg")
    text(f"{d}/variants.txt", "707-320\nA300B4\n")
    text(f"{d}/images_variant_train.txt", "0000001 707-320\n0000002 A300B4\n")
    text(f"{d}/images_variant_test.txt", "0000003 707-320\n0000004 A300B4\n")
    # DTD (jmexpert style: dtd/{images,imdb,labels})
    d = f"{inp}/describable-textures-dataset-dtd/dtd"
    for i in (1, 2):
        jpg(f"{d}/images/banded/banded_000{i}.jpg")
    os.makedirs(f"{d}/imdb", exist_ok=True)
    text(f"{d}/labels/train1.txt", "banded/banded_0001.jpg\n")
    text(f"{d}/labels/test1.txt", "banded/banded_0002.jpg\n")
    # EuroSAT RGB with extra csv files next to the class folders
    d = f"{inp}/eurosat-rgb/EuroSAT/2750"
    for c in ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
              "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]:
        jpg(f"{d}/{c}/{c}_1.jpg")
    text(f"{inp}/eurosat-rgb/EuroSAT/train.csv", "x\n")
    # Food-101 nested twice
    d = f"{inp}/food-101/food-101/food-101"
    cls = ["apple_pie", "baby_back_ribs", "waffles"]
    for c in cls:
        jpg(f"{d}/images/{c}/1.jpg")
        jpg(f"{d}/images/{c}/2.jpg")
    text(f"{d}/meta/train.json", json.dumps({c: [f"{c}/1"] for c in cls}))
    text(f"{d}/meta/test.json", json.dumps({c: [f"{c}/2"] for c in cls}))
    text(f"{d}/meta/classes.txt", "\n".join(cls))
    # MNIST, hojjatk style (file inside a folder of the same name, dotted names), one .gz,
    # plus a Fashion-MNIST decoy with the canonical names
    d = f"{inp}/mnist-dataset"
    blob(f"{d}/train-images-idx3-ubyte/train-images-idx3-ubyte", idx_images(3))
    blob(f"{d}/train-labels.idx1-ubyte", idx_labels(3))
    blob(f"{d}/t10k-images.idx3-ubyte", idx_images(2))
    blob(f"{d}/t10k-labels-idx1-ubyte.gz", gzip.compress(idx_labels(2)))
    blob(f"{inp}/fashionmnist/t10k-labels-idx1-ubyte", idx_labels(9))
    # Oxford Pets with images/ and annotations/ in separate folders
    jpg(f"{inp}/pets/images/images/Abyssinian_1.jpg")
    jpg(f"{inp}/pets/images/images/staffordshire_bull_terrier_1.jpg")
    a = f"{inp}/pets/annotations/annotations"
    text(f"{a}/trainval.txt", "Abyssinian_1 1 1 1\n")
    text(f"{a}/test.txt", "staffordshire_bull_terrier_1 35 1 35\n")
    text(f"{a}/list.txt", "#\n")
    # SUN397 original layout
    d = f"{inp}/sun397/SUN397"
    text(f"{d}/ClassName.txt", "/a/abbey\n/b/bakery/shop\n")
    for i in range(3):
        jpg(f"{d}/a/abbey/sun_a{i}.jpg")
        jpg(f"{d}/b/bakery/shop/sun_b{i}.jpg")
    # TinyImageNet
    d = f"{inp}/tiny-imagenet/tiny-imagenet-200"
    text(f"{d}/wnids.txt", "n01\n")
    text(f"{d}/words.txt", "n01\tgoldfish, Carassius auratus\n")
    jpg(f"{d}/train/n01/images/n01_0.JPEG")
    jpg(f"{d}/val/images/val_0.JPEG")
    text(f"{d}/val/val_annotations.txt", "val_0.JPEG\tn01\t0\t0\t1\t1\n")
    # --- layout-only datasets
    for c in ["accordion", "airplanes", "Faces_easy"]:
        jpg(f"{inp}/caltech101/101_ObjectCategories/101_ObjectCategories/{c}/image_0001.jpg")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("train", "test", "meta"):
            info = tarfile.TarInfo(f"cifar-100-python/{name}")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    blob(f"{inp}/cifar100-archive/cifar-100-python.tar.gz", buf.getvalue())
    jpg(f"{inp}/flowers/102flowers/jpg/image_00001.jpg")
    jpg(f"{inp}/flowers/102flowers/jpg/image_08189.jpg")
    blob(f"{inp}/flowers/labels/imagelabels.mat")
    blob(f"{inp}/flowers/labels/setid.mat")
    c = f"{inp}/stanford-cars"
    blob(f"{c}/car_devkit/devkit/cars_meta.mat")
    blob(f"{c}/car_devkit/devkit/cars_train_annos.mat")
    blob(f"{c}/cars_test_annos_withlabels (1).mat")
    for n in ("00001", "08041", "08042", "08144"):
        jpg(f"{c}/cars_train/cars_train/{n}.jpg")
    for n in ("00001", "08041"):
        jpg(f"{c}/cars_test/cars_test/{n}.jpg")


def run(*args):
    proc = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr)
    return proc.returncode


def main():
    tmp = tempfile.mkdtemp()
    inp, data = os.path.join(tmp, "input"), os.path.join(tmp, "data")
    build(inp)
    common = ["--input", inp, "--data-root", data, "--link-mode", "copy"]

    code = run(*common, "--check", "--datasets", "Aircraft", "DTD", "EuroSAT", "Food", "MNIST",
               "OxfordPet", "SUN397", "TinyImageNet")
    assert code == 0, "checked datasets must all load"

    code = run(*common, "--datasets", "Caltech101", "CIFAR100", "Flowers", "StanfordCars")
    assert code == 0, "layout-only datasets must all be detected"
    for rel in ["caltech101/101_ObjectCategories/accordion",
                "caltech101/101_ObjectCategories/BACKGROUND_Google",
                "cifar-100-python/meta", "flowers-102/jpg/image_08189.jpg",
                "flowers-102/setid.mat", "stanford_cars/devkit/cars_meta.mat",
                "stanford_cars/cars_test_annos_withlabels.mat",
                "stanford_cars/cars_train/08144.jpg", "stanford_cars/cars_test/08041.jpg"]:
        assert os.path.exists(os.path.join(data, rel)), rel
    assert not os.path.exists(os.path.join(data, "stanford_cars/cars_test/08144.jpg"))
    with open(os.path.join(data, "MNIST/raw/t10k-labels-idx1-ubyte"), "rb") as f:
        assert struct.unpack(">II", f.read(8))[1] == 2, "picked the Fashion-MNIST decoy"

    code = run(*common, "--datasets", "SUN397", "--map", f"SUN397={inp}/eurosat-rgb")
    assert code == 1, "a missing dataset must give exit code 1"
    print("ALL OK")


if __name__ == "__main__":
    main()
