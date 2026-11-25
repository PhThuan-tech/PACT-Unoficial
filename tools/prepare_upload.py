"""Build the file to upload to Kaggle as ONE dataset (run on a machine with internet).

    python tools/prepare_upload.py                              # everything (downloads ~600 MB)
    python tools/prepare_upload.py --skip-clip --skip-wheels    # reuse dist/pact-kit/, refresh code

Output (Kaggle only accepts files / archives, not folders):
    dist/pact-code.zip       <- upload as dataset `pact-code`, a new version whenever code changes
        PACT/                    this repository (+ BUILD_INFO.json: newest code wins on Kaggle)
        wheels/*.whl             open_clip_torch + ftfy (pure Python, missing from the Kaggle image)
        pact_demo.ipynb, UPLOAD.md
    dist/pact-clip.zip       <- upload once as dataset `pact-clip` (an older pact-kit dataset
                                already contains the same weights)
        clip/ViT-B-16-quickgelu-openai.pt   CLIP ViT-B/16 OpenAI weights (state_dict, ~600 MB)
        clip/reference.pt        features for the offline check
    dist/pact_demo.ipynb     <- import this into a Kaggle notebook
    dist/pact-kit/           working folder (wheels/, clip/ are reused by --skip-*)

Your global Python environment is not modified: if open_clip is not installed, it is
installed into dist/.pylib just for the export.
"""

import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INCLUDE = ["pact", "configs", "tools", "tests", "kaggle", "analysis", "README.md", "requirements.txt"]
OPEN_CLIP_SPEC = "open_clip_torch>=2.20,<3.0"
STORED = (".pt", ".whl", ".zip")  # already compressed / incompressible: store, do not deflate

GUIDE = """# Đưa PACT lên Kaggle

1. Trọng số CLIP (một lần): upload `pact-clip.zip` thành dataset `pact-clip`. Nếu đã có
   dataset `pact-kit` cũ thì bỏ qua bước này: nó đã chứa đúng trọng số đó.
2. Code: upload `pact-code.zip` thành dataset `pact-code` (~2 MB). Mỗi khi code đổi, tạo
   version mới của dataset này. Kaggle giải nén hay giữ nguyên zip đều được.
3. Kaggle -> Code -> New Notebook -> File -> Import Notebook -> chọn `pact_demo.ipynb`.
4. Trong notebook: Add Input -> `pact-code` + `pact-kit` (hoặc `pact-clip`) + các dataset ảnh.
   Accelerator: GPU (4x L4 hoặc 2x T4). Internet có thể để TẮT.
5. Chạy các ô từ trên xuống; các ô đầu kiểm tra mọi thứ trước khi train.
"""


def repo_files():
    """(absolute path, path inside the archive) for every file of the repository to ship."""
    for item in INCLUDE:
        path = os.path.join(ROOT, item)
        if os.path.isfile(path):
            yield path, f"PACT/{item}"
            continue
        for folder, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if not name.endswith(".pyc"):
                    full = os.path.join(folder, name)
                    yield full, "PACT/" + os.path.relpath(full, ROOT).replace(os.sep, "/")


def _add(zf, path, arcname):
    method = zipfile.ZIP_STORED if path.endswith(STORED) else zipfile.ZIP_DEFLATED
    zf.write(path, arcname, compress_type=method)


def build_info():
    """Identity of the code being shipped: id = <time>-<commit>[+dirty]-<hash of shipped files>.

    The hash covers the content of every shipped file, so two zips with the same id contain the
    same code even when the working tree has uncommitted changes (then the commit alone is not enough).
    """
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    digest = hashlib.sha256()
    for path, arc in sorted(repo_files(), key=lambda x: x[1]):
        if arc.endswith(".ipynb") or arc == "PACT/kaggle/pact_demo.ipynb":
            continue  # generated notebook (embeds timestamps), not code
        with open(path, "rb") as f:
            digest.update(arc.encode() + b"\x00" + f.read())
    built = time.strftime("%Y-%m-%dT%H:%M:%S")
    commit, dirty = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain"))
    code_hash = digest.hexdigest()[:8]
    tag = f"{built[:10].replace('-', '')}-{built[11:16].replace(':', '')}-{commit or 'nogit'}{'+dirty' if dirty else ''}-{code_hash}"
    return {"built": built, "id": tag, "commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty, "code_hash": code_hash}


def make_kit_zip(kit_dir, dest, parts=("code", "wheels", "clip")):
    """Archive with the repository as a folder (no nested zip) and the chosen parts.

    "code" also writes PACT/BUILD_INFO.json: the notebook uses its timestamp to pick the
    newest code when several datasets (e.g. an old pact-kit and a new pact-code) are attached.
    """
    tmp = dest + ".part"
    with zipfile.ZipFile(tmp, "w") as zf:
        if "code" in parts:
            for path, arc in repo_files():
                _add(zf, path, arc)
            info = build_info()
            zf.writestr("PACT/BUILD_INFO.json", json.dumps(info, indent=2))
            print(f"[build] {info['id']}")
            _add(zf, os.path.join(ROOT, "kaggle", "pact_demo.ipynb"), "pact_demo.ipynb")
            zf.writestr("UPLOAD.md", GUIDE)
        for sub in ("wheels", "clip"):
            folder = os.path.join(kit_dir, sub)
            if sub not in parts or not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                _add(zf, os.path.join(folder, name), f"{sub}/{name}")
    os.replace(tmp, dest)  # never leave a half-written zip behind
    with zipfile.ZipFile(dest) as zf:
        names = zf.namelist()
        bad = zf.testzip()
    if bad:
        raise RuntimeError(f"corrupted entry in {dest}: {bad}")
    return names


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args, "--disable-pip-version-check"],
                   check=True)


def download_wheels(dest):
    os.makedirs(dest, exist_ok=True)
    # both are pure-Python (py3-none-any): the same files install on Kaggle's Linux image.
    # Their other dependencies (torch, torchvision, timm, regex, huggingface_hub, wcwidth...)
    # are already in the Kaggle image; the notebook checks it.
    pip("download", OPEN_CLIP_SPEC, "ftfy", "--no-deps", "--only-binary=:all:", "-d", dest, "-q")
    print("[ok] wheels:", ", ".join(sorted(os.listdir(dest))))


def export_clip(dest, pylib):
    env = dict(os.environ)
    try:
        import open_clip  # noqa: F401
    except ImportError:
        print(f"open_clip not installed here: installing it into {pylib} (temporary)")
        pip("install", "--target", pylib, OPEN_CLIP_SPEC, "ftfy", "wcwidth", "--no-deps", "-q")
        env["PYTHONPATH"] = pylib + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "tools", "clip_offline.py"),
                    "export", "--out-dir", dest], check=True, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--skip-clip", action="store_true", help="reuse dist/pact-kit/clip/")
    ap.add_argument("--skip-wheels", action="store_true", help="reuse dist/pact-kit/wheels/")
    args = ap.parse_args()
    kit = os.path.join(args.dist, "pact-kit")
    os.makedirs(kit, exist_ok=True)

    if not args.skip_wheels:
        download_wheels(os.path.join(kit, "wheels"))
    if not args.skip_clip:
        export_clip(os.path.join(kit, "clip"), os.path.join(args.dist, ".pylib"))
    stale = os.path.join(kit, "pact-repo.zip")  # older layout: the code now goes in as PACT/
    if os.path.exists(stale):
        os.remove(stale)

    clip_files = ("clip/ViT-B-16-quickgelu-openai.pt", "clip/reference.pt")
    has_clip = all(os.path.exists(os.path.join(kit, p)) for p in clip_files)
    missing = [p for p in ("wheels",) + (() if args.skip_clip else clip_files)
               if not os.path.exists(os.path.join(kit, p))]
    if missing:
        sys.exit(f"missing in {kit}: {missing} (run without --skip-clip / --skip-wheels)")

    shutil.copy2(os.path.join(ROOT, "kaggle", "pact_demo.ipynb"), args.dist)
    code_zip = os.path.join(args.dist, "pact-code.zip")
    names = make_kit_zip(kit, code_zip, parts=("code", "wheels"))
    print(f"[ok] {code_zip}: {os.path.getsize(code_zip) / 2**20:.1f} MB, "
          f"{sum(n.startswith('PACT/') for n in names)} code files, "
          f"{sum(n.startswith('wheels/') for n in names)} wheels")
    clip_zip = os.path.join(args.dist, "pact-clip.zip")
    clip_src = os.path.join(kit, "clip", "ViT-B-16-quickgelu-openai.pt")
    if not has_clip:
        print("[skip] pact-clip.zip: no CLIP weights in dist/pact-kit/clip/ (use an existing "
              "pact-clip / pact-kit dataset on Kaggle)")
    else:
        if not os.path.exists(clip_zip) or os.path.getmtime(clip_zip) < os.path.getmtime(clip_src):
            make_kit_zip(kit, clip_zip, parts=("clip",))
        print(f"[ok] {clip_zip}: {os.path.getsize(clip_zip) / 2**20:.0f} MB (upload once; an "
              f"existing pact-kit dataset already holds the same weights)")
    print(f"[ok] {os.path.join(args.dist, 'pact_demo.ipynb')}")
    print("\nNext: upload pact-code.zip (new version each time the code changes), import "
          "pact_demo.ipynb (see UPLOAD.md inside the zip).")


if __name__ == "__main__":
    main()
