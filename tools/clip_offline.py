"""Export CLIP ViT-B/16 (OpenAI) weights to a file for offline use, and verify the file.

On a machine WITH internet (your PC):
    python tools/clip_offline.py export --out-dir dist/pact-kit/clip
        -> ViT-B-16-quickgelu-openai.pt  (plain state_dict, fp32, ~600 MB)
        -> reference.pt                  (features of fixed inputs, for verification)

On Kaggle (offline):
    python tools/clip_offline.py verify --weights .../ViT-B-16-quickgelu-openai.pt \
        --reference .../reference.pt
    then run PACT with
        --set model.arch=ViT-B-16-quickgelu model.pretrained=.../ViT-B-16-quickgelu-openai.pt

Why "-quickgelu": the OpenAI weights were trained with QuickGELU. open_clip >= 2.2x builds
"ViT-B-16" with plain GELU even for pretrained="openai" (it only warns), which silently
lowers accuracy. "verify" also prints how close the plain-GELU variant would be (only
meaningful with the real weights; with --random-init both variants coincide).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

ARCH = "ViT-B-16-quickgelu"
PROMPTS = ["a photo of a cat.", "a photo of a dog.", "a photo of an airplane."]
WEIGHTS_NAME = "ViT-B-16-quickgelu-openai.pt"


def _fixed_images():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 3, 224, 224, generator=g)


@torch.no_grad()
def _features(model, tokenizer, device):
    model = model.to(device).eval()
    img = F.normalize(model.encode_image(_fixed_images().to(device)).float(), dim=-1)
    txt = F.normalize(model.encode_text(tokenizer(PROMPTS).to(device)).float(), dim=-1)
    return img.cpu(), txt.cpu()


def export(out_dir, random_init=False):
    import open_clip

    os.makedirs(out_dir, exist_ok=True)
    pretrained = None if random_init else "openai"  # random_init: offline self-test only
    model = open_clip.create_model(ARCH, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(ARCH)
    weights = os.path.join(out_dir, WEIGHTS_NAME)
    torch.save({k: v.float() for k, v in model.state_dict().items()}, weights)
    img, txt = _features(model, tokenizer, "cpu")
    ref = os.path.join(out_dir, "reference.pt")
    torch.save({"arch": ARCH, "prompts": PROMPTS, "image": img, "text": txt,
                "open_clip": open_clip.__version__, "random_init": random_init}, ref)
    print(f"saved {weights} ({os.path.getsize(weights) / 2**20:.0f} MB) and {ref}")
    return verify(weights, ref, "cpu")


def verify(weights, reference, device):
    import open_clip

    ref = torch.load(reference, map_location="cpu")
    tokenizer = open_clip.get_tokenizer(ref["arch"])
    model = open_clip.create_model(ref["arch"], pretrained=weights)
    img, txt = _features(model, tokenizer, device)
    cos = min((img * ref["image"]).sum(-1).min().item(), (txt * ref["text"]).sum(-1).min().item())
    probs = (100 * img[:1] @ txt.t()).softmax(-1)
    # the same weights in the plain-GELU architecture, to show the mismatch is detectable
    gelu = open_clip.create_model(ref["arch"].replace("-quickgelu", ""), pretrained=weights)
    gimg, _ = _features(gelu, tokenizer, device)
    gcos = (gimg * ref["image"]).sum(-1).min().item()
    ok = cos > 0.999
    print(f"[{'PASS' if ok else 'FAIL'}] {ref['arch']} from file: min cosine to reference = "
          f"{cos:.6f} (need > 0.999); plain-GELU arch would give {gcos:.4f}")
    print(f"       open_clip here {open_clip.__version__}, at export {ref['open_clip']}; "
          f"device {device}; sample probs {[round(p, 3) for p in probs[0].tolist()]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--random-init", action="store_true", help="self-test without download")
    v = sub.add_parser("verify")
    v.add_argument("--weights", required=True)
    v.add_argument("--reference", required=True)
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ok = export(args.out_dir, args.random_init) if args.cmd == "export" else \
        verify(args.weights, args.reference, args.device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
