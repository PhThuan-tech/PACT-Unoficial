"""Zero-shot text classifier built once from the frozen CLIP text encoder."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_text_classifier(clip_model, tokenizer, classnames, templates, device,
                          template_mode="first", chunk=256):
    """Returns a (num_classes, dim) matrix of L2-normalised class embeddings.

    template_mode "first" uses only the first template (ZSCL / MoE-Adapters training);
    "ensemble" averages the embeddings of all templates (CLIP prompt ensembling, used by
    ZSCL / MoE-Adapters for evaluation).
    """
    if template_mode == "first":
        templates = templates[:1]
    texts = [t.format(c) for c in classnames for t in templates]
    embeddings = []
    for i in range(0, len(texts), chunk):
        tokens = tokenizer(texts[i:i + chunk]).to(device)
        embeddings.append(F.normalize(clip_model.encode_text(tokens).float(), dim=-1))
    embeddings = torch.cat(embeddings).view(len(classnames), len(templates), -1).mean(dim=1)
    return F.normalize(embeddings, dim=-1)
