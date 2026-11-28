"""Configuration: defaults, YAML loading and `key.sub=value` command-line overrides.

Every value marked "ASSUMPTION" is not specified in the PACT paper (CVPR 2026, main text)
and was chosen by this re-implementation. See README.md, section "Giả định".
"""

import copy

import yaml

DEFAULTS = {
    "setting": "mtil",  # "mtil" | "cil"
    "seed": 42,
    "device": "cuda",
    "output_dir": "outputs/run",
    "data_root": "data",
    "download": True,
    "num_workers": 4,

    "model": {
        "arch": "ViT-B-16-quickgelu",  # paper: CLIP ViT-B/16. OpenAI weights were trained with
                                       # QuickGELU; open_clip >= 2.2x no longer switches it on for
                                       # "ViT-B-16" + "openai" (only warns), so use -quickgelu.
        "pretrained": "openai",   # or a local state_dict file (offline, e.g. Kaggle)
    },

    "adapter": {
        "type": "mlp",            # "mlp" (PACT) | "lora" (ablation baselines) | "none" (zero-shot)
        "layers": [3, 6, 9, 12],  # paper: 1-indexed ViT layers 3, 6, 9, 12 (one every three layers)
        "per_task": True,         # False = one adapter shared by all tasks ("LoRA alone" ablation)
        "clone_mlp": True,        # SAI (3): copy the parallel MLP's weights
        "use_gate": True,         # SAI (4): convex mix (1-g) h_base + g h_adapter
        "gate_mode": "layerwise", # supp. "Analysis of convex mixing": main results use a FIXED
                                  # schedule increasing with depth, last-layer g = 0.5 (Avg 81.05);
                                  # "constant": g = gate_value at every layer (supp: best 0.1);
                                  # "learned": g = sigmoid(gamma_t) trained (main-paper wording)
        "gate_last": 0.5,         # layerwise: g at the last adapter layer (supp)
        "gate_power": 1.0,        # ASSUMPTION: g_l = gate_last * (l / l_last)^power (form not given):
                                  # 1.0 -> 0.125 / 0.25 / 0.375 / 0.5 at layers 3 / 6 / 9 / 12
        "gate_value": 0.1,        # constant: g at every layer
        "gate_init": 0.0,         # learned: initial gamma_1 (g = 0.5)
        "gate_init_mode": "preserve",  # learned: "preserve" -> gamma_t (t>=2) chosen so the
                                       # conditional branch is unchanged at insertion,
                                       # g_t = g_{t-1} t/(t-1); "fixed": every gamma_t = gate_init
        "gate_max_init": 0.95,    # learned + preserve: cap of g_t (avoids a saturated sigmoid)
        "lora_rank": 16,          # ASSUMPTION: only used when type == "lora"
        "noise_std": 0.1,         # paper Eq. (4)/(5): + 0.1 * eps, applied during training only (ASSUMPTION)
    },

    "head": {
        "type": "learned",        # supp: "only the current adapter and two lightweight heads are
                                  # updated" -> per task two trainable heads (cond, marg).
                                  # "text": frozen CLIP text classifier shared by both branches
        "init": "eval",           # ASSUMPTION: head = cosine classifier initialised from the text
                                  # embeddings ("eval": template ensemble, "train": first template)
        "lr_mult": 1.0,           # ASSUMPTION: head lr = train.lr * lr_mult
    },

    "pact": {
        "use_marg": True,         # L_marg (plasticity, marginal branch)
        "use_pa": True,           # L_pa (prior anchoring, stability)
        "use_phase": True,        # alpha_t = exp(-d_j*/tau); False -> alpha_t = 1
        "post_convergence": True, # False -> constraints active from iteration 0 (alpha_tilde = alpha)
        "tau": 0.1,               # ASSUMPTION
        "num_probes": 256,        # ASSUMPTION: M, size of the Gaussian probe bank
        "probe_seed": 0,
        "alpha_every": 10,        # recompute alpha_t every N iterations (ASSUMPTION)
        "first_task_reference": "pretrained",  # ASSUMPTION: task 1 has no history; compare to frozen MLP
        "pa_sigma_rel": 0.022,    # ASSUMPTION: relative drift tolerance of the Gaussian prior in L_pa.
                                  # For a relative parameter drift r, L_pa = r^2 / (2 sigma_rel^2).
                                  # Measured on CLIP ViT-B/16 (Kaggle quick runs): probe distance to the
                                  # MLP d_mlp ~ 80-125 x r^2 (NOT d ~ r^2 as on a random MLP), so
                                  # sigma_rel = sqrt(tau / (2 * 100)) ~ 0.022 makes L_pa ~ d / tau =
                                  # -ln(alpha) w.r.t. the MLP (observed L_pa 0.1-0.2 vs L_cond 0.3-1.5).
                                  # The former 0.22 made L_pa ~100x weaker (~0.002, no effect).
                                  # Re-tune together with tau using the logged d_mlp / d_min / L_pa.
        "schedule": {             # alpha_tilde = s * alpha_t, s in {0, 1} (supp. Sec. 6.3)
            "mode": "window",     # "window": hard switch of supp. Eq. 40-44 on the L_cond sequence;
                                  # "stability": EMA test (former assumption); "linear" | "none"
            "window": 100,        # ASSUMPTION: W (capped at 10% of the run)
            "w1": 0.5,            # ASSUMPTION: weight of sqrt(v) / (mean + eps)
            "w2": 0.5,            # ASSUMPTION: weight of |slope|  (w1 + w2 = 1)
            "gamma": 0.1,         # ASSUMPTION: threshold, switch on once phi <= gamma (logged: "phi")
            "latch": True,        # ASSUMPTION: once on, stays on
            "warmup_frac": 0.2,   # ASSUMPTION: never during the first 20% of iterations
            "latest_frac": 0.5,   # ASSUMPTION: safety net, switch on at 50% if phi never <= gamma
            "ramp_iters": 0,      # 0 = hard switch (supp. Eq. 44, Fig. 6 "loss initially jumps");
                                  # > 0: ramp s over this many iterations (capped at 20% of the run)
            "ema": 0.98,          # mode "stability" only
            "rel_tol": 0.05,      # mode "stability" only
            "linear_start": 0.5,  # mode "linear": ramp between these fractions of training
            "linear_end": 0.8,
        },
    },

    "train": {
        "iterations": 3000,       # paper: 3k (MTIL), 1k (CIL)
        "batch_size": 64,         # ASSUMPTION (ZSCL / MoE-Adapters default)
        "grad_accum": 1,
        "lr": 1.0e-5,             # ASSUMPTION
        "gate_lr_mult": 100.0,    # ASSUMPTION: gates are scalars, they get a larger learning rate
        "weight_decay": 0.0,      # ASSUMPTION
        "warmup": 100,            # cosine schedule with linear warmup (ZSCL style)
        "label_smoothing": 0.2,   # ZSCL / MoE-Adapters training scripts (--ls 0.2)
        "template_mode": "first", # text classifier used for training without learned heads (ZSCL)
        "amp": True,
        "log_every": 50,
    },

    "eval": {
        # MTIL test protocols (several are computed from the same run, the first one is the
        # main `acc_matrix`). Supp. Table 8: accuracy of every learned task stays bit-identical
        # afterwards and the proof mentions an undefined "routing algorithm" -> learned data
        # is evaluated with that task's own frozen snapshot (adapters 1..k, gate k, head k).
        #   router:          per test batch, a prototype router on frozen CLIP features picks a
        #                    learned task (its snapshot) or falls back to frozen CLIP (ASSUMPTION)
        #   oracle_zeroshot: learned -> own snapshot (task id known); unseen -> frozen CLIP
        #   oracle_latest:   learned -> own snapshot; unseen -> latest model
        #   latest:          everything through the latest model (former repo behaviour)
        #   oracle_hybrid:   learned -> own snapshot via the marginal branch (adapter k alone, marg head k);
        #                    unseen -> latest model (phase B: the cond mean dilutes adapter k, -1.6 Last)
        "protocols": ["router", "oracle_latest", "oracle_zeroshot", "latest", "oracle_hybrid"],
        "branch": "cond",         # ASSUMPTION: snapshot k uses the conditional branch (adapters 1..k)
        "batch_size": 128,
        "max_samples": None,      # e.g. 2000 for quick runs; None = full test sets
        "template_mode": "ensemble",  # ZSCL / MoE evaluation: zeroshot_classifier(classnames,
                                      # dataset.templates) = ensemble of all templates
        "amp": False,             # upstream evaluates in fp32 (TF32 matmuls allowed)
        "zero_shot_row": True,    # also evaluate the untouched CLIP before task 1
        "router": {
            "samples": 1000,      # training images per task for the prototype
            "quantile": 0.05,     # ASSUMPTION: accept task k if the batch mean cosine to its
                                  # prototype >= this quantile of its training cosines
        },
    },

    "mtil": {
        "order": ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
                  "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"],
        "shots": None,            # few-shot MTIL: 5 or 16 samples per class
        "fewshot_seed": 42,      # ZSCL / MoE-Adapters seed
    },

    "cil": {
        "dataset": "CIFAR100",    # "CIFAR100" | "TinyImageNet"
        "initial_classes": 10,    # classes in the first task
        "increment": 10,          # classes per later task
        "class_order": "auto",    # "auto" = configs/class_orders/<dataset>.yaml (DyTox order,
                                  # as in ZSCL / MoE-Adapters); a YAML path; a list; or "seed"
        "class_order_seed": 1993, # used when class_order == "seed"
        "template": "a bad photo of a {}.",  # MoE-Adapters CIL prompt
        "train_label_space": "current",  # ASSUMPTION: "current" task classes | "seen" classes
        "head": "text",           # CIL has no task id: "text" = frozen text classifier at test
                                  # (heads are not trained); "concat" = learned cond heads of
                                  # all tasks concatenated (needs train_label_space "current")
    },
}

ORDER_I = list(DEFAULTS["mtil"]["order"])
ORDER_II = ["StanfordCars", "Food", "MNIST", "OxfordPet", "Flowers", "SUN397",
            "Aircraft", "Caltech101", "DTD", "EuroSAT", "CIFAR100"]


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def set_by_path(cfg, dotted_key, value):
    node = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path=None, overrides=()):
    """Defaults <- YAML file <- `a.b=value` overrides (values parsed as YAML)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # YAML reads "1e-3" as a string
            try:
                value = float(value)
            except ValueError:
                pass
        set_by_path(cfg, key.strip(), value)
    if cfg["mtil"]["order"] == "I":
        cfg["mtil"]["order"] = list(ORDER_I)
    elif cfg["mtil"]["order"] == "II":
        cfg["mtil"]["order"] = list(ORDER_II)
    return cfg
