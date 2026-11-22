# Inherited Protocols and Code Behind PACT (CVPR 2026): ZSCL, MoE-Adapters4CL, DIKI/IAP, DyTox, NsCE [63], and Adapter-CL Baselines

Method note: I cloned the repos at these commits and read the code directly:
- MoE-Adapters4CL: commit `1fe7fb9`, default branch `MoE-Adapters`
- ZSCL: `b96e714` (main)
- DIKI: `8ef89e7` (main)
- NsCE [63]: `e5ba8c3` (main)

Links below point to `blob/<commit>/<path>#L..`. "Fact" bullets are what the code or paper says. "Inferences" are my reasoning about what PACT most likely does.

Base URLs:
- MOE = https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9
- ZSCL = https://github.com/Thunderbeee/ZSCL/blob/b96e714
- DIKI = https://github.com/lloongx/DIKI/blob/8ef89e7

---

## Q1. MTIL training hyperparameters in ZSCL and MoE-Adapters4CL (lr, batch, iterations, optimizer, wd, warmup, label smoothing, precision, templates)

### Takeaway
Both codebases share the same `src/args.py` defaults and the same training loop: AdamW with betas (0.9, 0.999), weight decay 0, batch size 64 (128 for eval), 100 linear warmup steps then cosine decay, label smoothing 0.2 set in the scripts, seed 42, and fp32 weights. MoE-Adapters trains 1000 iterations per task in its public script, with per-dataset lr between 1e-4 and 5e-3. The paper's 3k-iteration variant "Ours†" is the source of PACT's "3k iterations (MTIL)". Training uses one prompt template and only the current task's class names. Evaluation uses the dataset's full template ensemble and only the evaluated dataset's class names.

### Cited Findings
**Shared defaults (`args.py`)**
- Defaults in MoE-Adapters `mtil/src/args.py` L10-20: `--model ViT-B/16`, `--batch-size 64`, `--batch-size-eval 128`, `--lr 0.001`, `--wd 0.0`, `--ls 0.0`, `--warmup_length 100`, `--beta2 0.999`, `--seed 42` — [MOE mtil/src/args.py#L10-L20](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/args.py#L10-L20). ZSCL `mtil/src/args.py` has identical defaults — [ZSCL mtil/src/args.py#L10-L20](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/src/args.py#L10-L20).

**MoE-Adapters optimizer and schedule**
- Optimizer: `torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2))` — [MOE mtil/src/models/finetune.py#L122-L127](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/finetune.py#L122-L127).
- Schedule: `utils.cosine_lr(optimizer, args.lr, args.warmup_length, total_iterations)` — same lines.
- `cosine_lr` does linear warmup `base_lr*(step+1)/warmup_length`, then `0.5*(1+cos(pi*e/es))*base_lr` — [MOE mtil/src/utils.py#L13-L29](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/utils.py#L13-L29).
- Trainable parameters are only those whose names contain `adaptmlp`, `router` or `noise`. Everything else is frozen — [MOE finetune.py#L84-L113](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/finetune.py#L84-L113).

**MoE-Adapters full-shot script (Order I)**
- Order I dataset sequence: `Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397`.
- Per-task lr: `lr=(5e-3 1e-3 5e-3 1e-3 1e-4 1e-3 1e-3 1e-4 1e-3 1e-3 1e-3)`.
- Every task uses `--ls 0.2 --iterations 1000`, with `--experts_num 22` and `--frozen` (the frozen-expert strategy) from task 2 onward — [MOE mtil/scripts/train/train_full_shot_router11_experts22_1000iters.sh](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/train/train_full_shot_router11_experts22_1000iters.sh).

**MoE-Adapters paper**
- Table 1 caption: "'Ours†' and 'Ours' indicate our method trained on 3k and 1k iterations, respectively" — [MoE-Adapters arXiv 2403.11549](https://arxiv.org/html/2403.11549).
- Sec. 4.1 (as summarized by the fetch tool): batch size 64, AdamW, lr searched in {1e-3, 1e-4}, label smoothing searched in {0.1, 0.2}, 22 experts, top-k = 2 — [MoE-Adapters arXiv](https://arxiv.org/html/2403.11549).

**ZSCL**
- `scripts/ZSCL.sh`: `--train-mode=whole` (full fine-tuning), `--ls 0.2`, `--iterations 1000`, lr `5e-5` for Aircraft and `1e-5` for the rest (the lr array has `5e-5` at index 7 = MNIST), with WE (`--we --avg_freq 100`), `--l2 1`, ref-dataset ImageNet and ref-sentences conceptual_captions — [ZSCL mtil/scripts/ZSCL.sh](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/scripts/ZSCL.sh).
- ZSCL paper Sec. 5 / App. B: "we train 1K iterations for each task", AdamW, batch size 64 for MTIL, and "weight decay 0 and label smoothing 0.2 are good choices" — [ZSCL arXiv 2303.06628](https://arxiv.org/html/2303.06628).

**Precision**
- Both repos' `build_model` end with `for p in model.parameters(): p.data = p.data.float()`, so training and eval run in fp32. `convert_weights` (fp16) is defined but not used on this path.
  - [MOE mtil/clip/model.py#L776-L817](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/clip/model.py#L776-L817)
  - [ZSCL mtil/clip/model.py#L446](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/clip/model.py#L446)
  - [MOE cil/clip/model.py#L712](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/clip/model.py#L712)

**Training classifier**
- Texts are `[template(x) for x in dataset.classnames]`, where `template = dataset.template`, the first template of the dataset's list. Loss is `logit_scale.exp() * out @ embeddings.t()` with `F.cross_entropy(..., label_smoothing=args.ls)`.
- The text encoder runs on every step, and text-side adapters are trained too (`AdapterDoubleEncoder`).
- So the label space during training is the current dataset's classes only — [MOE finetune.py#L136-L182](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/finetune.py#L136-L182).
- `ClassificationDataset.template` returns `self.templates[0]`. Each dataset defines a list of templates; for example, Aircraft has `"a photo of a {c}, a type of aircraft."` and `"a photo of the {c}, a type of aircraft."` — [ZSCL mtil/src/datasets/collections.py#L60-L100](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/src/datasets/collections.py#L60-L100).

**Test classifier**
- `zeroshot_classifier(dataset.classnames, dataset.templates, model)` encodes every template per class, L2-normalizes each, averages, and re-normalizes.
- Logits are `100.0 * image_features @ zeroshot_weights`, over the evaluated dataset's own class names only.
  - [ZSCL mtil/src/models/evaluation.py#L19-L57](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/src/models/evaluation.py#L19-L57)
  - [MOE mtil/src/models/evaluation.py#L20-L41](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/evaluation.py#L20-L41)

**MoE-Adapters adapter module**
- Each "expert" is a bottleneck adapter: LayerNorm in, `down_proj` d→64, ReLU, dropout 0.1, `up_proj` 64→d.
- Init is "lora" style: down is kaiming, up is zero, biases zero.
- It is added in parallel: `x = x + mlp(ln_2(x)) + y`.
  - [MOE mtil/clip/adapter.py#L11-L69](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/clip/adapter.py#L11-L69)
  - [MOE mtil/clip/model.py#L463-L540](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/clip/model.py#L463-L540)

### Inferences
- PACT's "AdamW, 3k iterations (MTIL)" matches MoE-Adapters' "Ours†" (3k) on the same ZSCL code base. The most likely unstated defaults are: batch 64, warmup 100 then cosine, wd 0, label smoothing 0.2, fp32, seed 42, single-template training texts, and ensemble-template eval texts.
- MoE-Adapters tunes lr per dataset (1e-4 to 5e-3) for adapters. A full MLP-copy adapter like PACT's is far larger than a 64-dim bottleneck, so PACT likely needs a smaller lr. This is unverifiable.
- In MoE-Adapters the "classification head" is CLIP's text-embedding classifier, and the text encoder also carries adapters. PACT's "independent classification head" per branch could be a separate text-feature classifier or a linear head. The inherited code suggests a text-prototype head, but this is a guess.

### Gaps
- No public PACT code. Its lr, batch size and whether the text encoder is adapted are not in the inherited sources.
- The MoE-Adapters repo contains no 3k-iteration script (only `..._1000iters.sh`), so the lr schedule of the 3k variant is undocumented.
- The MoE-Adapters paper text cites the label-smoothing and lr search ranges, but the public script settles on ls = 0.2 and the per-dataset lr list above.

---

## Q2. How inference works on MTIL: task identity, MoE-Adapters' DDAS routing, and how the Transfer, Avg and Last metrics are computed

### Takeaway
No method in this lineage feeds an oracle task ID to the network. However, every image is classified only against the class names of the dataset it came from (a per-dataset label space). That is a strong implicit "task identity", because the candidate vocabulary is task-specific.

MoE-Adapters' DDAS routing works like this:
- It picks, per test batch, the task whose AlexNet-feature autoencoder has the lowest reconstruction MSE.
- Index 0, a TinyImageNet "reference" autoencoder, means "use frozen CLIP".
- If even the best MSE is above a threshold (0.065 full-shot, 0.06 few-shot), it also routes to frozen CLIP.

So for not-yet-learned datasets (the Transfer region), images are mostly sent to the unmodified CLIP, and Transfer approaches zero-shot CLIP.

Metrics come from an 11×11 accuracy matrix `arr[j, i]` = accuracy on dataset i after training task j:
- Transfer = mean over i = 1..10 of the mean over j < i of `arr[j, i]` (the upper-right triangle).
- Avg = the mean of the whole matrix.
- Last = the mean of the last row.

### Cited Findings
**DDAS routing code**
- Eval routing (`zeroshot_eval`), per batch:
  1. `input_to_ae = sigmoid(AlexNet_conv_features(images))`.
  2. Compute `encoder_criterion` (`nn.MSELoss()`, mean over the whole batch) for `Autoencoder_list[0]`, then for i = 1..11 keep the minimum.
  3. `if best_l > args.threshold: best_router = 0`.
  4. `task_id = best_router - 1`.
  5. `image_features = model.encode_image(images, task_id)`.
  6. `logits = 100.0 * image_features @ zeroshot_weights[task_id+1]`.

  Because the MSE is a batch mean, one routing decision is made per test batch (batch-size-eval 128), not per image — [MOE mtil/src/models/evaluation.py#L44-L85](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/evaluation.py#L44-L85); [MOE AutoEncoder.py#L66-L68](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/AutoEncoder.py#L66-L68).
- `zeroshot_classifier` builds text weights for every `task_id in range(-1, args.task_num)`, i.e. 12 versions. Index 0 is task -1, meaning no adapters, i.e. frozen CLIP. Each is built from the evaluated dataset's class names and templates only — [MOE evaluation.py#L20-L41](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/evaluation.py#L20-L41).
- In the transformer block, `if self.autorouter == True and val_task_id==-1: x = x + self.mlp(self.ln_2(x)); return x  # zero-shot`. Otherwise the task-specific router `router_list[val_task_id]` does noisy top-k gating over the 22 shared experts — [MOE mtil/clip/model.py#L463-L509](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/clip/model.py#L463-L509).
- Autoencoder: `Linear(256*13*13 → 100)`, ReLU, `Linear(100 → 256*13*13)`, Sigmoid, applied to frozen ImageNet AlexNet conv features. Each task's autoencoder is trained with Adam (lr 0.003 then `exp_lr_scheduler` init 0.01, wd 1e-4).
- Chooser training: 1000 iterations on TinyImageNet (index 0), then 300 iterations per MTIL dataset (`--train_chooser`).
  - [MOE AutoEncoder.py#L13-L46, L130-L200](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/AutoEncoder.py)
  - [train script](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/train/train_full_shot_router11_experts22_1000iters.sh)

**DDAS thresholds and eval loop**
- Thresholds: `threshold=655e-4` (full-shot) and `585e-4` in the few-shot training script, `600e-4` in the few-shot test script.
  - [full-shot test script](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/test/Full_Shot_order1.sh)
  - [few-shot test script](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/test/Few_Shot_order1_test.sh)
- Paper: "the reference data is TinyImageNet, and the threshold is 0.065 and 0.06 for full-shot and few-shot", and "Should all scores surpass a specific threshold, Thres, the system classifies the input as 'unseen data'" — [MoE-Adapters arXiv](https://arxiv.org/html/2403.11549).
- Eval loop: for each eval dataset j and each checkpoint i (after training task i), the script loads `${dataset[i]}.pth` and the chooser `${chooser[i]}_autochooser.pth`, with `--task_id 200 --autorouter --threshold`. This produces the full 11×11 matrix — [MOE Full_Shot_order1.sh](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/test/Full_Shot_order1.sh).
- At checkpoint i, the autoencoder list (12 slots) has only slots 0..i trained. The others are randomly initialized and would usually give high MSE — [MOE AutoEncoder.py#L130-L145](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/AutoEncoder.py).

**ZSCL metric code (`results.ipynb`, `statistics()`)**

```python
arr = np.delete(arr, 2, axis=1)  # delete cifar10 eval result
avg = arr.mean(axis=0)
last = np.array(arr[-1, :])
transfer = np.array([np.mean([arr[j, i] for j in range(i)]) for i in range(1, 11)])
```

Reported numbers are `np.around(x.mean(), 1)`. Order II reindexes rows and columns with `[9, 6, 7, 8, 5, 10, 0, 1, 3, 4, 2]` — [ZSCL mtil/results.ipynb](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/results.ipynb).

- I ran this formula on ZSCL's shipped `results/t3_7.npy` (ZSCL, Order I) and got Transfer 68.1, Avg 75.4, Last 83.6, which match the paper. (Transfer+Last)/2 would be 75.9, so Avg is not the mean of Transfer and Last — [ZSCL mtil/results/](https://github.com/Thunderbeee/ZSCL/tree/b96e714/mtil/results).
- ZSCL paper Sec. 4: "The average accuracy on all datasets among all timestamps is the 'Avg' metric… 'Last' is the average performance of all tasks after CL… 'Transfer' is the average task performance in the upper-right triangle of the matrix" — [ZSCL arXiv](https://arxiv.org/html/2303.06628).

**Conflicting metric wording**
- The MoE-Adapters paper describes "Average" as "a composite metric measuring the mean performance across 'Transfer' and 'Last'" — [MoE-Adapters arXiv](https://arxiv.org/html/2403.11549). This conflicts with ZSCL's code and numbers above, and MoE-Adapters ships no metric script of its own. Its numbers are presented in the same tables as ZSCL's.
- IAP gives prose-only definitions. Transfer is the average over unseen tasks i+1..T, and Avg is "the mean performance across all tasks T at each step" — [IAP arXiv 2503.20612](https://arxiv.org/html/2503.20612v2).
- ZSCL calls MTIL "a cross-domain version of task incremental learning". The paper never explicitly says whether task ID is given, but evaluation uses each dataset's own class names — [ZSCL arXiv](https://arxiv.org/html/2303.06628); [ZSCL evaluation.py](https://github.com/Thunderbeee/ZSCL/blob/b96e714/mtil/src/models/evaluation.py#L60-L75).

### Inferences
- "Task-ID-free" in MTIL papers means the network does not receive a task index. But the test label space is always the evaluated dataset's classes, inherited from ZSCL's `eval_single_dataset`. PACT, which "follows ZSCL/MoE-Adapters", almost certainly evaluates the same way.
- PACT's Transfer column is computed on datasets not yet trained. If PACT's gate g_t (or the conditional branch) cannot recognize "unseen" inputs, its adapters change features on those datasets and Transfer drops below zero-shot CLIP (Order I ZSCL zero-shot Transfer ≈ 69.4 per the notebook's zero-shot row). MoE-Adapters avoids this by explicitly routing unseen batches to frozen CLIP through DDAS.
- If PACT's Transfer is near or above about 69, PACT must have some mechanism equivalent to "fall back to frozen CLIP for out-of-distribution inputs" — for example, g_t being near 0 for OOD data. Otherwise that is a sign the gate is input-dependent.
- If PACT copies MoE-Adapters' DDAS-style evaluation, routing would be batch-level. Batch-level routing is easier than per-image routing, because each test batch comes from one dataset.
- To reproduce PACT's numbers, use ZSCL's `statistics()` (code above), not "(Transfer+Last)/2".

### Gaps
- The source of the zero-shot Transfer baseline value: I computed it only conceptually from the notebook's hardcoded zero-shot row `[24.3, 88.4, 0.0(CIFAR10 placeholder), 68.2, 44.6, 54.9, 71.0, 88.5, 59.4, 89.0, 64.7, 65.2]`. I did not print its output. Running `statistics(a)` on it would give the exact value.
- Whether PACT uses per-image or per-batch routing/gating at test time is unknown.

---

## Q3. Few-shot MTIL protocols: 5-shot (MoE-Adapters) and 16-shot (DIKI, then IAP)

### Takeaway
5-shot MoE-Adapters setup:
- 5 images per class, sampled as the first 5 per label encountered in a shuffled train loader under seed 42.
- 500 iterations per task, AdamW, batch 64, ls 0.2, per-dataset lr list below.
- DDAS chooser trained with 1000 iterations on TinyImageNet and 300 per task (on 5-shot data), threshold about 0.06.

16-shot setup:
- Originates in DIKI's `MTIL-FS.yaml`: `num_shots: 16`, seed 42, SGD lr 5.0, 10 epochs, batch 128, cosine, no warmup. These are prompt-tuning hyperparameters.
- IAP says it follows DIKI for 16-shot and reports SGD, lr 5.0, 10 epochs, batch 128 and FP16.

### Cited Findings
**5-shot (MoE-Adapters)**
- Few-shot script settings:
  - `--few_shot=5`
  - `--iterations 500` for MoE-Adapters training
  - `lr=(8e-3 5e-4 2e-4 5e-3 1e-3 1e-3 1e-2 1e-4 1e-2 1e-2 1e-2)`
  - `--ls 0.2`
  - chooser `--iterations 1000` (TinyImageNet), then 300 per task with `--few_shot=5`
  - `threshold=585e-4`

  [MOE train_few_shot_router11_experts22_1000iters.sh](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/train/train_few_shot_router11_experts22_1000iters.sh)
- The test script uses `threshold=600e-4` — [MOE Few_Shot_order1_test.sh](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/scripts/test/Few_Shot_order1_test.sh).
- Few-shot subset construction: the code iterates `dataset.train_loader` (shuffle=True) and keeps the first `args.few_shot` images per label. It builds a `TensorDataset` and a `DataLoader(batch_size=args.batch_size, shuffle=True)`. `main` calls `utils.seed_all(args.seed)` (default 42, cudnn deterministic).
  - [MOE finetune.py#L39-L69](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/models/finetune.py#L39-L69)
  - [MOE main.py#L40](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/main.py#L40)
  - [MOE utils.py#L128-L134](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/mtil/src/utils.py#L128-L134)
- MoE-Adapters paper: few-shot MTIL is 5-shot per task with 500 iterations per task, and the few-shot DDAS threshold is 0.06 — [MoE-Adapters arXiv](https://arxiv.org/html/2403.11549).

**AFA (also 5-shot)**
- AFA follows the 5-shot protocol: 5 samples per class, MTIL Order I. It uses 300 iterations for MTIL-FS, AdamW, lr 5e-4, batch 32 (full-shot: 1000 iterations, lr 1e-3, batch 16) — [AFA arXiv 2505.07690](https://arxiv.org/html/2505.07690v1).

**16-shot (DIKI)**
- DIKI `configs/MTIL-FS.yaml`: `num_shots: 16`, `seed: 42`, `use_validation: True`, prompt depth 8 (vision and text), n_ctx 8, `batchwise_prompt: True`, optim SGD, lr 5.0, `max_epoch: 10`, batch_size 128, wd 0, cosine, warmup 0 — [DIKI configs/MTIL-FS.yaml](https://github.com/lloongx/DIKI/blob/8ef89e7/configs/MTIL-FS.yaml).
- DIKI shot sampling: `base_set(cfg.dataset_root, num_shots=cfg.num_shots, seed=cfg.seed)` — [DIKI/datasets.py#L33-L46](https://github.com/lloongx/DIKI/blob/8ef89e7/DIKI/datasets.py#L33-L46).

**16-shot (IAP)**
- IAP IV-B3 calls it the "16-shot MTIL setting, where only 16 samples per task are available during model training" and says it follows DIKI.
- IAP IV-A4 implementation: SGD, 10 epochs, lr 5.0, batch 128 train / 256 inference, FP16, one RTX 4090. No seeds or number of runs are reported.
- IAP's 16-shot table compares ZSCL, L2P, DualPrompt, S-Prompts and DIKI. It does not include MoE-Adapters.

  [IAP arXiv 2503.20612](https://arxiv.org/html/2503.20612v2)

### Inferences
- PACT says it uses "5-shot (MoE-Adapters)" and "16-shot (IAP)". The 5-shot protocol is iteration-based (500 iterations per task in MoE-Adapters), and PACT's stated iteration counts (3k/1k) may or may not apply to few-shot.
- The 16-shot protocol is epoch-based (10 epochs) with prompt-tuning hyperparameters (SGD lr 5.0) that cannot transfer to full MLP-copy adapters. PACT most likely kept its own AdamW setup and borrowed only the data split and shot count, with seed 42 by default in both repos.
- The wording "16 samples per task" in IAP's text is very likely a slip for 16 shots per class (CoOp-style `num_shots`, as in DIKI's `base_set`).
- PACT's 16-shot comparison rows (DIKI, IAP and others) were likely copied from IAP's Table III, which has no MoE-Adapters row. Any MoE-Adapters 16-shot number in PACT would have to be the authors' own re-run.

### Gaps
- MoE-Adapters' few-shot results are reported for one seed. I found no multi-seed protocol.
- I did not inspect DIKI's `base_set` per-dataset implementation, i.e. whether the 16 shots come from the train split or train+val.
- IAP's public code (github.com/FerdinandZJU/IAP) was not inspected.

---

## Q4. The CIL protocol on CIFAR-100 and TinyImageNet: class orders, training label space, iterations, evaluation, and the "100 base classes" caption

### Takeaway
Both ZSCL and MoE-Adapters use the DyTox/DER class order.

CIFAR-100 is split into 10, 20 or 50 equal steps (10-10, 5-5, 2-2), with no base task. TinyImageNet uses 100 base classes followed by 5, 10 or 20 steps (increments of 20, 10 or 5).

Training:
- 1 epoch per task (a hardcoded `EPOCH = 1`, so iterations = number of batches).
- Only the current task's class names are used, and labels are shifted to start at 0.
- Prompt "a bad photo of a {}.".

Evaluation:
- Task-agnostic argmax over all classes seen so far.
- "Avg" is the mean over steps of the accuracy on all seen classes.
- "Last" is the final-step accuracy.

PACT's "CIFAR100 ... with 100 base classes" caption is almost certainly a typo, copied from the TinyImageNet table caption.

### Cited Findings
**Class order**
- MoE-Adapters and ZSCL `cil/class_orders/cifar100.yaml` begin `[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, ...]`.
  - [MOE cil/class_orders/cifar100.yaml](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/class_orders/cifar100.yaml)
  - [ZSCL cil/class_orders/cifar100.yaml](https://github.com/Thunderbeee/ZSCL/blob/b96e714/cil/class_orders/cifar100.yaml)
- This is identical to DyTox `options/data/cifar100_order1.yaml`. DyTox's README says it used "the class ordering proposed by DER (and which all baselines also follow)" and provides 3 CIFAR orders whose results are averaged in the DyTox paper.
  - [DyTox cifar100_order1.yaml](https://github.com/arthurdouillard/dytox/blob/main/options/data/cifar100_order1.yaml)
  - [DyTox README](https://github.com/arthurdouillard/dytox/blob/main/README.md)
- The TinyImageNet order begins `[131, 181, 22, 172, 144, 92, ...]` — [MOE cil/class_orders/tinyimagenet.yaml](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/class_orders/tinyimagenet.yaml).

**Splits**
- `get_class_ids_per_task` yields `class_order[:initial_increment]`, then chunks of `increment` — [MOE cil/continual_clip/utils.py#L27-L30](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/continual_clip/utils.py#L27-L30).
- Configs:
  - CIFAR-100 `cifar100_10-10` / `5-5` / `2-2`: `initial_increment = increment` = 10/5/2.
  - TinyImageNet `tinyimagenet_100-20` / `100-10` / `100-5`: `initial_increment: 100`, `increment` = 20/10/5.

  [MOE cil/configs/class/](https://github.com/JiazuoYu/MoE-Adapters4CL/tree/1fe7fb9/cil/configs/class)

**Hyperparameters (MoE-Adapters CIL)**
- `lr: 1e-3`, `ls: 0.0`, `weight_decay: 0.0`, `prompt_template: "a bad photo of a {}."`.
- Batch size 128 (CIFAR 10-10 and 5-5), 64 (CIFAR 2-2 and all TinyImageNet).
- [MOE cifar100_10-10-MoE-Adapters.yaml](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/configs/class/cifar100_10-10-MoE-Adapters.yaml); [tinyimagenet_100-10.yaml](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/configs/class/tinyimagenet_100-10.yaml)
- ZSCL's CIL configs use `lr 7.1e-6`, `ls 0.2`, WE — [ZSCL tinyimagenet_100-10-ZSCL.yaml](https://github.com/Thunderbeee/ZSCL/blob/b96e714/cil/configs/class/tinyimagenet_100-10-ZSCL.yaml).

**Training loop (MoE-Adapters CIL)**
- `EPOCH = 1; total_iterations = EPOCH * num_batches`, with AdamW, `cosine_lr(optimizer, cfg.lr, 30, total_iterations)` (warmup 30).
- Texts are only the current task's class names.
- Targets are shifted: `shift = 100 + (task_id-1)*increment` for TinyImageNet, `task_id*increment` for CIFAR.
- Cross-entropy is over current-task classes only.
- MoE in CIL uses a single router with `experts_num = 2`, `top_k = 2`, hardcoded.
  - [MOE cil/continual_clip/models.py#L50-L131](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/continual_clip/models.py#L50-L131)
  - [MOE cil/clip/model.py#L290-L292](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/clip/model.py#L290-L292)
- ZSCL CIL also uses `EPOCH = 1` and the same label shift — [ZSCL cil/continual_clip/models.py#L50-L64, L190-L203](https://github.com/Thunderbeee/ZSCL/blob/b96e714/cil/continual_clip/models.py).

**Evaluation (MoE-Adapters CIL)**
- `adaptation()` appends the new task's class names to `current_class_names`, and the text tokens cover all seen classes.
- Test is on `eval_dataset[:task_id + 1]`, with prediction `argmax` of softmax over all seen classes. No task ID is used: `forward` calls `self.model(image, self.text_tokens, 0, is_train=False)`.
- The log writes `'last': acc_list[-1]` and `'avg': statistics.mean(acc_list)`.
  - [MOE cil/main.py#L45-L79](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/main.py#L45-L79)
  - [models.py#L36-L46](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/continual_clip/models.py#L36-L46)
- Shipped results: CIFAR-100 10-10 gives `{"last": 77.52, "avg": 85.21}`, and TinyImageNet 100-10 gives `{"last": 76.35, "avg": 80.23}`.
  - [MOE experiments/.../cifar100_10-10.../metrics.json](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/experiments/class/cifar100_10-10-MoE-Adapters/metrics.json)
  - [tinyimagenet_100-10 metrics.json](https://github.com/JiazuoYu/MoE-Adapters4CL/blob/1fe7fb9/cil/experiments/class/tinyimagenet_100-10-MoE-Adapters/metrics.json)

**Paper captions**
- ZSCL paper: TinyImageNet "the first step learns 100 classes, and the rest is learned with 5, 10, and 20 steps". It follows DyTox's evaluation setting with batch size 128. The Table 7 caption is "...TinyImageNet splits in class-incremental settings with 100 base classes"; the CIFAR Table 6 caption has no "base classes" wording — [ZSCL arXiv](https://arxiv.org/html/2303.06628).
- MoE-Adapters paper: "The 100 classes of CIFAR100 are divided into {10,20,50} subsets, and the 100 classes from TinyImageNet are divided into {5,10,20} subsets".
  - Table 3 caption: "Comparison of different methods on CIFAR100 in class-incremental setting."
  - Table 4 caption: "...TinyImageNet dataset in class-incremental settings with 100 base classes."

  [MoE-Adapters arXiv](https://arxiv.org/html/2403.11549)

### Inferences
- "100 base classes" means the first task contains 100 of TinyImageNet's 200 classes. The remaining 100 classes come in 5, 10 or 20 steps, for 6, 11 or 21 tasks in total.
- For CIFAR-100, "100 base classes" would leave zero incremental classes. Both upstream papers attach the phrase only to the TinyImageNet table, so PACT's Table 4 caption "CIFAR100 ... with 100 base classes" is almost certainly a copy/paste typo. Two readings are possible:
  - Table 4 is actually TinyImageNet, mislabeled.
  - The CIFAR table caption accidentally inherited TinyImageNet's wording.
- PACT's "1k iterations (CIL)" departs from upstream's 1 epoch per task:
  - CIFAR-100 10-step: 5000 images per task at batch 128 is about 40 iterations.
  - TinyImageNet base task: 50,000 images at batch 64 is about 782 iterations; later tasks are about 79–157.
  - So PACT trains much longer per task than MoE-Adapters, unless "1k" is a total or a typo.
- Following the inherited code, PACT's CIL training label space is most likely current-task classes only (shifted labels). Its test label space is all seen classes, task-agnostic, with prompt "a bad photo of a {}.".

### Gaps
- Whether PACT averages 3 DyTox class orders (as DyTox does) or uses only order 1 (as ZSCL and MoE-Adapters ship) is unknown. ZSCL and MoE-Adapters ship only one order file.
- I did not open the TinyImageNet DyTox order file to verify it is identical to ZSCL's.

---

## Q5. The PAC-Bayes bound in NeurIPS 2024 [63] (Wang, Geng, Wan, Li, Chen), the conditional/marginal KL question, and public code

### Takeaway
[63] (arXiv 2409.19245, the NsCE paper) gives an online-CL PAC-Bayes bound:

ℛ ≤ ℛ̂ + Σ_t λK²/(min(v_s,v_m)Δ_t) + Σ_t KL(Q_t‖Q_{t−1})/λ + T·log(T/δ)/λ

The third term is the "task divergence" term, which the paper says covers myopia and forgetting.

I found no conditional/marginal KL decomposition in [63]: neither in the main text, the appendix I could read, nor the code. Its public code contains no KL computation; the bound is motivational only.

PACT's conditional-vs-marginal KL split is therefore PACT's own extension, and its practical implementation (adapter-mean + noise vs current adapter, two heads) cannot be copied from [63].

### Cited Findings
- [63] is "Forgetting, Ignorance or Myopia: Revisiting Key Challenges in Online Continual Learning", by Xinrui Wang, Chuanxing Geng, Wenhai Wan, Shao-yuan Li and Songcan Chen, NeurIPS 2024, arXiv 2409.19245 — [arXiv](https://arxiv.org/abs/2409.19245v2); [NeurIPS poster page](https://neurips.cc/virtual/2024/poster/93601).
- Theorem 4.1 (Sec. 4) holds with probability 1−δ for any λ > 0 and an online predictive sequence (Q_0,…,Q_T), where m_t = min(v_s, v_m)Δ_t, v_s and v_m relate to stream speed and model throughput, and the loss is bounded by K:
  - ℛ ≤ ℛ̂ + Σ_t λK²/(min(v_s,v_m)Δ_t) [ℳ, throughput term] + Σ_t KL(Q_t‖Q_{t−1})/λ [𝒟, task divergence] + T log(T/δ)/λ.
  - ℛ = Σ_t E_{h_t∼Q_t} E_{z_t∼μ_t} ℓ(h_t, z_t).

  The appendix proof (Thm B.5) has an extra factor of 2 in the denominator of the throughput term (λK²/(2·min(v_s,v_m)Δ_t)), so the main text and appendix are inconsistent — [arXiv 2409.19245v2 HTML](https://arxiv.org/html/2409.19245v2).
- The paper states "model's myopia and forgetting can be perceived as two complementary aspects of the proposed task divergence term". The appendix notes that the only usable prior in the predictive sequence is Q_0 — [arXiv HTML](https://arxiv.org/html/2409.19245v2).
- The bound "is specifically used to illustrate the generalization risk associated with each individual task" and "can not extend to data streams composed of disjoint tasks" — [arXiv HTML](https://arxiv.org/html/2409.19245v2).
- The NsCE method's actual losses are:
  - CE.
  - A sparsity regularizer: L_s = −Σ_c [(|w₁ᶜ|+…+|w_dᶜ|)/d] / ‖wᶜ‖₂.
  - Maximum-separation: L_p = (1/C_t²) Σ (⟨f(x_i),f(x_j)⟩ − p_ij)².
  - Total L = L_ce + γ(L_p + L_s) with γ = 0.01.
  - Targeted pairwise replay on confusion-matrix pairs above τ.

  None of these is a KL term — [arXiv HTML](https://arxiv.org/html/2409.19245v2).
- A scan of the remaining text (characters 100k–161k) found no decomposition of the KL or task-divergence term into conditional and marginal parts, and the string "KL" does not appear there — [arXiv HTML, offset 100000](https://arxiv.org/html/2409.19245v2).
- Code: https://github.com/wxr99/Forgetting-Ignorance-or-Myopia-Revisiting-Key-Challenges-in-Online-Continual-Learning (commit `e5ba8c3`). A grep over all `.py` files for `kl_div|KLD|kl(|KLDiv` returned no matches — [repo](https://github.com/wxr99/Forgetting-Ignorance-or-Myopia-Revisiting-Key-Challenges-in-Online-Continual-Learning).

### Inferences
- PACT's "conditional branch" (mean of all task adapters + 0.1·N(0,I)) and "marginal branch" (current adapter), each with its own head, look like a Monte-Carlo or surrogate realization of a split such as KL(Q_t‖Q_{t−1}) → (marginal-over-tasks prior) vs (task-conditional posterior). In practice this is most likely implemented as a KL or JS divergence between the two heads' softmax outputs, or as an L2 or feature distance. This is the standard way such terms are approximated, but nothing in [63]'s code dictates the form.
- Because [63]'s bound uses KL between consecutive posteriors over hypotheses (not data distributions), PACT's KL between "conditional" and "marginal" predictive distributions is a reinterpretation. Treat it as PACT's own design.

### Gaps
- I could not read the full appendix PDF to rule out a joint = marginal + conditional chain-rule remark in a later appendix. The fetched HTML covered the full text in two passes and found none.
- No public code shows how PACT computes its KL terms.

---

## Q6. How other adapter- and prompt-based CLIP CL methods in PACT's Table 1 handle inference (task ID, routing, ensembling)

### Takeaway
None of these methods uses an oracle task ID at test time. All use the evaluated dataset's class names (MTIL convention).

They differ in how they choose task-specific parameters:
- **MoE-Adapters:** batch-level autoencoder routing (DDAS) with a frozen-CLIP fallback.
- **DIKI:** per-task Gaussian on frozen-CLIP features; max log-likelihood picks the prompt (majority vote per batch), and sigmoid(logp/512 − 1) gives a soft residual weight.
- **IAP:** a DIKI-like two-stage confidence (task-level then class-level Gaussians) with thresholds 0.2/0.8.
- **AFA:** K-means prototypes per task, cosine similarity, threshold 0.75 to fall back to a generalization adapter.
- **TRGE:** task-ID recognition through prototype distance plus MLLM-generated task descriptions, then fusion with frozen CLIP.
- **ConDU:** a reconstructed task expert when the task is identified; otherwise a similarity-weighted fusion of top-K experts.
- **GIFT:** a single fully fine-tuned model with no routing (inferred).

### Cited Findings
**DIKI**
- Without `task_ids`, DIKI builds `MultivariateNormal(means[i], covars[i])` per learned task on frozen-CLIP image features. It takes `log_probs.topk(k=1)` and `batch_weight = sigmoid(topk/512 - 1.0)`.
- With `batchwise_prompt`, the majority prompt index in the batch is used.
- `evaluate()` calls `self.model(inputs)` (no task_ids) after `update_classnames(cur_task)`, i.e. the evaluated dataset's class names.
- Mean and covariance come from frozen features of the training data, with `+1e-7·I`.

  [DIKI/DIKI.py#L153-L200, L436-L470, L360-L375](https://github.com/lloongx/DIKI/blob/8ef89e7/DIKI/DIKI.py)

**IAP**
- "evaluated without accessing to the task identifier" (III-A1).
- IA-CDDP fits a Gaussian per task on frozen-CLIP features, and E_max is the sigmoid of the max log-likelihood. If E_max ≤ 0.2, the weight is 0 (original CLIP). If E_max ≥ 0.8, the weight is 1. In between, the weight is the mean over the top-5 class Gaussians. The text encoder uses the batch-mean score.
- The paper does not state explicitly whether the label space is per-dataset or the union.

  [IAP arXiv](https://arxiv.org/html/2503.20612v2)

**AFA**
- Task IDs are unknown (Sec. 3.1). The domain selector uses K = 5 K-means prototypes per task on frozen-CLIP features, with cosine similarity; if the max is below 0.75, the input goes to AFFA (unseen), else to that task's ABFA router.
- 22 LoRA-MoE experts, rank 16, top-2.

  [AFA arXiv 2505.07690](https://arxiv.org/html/2505.07690v1)

**TRGE (arXiv 2508.07738)**
- "Separation and Collaboration: Two-Level Routing Grouped Mixture-of-Experts for Multi-Domain Continual Learning".
- Inter-group routing relies on task identifiers recognized through task-prototype distance and MLLM-generated semantic task descriptions. Outputs of the frozen CLIP and the TRGE adapter are fused.

  [arXiv 2508.07738](https://arxiv.org/abs/2508.07738)

**ConDU (arXiv 2503.10705, ICLR 2026)**
- Maintains a unified model plus task triggers and prototypes.
- With a known task, it reconstructs that task's expert. Otherwise (unknown task ID or unseen task) it fuses the predictions of K decoupled experts, weighted by feature cosine similarity.
- This description comes from search-result summaries of the arXiv page and ICLR slides; I did not verify it in the full text.

  [arXiv 2503.10705](https://arxiv.org/abs/2503.10705); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007297)

**GIFT (CVPR 2025)**
- Diffusion-generated synthetic image-text pairs, contrastive distillation, an image-text alignment constraint, and adaptive weight consolidation (Fisher). It is a fine-tuning method evaluated on MTIL Order I/II and CIL (CIFAR100 and TinyImageNet).

  [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Synthetic_Data_is_an_Elegant_GIFT_for_Continual_Vision-Language_Models_CVPR_2025_paper.html); [arXiv 2503.04229](https://www.arxiv.org/abs/2503.04229)

### Inferences
- Every PEFT competitor in PACT's Table 1 includes an explicit "is this input from a learned task?" detector with a frozen-CLIP fallback: MoE-Adapters, DIKI, IAP, AFA and TRGE. This is what protects Transfer.
- PACT's description (a sigmoid gate g_t mixing base and adapter; conditional branch = mean over all adapters) has no explicit task-ID predictor. PACT therefore probably works in one of two ways:
  - (a) g_t is computed from the input, acting as a soft domain detector that drives g→0 on unseen data.
  - (b) PACT evaluates with per-dataset class names, using the conditional (all-adapter-mean) branch, possibly ensembled with zero-shot CLIP.
- Under the inherited protocol, test batches are single-dataset. A batch-level decision (as in MoE-Adapters and DIKI) is therefore "allowed" by convention and would be an easy, likely unreported, implementation choice.
- GIFT, being full fine-tuning, presumably has no routing. I could not confirm this from its text.

### Gaps
- I did not read GIFT's inference section or ConDU's full paper and code. The ConDU details come from summaries.
- I did not obtain how "one LoRA per task" (PACT Table 5 baseline) is evaluated. No inherited repo contains such a baseline: MoE-Adapters' closest analogue is a single adapter (the `else: # one adapter` branch in `model.py`) or 22 experts with per-task routers. PACT's "one LoRA per task" probably needs either a task-ID oracle or the same gate/routing as PACT, and this is not documented in the inherited sources.
- I did not inspect IAP's code to resolve whether the test label space is per-dataset.
