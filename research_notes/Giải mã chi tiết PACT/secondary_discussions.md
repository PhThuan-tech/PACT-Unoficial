# Secondary discussions, explanations and follow-up work on PACT (CVPR 2026)

Scope note: Searched on 2026-11-19 with English and Chinese web searches, GitHub (REST API, unauthenticated), CVF Open Access / CVPR virtual site, papernotes.org (EN + ZH), chatpaper.com, and full-text greps of two 2026 arXiv CL-for-VLM papers. Semantic Scholar's API was rate-limited (HTTP 429) and its website returned 403, so a formal "cited-by" list was **not** obtained. GitHub code search requires auth and was not run.

Overall result: **secondary coverage of PACT is thin.** The only third-party explanation with any substance is an auto-generated-style "paper note" on papernotes.org (EN + ZH). No blogs, Zhihu/CSDN/WeChat posts, Bilibili talks, 机器之心/新智元/量子位 coverage, Reddit/X threads, or reproduction reports were found. The official code repo is an empty placeholder. The most useful find for implementation questions was the **official CVF supplementary PDF** (outside the main paper, but first-party), which is summarised under Question 1 so the report writer has it. It is clearly labelled as an official source, not secondary.

## Q1. Has anyone described PACT's test-time evaluation, alpha_tilde schedule, probe bank size, tau, lr, batch size, or how the KL terms are computed?

### Takeaway
No secondary source gives these details. The papernotes.org note only restates the main paper (it says outright that M, tau, lr, batch size, the stability metric and MTIL routing are not given). The official supplementary PDF fills in part of the gap. It defines the "windowed stability criterion", which is a hard 0/1 gate on loss statistics that appears to produce alpha_tilde. It also covers the g_t convex-mixing schedule (layer-wise increasing, last layer g = 0.5), the insertion-interval ablation, Order-II results, the compute cost, and a per-task accuracy matrix. It still gives no probe size M, no tau, lr, batch size, window W, weights w1/w2 or threshold value, and no explicit test-time routing description.

### Cited Findings
**Third-party paper note (papernotes.org; it paraphrases the paper and adds no new data):**
- Describes a frozen CLIP ViT-B/16 with new adapters inserted sparsely at layers 3, 6, 9 and 12. It has a dual-branch ViT: the conditional branch averages historical adapter outputs and adds the current adapter, and the marginal branch uses only the current adapter. Both branches are trained with cross-entropy as a "trainable surrogate for the KL term", and both inject Gaussian noise 0.1·ε, ε~N(0,I) — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- Phase-like weight: a fixed Gaussian probe set Z={z_m}, m=1..M, z_m~N(0,I_d), shared across layers and steps, goes through the current and historical adapters. The note computes a scale-normalised Frobenius distance, picks the nearest historical adapter j*, and sets α_t = exp(−d_{j*}/τ). This is "modulated by training stability" into α̃_t ∈ [0,1]. Total loss L_total = L_cond + α̃_t(L_marg + L_pa). The marginal and PA losses only switch on after stabilisation — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/); [papernotes ZH](https://papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- KL terms are given only symbolically: P = KL(Q_t^a(θ_a^t | θ_a^[t−1]) ‖ Q_t^a(θ_a^t)), S = KL(Q_t^a(θ_a^t) ‖ Q_{t−1}^a(θ_a^t)), L_pa = KL(Q_t^a ‖ Q_mlp^a). The note does not specify the posterior family or the estimator — [papernotes ZH](https://papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- Training details reported: AdamW, 3k steps for MTIL and 1k steps for CIL. No lr or batch size — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- The note says explicitly that the concrete α̃ computation is in the appendix ("其具体计算流程论文放在附录、正文未完全展开"). It does not describe any task-ID or router mechanism at test time. The only gate it mentions is the SAI residual gate g_t = σ(γ_t) — [papernotes ZH](https://papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)

**Official supplementary PDF (first-party, outside the main paper):** [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf), linked from the [CVF HTML page](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_PACT_Phase-Like_Transition_Constraints_in_Adapter-Based_Continual_Learning_of_Vision-Language_CVPR_2026_paper.html)
- **Sec. 6.3 "Windowed stability criterion"** (this is the post-convergence detector):
  - Over the training-loss sequence {ℓ_i}, with window size W, it computes a windowed mean ℓ̄_t, a variance v_t (with 1/(W−1)), and a finite-difference slope m_t = (1/W)Σ(ℓ_k − ℓ_{k−1}).
  - These combine into the composite indicator φ_t = w1·√v_t/(ℓ̄_t+ε) + w2·|m_t|, with w1, w2 ≥ 0 and w1+w2 = 1.
  - The gate is a **hard** function, g_t = 𝟙[φ_t ≤ γ], where γ is a "stability threshold".
  - The extracted text gives no values for W, w1, w2, ε or γ. — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Training dynamics (Fig. 6):** the x-axis is "task_index × 3000 + local_iter", so there are ≈3000 iterations per MTIL task. Within each task the `cond` objective first converges (loss ≈1–2). "When PACT is activated, the loss initially jumps because the PAC-Bayesian regularizer and proximity constraint are added", and it then settles on a slightly higher plateau. PACT is described as "an effective post-convergence regularizer" — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **SAI (Corollary 2):**
  - The adapter sits in parallel with the frozen MLP, has the same architecture, and is cloned from the MLP parameters.
  - It is combined by convex mixing, h_out = (1−g_t)h_base + g_t·h_adapter with g_t = σ(γ_t) and h_adapter ∈ {h_cond, h_marg}.
  - Insertion is sparse, "e.g., one adapter every three layers". — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Convex-mixing schedule ablation (Fig. 4):**
  - A fixed g_t shared by all layers peaks at g = 0.1 with Avg ≈ 79.5%.
  - A "layer-wise increasing" g_t^ℓ (close to 0 in shallow layers, rising with depth, with the last-layer value as the hyperparameter) is better almost everywhere and peaks at last-layer g = 0.5 with Avg 81.05%. That configuration is used in the main experiments.
  - The functional form of the layer-wise schedule is not given in the extracted text. — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Insertion-interval ablation (Fig. 5), MTIL Avg:**

  | Interval k | Avg |
  |---|---|
  | 1 | ≈81.6 |
  | 2 | ≈81.2 |
  | 3 | 81.05 (adopted) |
  | 4 | 78.2 |
  | 6 | 75.1 |
  | 12 | 71.3 |

  — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Compute (Table 7):** PACT has 37.7M trainable parameters, 15368 MiB GPU memory and 1.36 s/it, against MoE-Adapters at 59.8M, 22358 MiB and 1.58 s/it. That is a reduction of −36.96% params, −31.26% memory and −13.92% time. The 36.96% figure is therefore measured **against MoE-Adapters specifically**. Only the current adapter and "two lightweight heads are updated", and the backbone and historical adapters are frozen — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **MTIL Order-II (Table 6):** Transfer 67.55, Avg 78.00, Last 89.50. That is +1.65, +2.30 and +4.00 over the strongest prior method (baselines include MoE-Adapters, DIKI and GIFT) — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Per-task accuracy matrix, MTIL Order-I (Table 8):**
  - After a task is trained, its column stays **exactly constant** in every later row. Examples: Aircraft 65.58, Caltech101 96.60, CIFAR100 91.21, DTD 80.53, EuroSAT 97.98. The authors read this as "negligible forgetting".
  - The final-row mean is 90.54, which matches the reported Last.
  - Some not-yet-trained columns stay constant across rows: CIFAR100 70.21, DTD 46.68, EuroSAT 54.90, Flowers 71.05, Food 89.52.
  - Others drift downward before they are trained: MNIST 68.20→63.60, OxfordPet 90.54→89.65, StanfordCars 69.50→65.30, SUN397 70.50→66.80.
  - — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Bounded-loss assumption:** the supplement says cross-entropy is "effectively bounded" in practice. It also notes that "we activate the PACT regularizers in a post-convergence regime" where losses lie in a narrow range, and that the analysis can be read as applying to a clipped loss. P_t^a is "instantiated as the corresponding MLP-induced prior" — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)
- **Searched, not found:** grepping the extracted supplement text for "probe", "learning rate", "batch", "router", "task-id/identity", "inference" and "AdamW" returned no hits for the probe bank, lr, batch size or test-time routing — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wang_PACT_Phase-Like_Transition_CVPR_2026_supplemental.pdf)

### Inferences
- **α̃_t appears to be stability-gated.** Probably α̃_t = α_t · 𝟙[φ_t ≤ γ], or something close, i.e. the proximity weight is switched on by the windowed loss-stability gate. The supplement calls the gate "hard", and the papernotes summary says α_t is "modulated by training stability" into [0,1]. This product form is our reconstruction; neither source states it.
- **Symbol clash:** γ is used both as the stability threshold (Eq. 44) and as the SAI gate logit γ_t (g_t = σ(γ_t)). The letter g_t is also used for both the hard stability gate and the convex-mixing coefficient. Re-implementers should keep these apart.
- **The SAI gate seems effectively fixed, not freely learned.** The supplement sweeps g_t as a hyperparameter (fixed value, or a layer-wise schedule with last layer = 0.5), even though it writes g_t = σ(γ_t) with "task-specific parameter γ_t". Whether γ_t is trained or set is ambiguous.
- **Table 8 suggests task-identity-based selection at test time for seen tasks (speculation).** Bit-identical accuracies on every previously seen task across all later stages are what you would expect if each seen task is evaluated with its own frozen adapter, selected by an oracle task ID or a near-perfect router. A shared, continually updated pathway would normally move these numbers at least slightly.
- **Unseen-task columns look mixed.** Some are constant and close to (slightly above) zero-shot CLIP; others drift before training. One possibility is that unseen-task inputs are sometimes routed to the frozen backbone (as with MoE-Adapters' auto-selector) and sometimes to an aggregate of historical adapters. This is conjecture and needs the paper's/authors' confirmation.
- **Trainable-parameter arithmetic (speculation):**
  - 37.7M ≈ 8 × the parameters of one ViT-B/16 MLP block (2·768·3072 + biases ≈ 4.72M).
  - A single adapter at only 4 layers would be ≈18.9M.
  - So the count may cover two cloned MLPs per insertion point (e.g. separate cond/marg copies), or insertion in both encoders, or include the "two lightweight heads".
  - Re-implementers should check this against the main paper.

### Gaps
- Probe bank size M, the probe dimension, τ, the exact "scale-normalised Frobenius distance", lr, batch size, W, w1/w2, ε, the γ threshold value, and the functional form of the layer-wise g_t schedule. None appear in any secondary source, and they were not found in the supplement text extraction. The PDF text was partly garbled, so a visual read of the supplement is advised.
- No secondary source explains the test-time adapter/branch/gate selection on MTIL or how Transfer is computed for PACT.
- How the KL terms are actually estimated, beyond the CE surrogates for the plasticity term, and how L_pa = KL(Q_t^a‖Q_mlp^a) is computed in practice (e.g. as KL between output distributions or as a parameter-space proxy). Not described anywhere secondary.

## Q2. Are there unofficial re-implementations or reproduction reports?

### Takeaway
None found. The official repository is an empty placeholder (README only). Its single open issue, from August 2026, asks when the code will be released and has no reply. No reproduction reports, forks or third-party implementations surfaced.

### Cited Findings
- The papernotes summary and search engines cite the code URL as `github.com/xwangrs/PACTCVPR2026`. That URL returns **HTTP 404** — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/); checked via WebFetch.
- The actual author repo is `xwangrs/PACT-CVPR2026` (with a hyphen):
  - Created and last pushed 2026-03-17.
  - Repo size 0, no license, 0 stars, 0 forks.
  - Its only file is a 15-byte README.md containing "# PACT-CVPR2026".
  - — [GitHub repo](https://github.com/xwangrs/PACT-CVPR2026); [GitHub API](https://api.github.com/repos/xwangrs/PACT-CVPR2026)
- The repo's single issue, "Question About the Code Release for PACT", was opened 2026-08-10 by user `jaebyeong98`. The user says they saw the PACT poster at CVPR 2026 and asks for a code-release timeline. The issue is open with 0 comments — [GitHub issues API](https://api.github.com/repos/xwangrs/PACT-CVPR2026/issues?state=all)
- The author account `xwangrs` has 4 public repos:
  - Approximation_FSCIL-ECCV2024 (26★)
  - PACT-CVPR2026 (empty)
  - Quantile-Head-for-VLA (pushed 2026-09-29)
  - robot-arm-research-atlas (described as a VLA/world-model research atlas, pushed 2026-10-13)
  - — [GitHub API user repos](https://api.github.com/users/xwangrs/repos)
- The CVPR abstract (mirrored on chatpaper) still says "Our code will be released publicly"; chatpaper links no code — [chatpaper](https://chatpaper.com/paper/294187)
- GitHub repository searches for "PACT continual learning", "PACT CLIP adapter", "PACTCVPR2026" and "PACT MTIL" returned no PACT re-implementation, only unrelated repos — [GitHub search API](https://api.github.com/search/repositories?q=PACT+CLIP+adapter)

### Inferences
- The author's recent public activity has moved to VLA/robotics. Combined with the unanswered issue, this suggests an official code release is not imminent (speculation).
- Without official code, any re-implementation, including this project ("PACT-Unoficial"), will need to choose its own values for M, τ, the stability-gate parameters, lr and batch size.

### Gaps
- GitHub *code* search (which would find files mentioning "phase-like transition", "Stable Adapter Initialization" or "Prior Anchoring") needs authentication and was not run.
- Papers with Code is defunct/redirected, and no PACT entry was found. A Hugging Face Papers page was not found either (PACT appears to have no arXiv version, so HF/alphaXiv pages are unlikely).

## Q3. Do later (2026) papers report PACT numbers, or criticise/clarify its method?

### Takeaway
No later paper citing, comparing against or critiquing PACT was found. Two 2026 CL-for-VLM arXiv papers that would be natural citers were checked by full-text grep, and neither mentions PACT. The only critique available is the "Limitations" section of the papernotes summary.

### Cited Findings
- "Incremental Learning of Vision-Language Models via Task Subspace Projection and Dynamic LoRA" (Fu et al., arXiv 2611.19213, Nov 2026) has no occurrence of "PACT" or "phase-like" in its extracted full text — [arXiv 2611.19213](https://arxiv.org/pdf/2611.19213)
- "DIMoE-Adapters: Dynamic Expert Evolution for Continual Learning in Vision-Language Models" (Qin et al., arXiv 2605.07494, May 2026) has no occurrence of "PACT" in its extracted full text — [arXiv 2605.07494](https://arxiv.org/pdf/2605.07494)
- "CMAP: Cross-Modal Adaptive Prompting for Multi-Domain Task-Incremental Learning" (arXiv 2605.25708, an MTIL method) does not mention PACT — [arXiv 2605.25708](https://arxiv.org/html/2605.25708)
- The papernotes "Limitations" section is a third-party critique, likely LLM-generated. It says:
  - The method relies on "post-convergence" detection, and the "sensitivity to stability thresholds/scheduling is hard to assess".
  - Using only the nearest historical adapter "might overlook complex relationships with multiple historical tasks as the task sequence grows".
  - Experiments cover only CLIP ViT-B/16 with adapters; generalisation to LoRA or prompts "remains to be verified".
  - The paper "lacks a fine-grained analysis of 'Task Similarity vs. Gain'".
  - — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- The papernotes "Highlights" section singles out the Gaussian-probe response distance as "a lightweight, parameter-free similarity metric", and calls the "constraints should be applied post-convergence" idea "highly reusable" — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- PACT's headline MTIL Order-I numbers, as reproduced by papernotes from the paper: Transfer 71.76, Avg 81.05, Last 90.54, against TRGE 69.8/78.5/87.6, AFA 70.3/78.5/87.2, ConDU 70.3/78.3/86.2 and MoE-Adapters 67.4/77.2/87.4. Few-shot (5-shot) PACT: 70.5/72.0/80.7 — [papernotes EN](https://en.papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- A search snippet extracted the 16-shot MTIL row from the official PDF as Transfer 72.7 / Avg 76.1 / Last 84.4. The tool flagged the PDF text as partly garbled, so verify this against the paper — [CVPR 2026 paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_PACT_Phase-Like_Transition_Constraints_in_Adapter-Based_Continual_Learning_of_Vision-Language_CVPR_2026_paper.pdf)

### Inferences
- PACT has no arXiv preprint and no code. Together with only ~5 months since CVPR, that likely explains why no follow-up work compares against it yet. MTIL papers usually copy baseline numbers from arXiv/code-backed methods.
- No source examines PACT's orthogonality critique, i.e. its claim that orthogonal-subspace methods create "knowledge islands". The claim is unexamined in the secondary literature.
- The zero-forgetting pattern in Table 8 is relevant here. If seen tasks are evaluated with task-specific frozen adapters, PACT's "Last" advantage rests partly on per-task isolation at inference, which is a form of isolation in itself. Nobody has raised this publicly; it is our own observation and should be treated as an open question.

### Gaps
- A formal citation list (Semantic Scholar / Google Scholar "cited by") could not be retrieved: the S2 API returned 429 and the S2 web page 403. Some 2026 citing papers may therefore have been missed.
- No OpenReview reviews exist publicly (CVPR reviews are not public), so no reviewer critique is available.

## Q4. Author talks/interviews and Chinese-language or social-media coverage

### Takeaway
None found. No Zhihu, CSDN, WeChat 公众号, Bilibili, 机器之心/新智元/量子位, Reddit or X discussion of this PACT turned up. The only Chinese-language treatment is the papernotes.org ZH note, which paraphrases the paper. The CVPR virtual page lists a poster session but no video or slides.

### Cited Findings
- CVPR 2026 virtual poster page:
  - Poster session Saturday, June 6, 2026, 11:45 AM–1:45 PM MDT.
  - A poster PNG image is linked.
  - No video, slides, project page or code link.
  - Authors listed as Xuan Wang, Guiguang Ding, Jungong Han.
  - — [CVPR virtual poster 37161](https://cvpr.thecvf.com/virtual/2026/poster/37161)
- Chinese-language note (papernotes.org, ZH): this is a structured paraphrase (motivation, method, results, limitations) with no author input or new numbers — [papernotes ZH](https://papernotes.org/CVPR2026/multimodal_vlm/pact_phase-like_transition_constraints_in_adapter-based_continual_learning_of_vi/)
- chatpaper.com pages (EN/DE/FR/ES) carry only the abstract. Their AI summary is behind a login — [chatpaper](https://chatpaper.com/paper/294187)
- A Chinese search ("PACT 相变 约束 适配器 持续学习 …", "… 知乎 OR CSDN OR 微信 论文解读") found only generic PAC-Bayes CSDN posts and an unrelated 雷锋网 CVPR-2026 overview, nothing on this paper — [CSDN generic PAC-Bayes](https://blog.csdn.net/ZONGDAOFU/article/details/139962065); [雷锋网 CVPR 2026 overview](https://m.leiphone.com/category/ai/irfoOx0RzJjhQLBx.html)
- Name collision: "PACT: Pruning and Clustering-Based Token Reduction for Faster Visual Language Models" (Dhouib et al., CVPR 2025) is an unrelated paper. Searches for "PACT VLM" mostly return it — [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/32558)
- The only public trace of a community reaction is the GitHub issue from an attendee who "saw the PACT poster at CVPR 2026" and wants the code — [GitHub issues API](https://api.github.com/repos/xwangrs/PACT-CVPR2026/issues?state=all)

### Inferences
- The same method summaries (dual-branch, SAI, PA, 36.96%) recur across every secondary page. They all derive from the abstract or the paper, so they should not count as independent confirmation.
- Abstract-paraphrasing sites (chatpaper, the "wizwand"/sota2 mirror, and search-engine AI summaries) add no information.

### Gaps
- WeChat 公众号 articles are poorly indexed by the search tools used. A direct search inside WeChat/Sogou-WeChat (搜狗微信) or Zhihu for "PACT 持续学习 CVPR 2026" / "相变约束 适配器" could still find posts.
- Bilibili was not directly searchable via the tools; no talk was found through web search.
- Tsinghua lab pages for Guiguang Ding (丁贵广) and Jungong Han (韩军功) were not checked for news items or talk slides.
