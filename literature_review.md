## Literature Review

### Research Area Overview
The current literature on AI-generated text detection shows a repeated pattern: style-level detectors are vulnerable to rewriting/paraphrasing attacks, while provenance-style methods (watermarking, retrieval against generation logs) are more robust but require infrastructure support. This directly aligns with the hypothesis that iterative rejection prompts ("too AI-sounding") can move model outputs into a lower-detectability regime more effectively than single-shot style edits.

### Search Keywords Used
- AI-generated text detection
- adversarial attacks against AI text detectors
- paraphrasing evade detector
- watermarking large language models
- humanizing machine-generated text
- prompt-based evasion of LLM detectors
- contrastive paraphrase attacks
- robust AI-text detection adversarial learning

### Key Papers

#### Paper 1: Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense
- **Authors**: Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer
- **Year**: 2023
- **Source**: NeurIPS 2023 / arXiv:2303.13408
- **Key Contribution**: Shows paraphrasing can dramatically break popular detectors; proposes retrieval-based defense.
- **Methodology**: Uses 11B DIPPER paraphraser with controllable lexical/order diversity.
- **Datasets Used**: Long-form QA and open-generation settings; retrieval corpora up to 15M generations.
- **Baselines**: DetectGPT, GPTZero, OpenAI classifier, watermark detection, RankGen.
- **Evaluation Metrics**: ROC/AUC, TPR at fixed FPR (notably 1% FPR), semantic similarity preservation.
- **Results**: Example from paper: DetectGPT detection can drop from 70.3% to 4.6% at 1% FPR after paraphrasing.
- **Code Available**: Yes (`code/ai-detection-paraphrases`).
- **Relevance to Our Research**: Strong direct support that rewriting loops can exploit detector weaknesses.

#### Paper 2: Large Language Models can be Guided to Evade AI-Generated Text Detection
- **Authors**: Ning Lu et al.
- **Year**: 2023 (published TMLR 2024)
- **Source**: TMLR
- **Key Contribution**: Introduces SICO, prompt-construction method for detector evasion without external paraphraser.
- **Methodology**: In-context example optimization and feature-guided prompts; proxy-detector feedback loop.
- **Datasets Used**: HC3-style and task-oriented setups (writing, QA, reviews; includes SQuAD/Eli5/Yelp-derived tasks).
- **Baselines**: Parrot, DIPPER, GPT-Para, Human Prompt.
- **Evaluation Metrics**: Detector AUC across GPT3-D/GPT2-D/DetectGPT/OpenAI/GPTZero/Log-Rank.
- **Results**: Reported large AUC reductions (e.g., on writing task from high baseline AUC to ~0.2-0.3 under SICO variants).
- **Code Available**: Paper code not directly gathered; method is reproducible from paper details.
- **Relevance to Our Research**: Closest conceptual precedent for "iterative feedback context shifts generation regime".

#### Paper 3: Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack
- **Authors**: Ying Zhou, Ben He, Le Sun
- **Year**: 2024
- **Source**: arXiv:2404.01907
- **Key Contribution**: Adversarial framework (white-box and black-box) for minor perturbation attacks on detectors.
- **Methodology**: Dynamic attacker-detector optimization; constrained perturbation modules.
- **Datasets Used**: CheckGPT, HC3.
- **Baselines**: Word-level, sentence-level, full-text rewriting, DIPPER-based attacks.
- **Evaluation Metrics**: AUC and text-quality constraints (e.g., readability/perplexity change).
- **Results**: Detectors can be compromised quickly in dynamic attack settings; robustness remains limited.
- **Code Available**: Not directly acquired in this pass.
- **Relevance to Our Research**: Supports iterative attack dynamics and detector adaptation failure modes.

#### Paper 4: Can AI-Generated Text be Reliably Detected?
- **Authors**: (see PDF metadata)
- **Year**: 2023
- **Source**: arXiv:2303.11156
- **Key Contribution**: Benchmarks reliability limitations for AI-text detection.
- **Methodology**: Comparative detector evaluation across data and model settings.
- **Datasets Used**: Mixed detection benchmarks.
- **Baselines**: Multiple detector families.
- **Evaluation Metrics**: ROC/AUC, transfer behavior.
- **Relevance to Our Research**: Provides foundational assumptions and detector fragility context.

#### Paper 5: On the Possibilities of AI-Generated Text Detection
- **Authors**: (see PDF metadata)
- **Year**: 2023
- **Source**: arXiv:2304.04736
- **Key Contribution**: Discusses theoretical/empirical limits of detector generalization.
- **Relevance to Our Research**: Motivates strict train/test separation by generator and prompt strategy.

#### Paper 6: RADAR: Robust AI-Text Detection via Adversarial Learning
- **Authors**: (see PDF metadata)
- **Year**: 2023
- **Source**: arXiv:2307.03838
- **Key Contribution**: Adversarially trained detector to improve perturbation robustness.
- **Relevance to Our Research**: Candidate robust baseline detector in our experiments.

#### Paper 7: Provable Robust Watermarking for AI-Generated Text
- **Authors**: (see PDF metadata)
- **Year**: 2023
- **Source**: arXiv:2306.17439
- **Key Contribution**: Provable properties for watermark-based provenance methods.
- **Relevance to Our Research**: Non-style baseline less sensitive to textual "humanization" edits.

#### Paper 8: Are AI-Generated Text Detectors Robust to Adversarial Perturbations?
- **Authors**: (see PDF metadata)
- **Year**: 2024
- **Source**: arXiv:2406.01179
- **Key Contribution**: Focused robustness stress-test under adversarial perturbations.
- **Relevance to Our Research**: Reinforces evaluation under iterative attack loops rather than static prompts.

### Additional Relevant Papers Downloaded
- Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors (2025)
- BiMarker: Enhancing Text Watermark Detection for Large Language Models with Bipolar Watermarks (2025)
- Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection (2023/2024 variants)

### Common Methodologies
- Prompt-only evasion: feature-guided in-context prompting (SICO-like).
- Paraphrase/rewrite evasion: neural paraphrasers (DIPPER-like), back-translation, seq2seq rewriting.
- Detector robustness: adversarial training, contrastive training, domain transfer tests.
- Provenance methods: statistical watermark detection and retrieval over generation logs.

### Standard Baselines
- **Detector baselines**: DetectGPT, GPT2/Roberta classifiers, GPTZero-like systems, OpenAI detector (historical), log-rank/perplexity detectors.
- **Attack baselines**: single-shot paraphrase, word substitution, sentence replacement, back translation, prompted rewriting.
- **Defense baselines**: watermark detection, retrieval matching (BM25/semantic retrieval), adversarially trained classifiers.

### Evaluation Metrics
- **Primary**: ROC-AUC and TPR at low FPR (especially 1% FPR).
- **Secondary**: Accuracy/F1 (less informative under class imbalance and threshold sensitivity).
- **Attack quality controls**: semantic similarity, readability scores, perplexity shift.
- **Transfer stress tests**: train detector on one generator family, test on others.

### Datasets in the Literature
- **HC3**: Human vs ChatGPT QA corpus; used in several detector studies.
- **CheckGPT**: Human/AI content for detector benchmarking.
- **Task-conditioned corpora**: SQuAD/Eli5/Yelp-derived generation tasks for writing/QA/reviews.
- **Large synthetic mixes**: Pile-derived human/AI corpora for classifier scaling.

### Gaps and Opportunities
- Gap 1: Few papers directly isolate *iterative rejection language* as the treatment variable.
- Gap 2: Limited causal evidence on whether evasion comes from feature discovery vs decoding-regime shifts.
- Gap 3: Many evaluations lack strict semantic-equivalence constraints for attack outputs.
- Gap 4: Detector benchmarking often under-reports robustness under repeated interaction loops.

### Recommendations for Our Experiment
- **Recommended datasets**:
  - `datasets/hc3_raw` for human-vs-AI QA comparisons.
  - `datasets/ai_text_detection_pile` for scalable detector training/evaluation.
  - `datasets/ghostbuster_reuter` for domain/attribution stress tests.
- **Recommended baselines**:
  - Single-shot style prompt baseline: "rewrite to sound less AI-generated".
  - Iterative rejection loops: N-turn rejection protocol with fixed budget.
  - External paraphrase baseline (DIPPER-style or equivalent) where feasible.
  - Detector baselines: DetectGPT-style, fine-tuned classifier, watermark/retrieval methods.
- **Recommended metrics**:
  - Primary: AUC and TPR@1%FPR.
  - Secondary: semantic similarity and quality deltas.
- **Methodological considerations**:
  - Keep identical source texts across one-shot vs iterative conditions.
  - Control for token budget, number of rewrites, and model temperature.
  - Evaluate cross-detector transfer (avoid overfitting to one detector).
  - Include semantic-preservation threshold before counting successful evasion.
