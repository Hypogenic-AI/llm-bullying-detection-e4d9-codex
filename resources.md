## Resources Catalog

### Summary
This document catalogs papers, datasets, and code gathered for the "LLM Bullying" research hypothesis (iterative rejection prompts for detector evasion).

### Papers
Total papers downloaded: **12**

| Title | Year | File | Key Info |
|---|---:|---|---|
| Can AI-Generated Text be Reliably Detected? | 2023 | papers/2023_can_ai_generated_text_be_reliably_detected.pdf | Foundational detector reliability benchmark |
| Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense | 2023 | papers/2023_paraphrasing_evades_detectors_of_ai_generated_text_but_retrieval_is_an.pdf | Direct evidence of paraphrase-based evasion; retrieval defense |
| Large Language Models can be Guided to Evade AI-Generated Text Detection | 2023/2024 | papers/2023_large_language_models_can_be_guided_to_evade_ai_generated_text_detecti.pdf | Prompt-based detector evasion (SICO) |
| Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack | 2024 | papers/2024_humanizing_machine_generated_content_evading_ai_text_detection_through.pdf | Dynamic adversarial attacks |
| On the Possibilities of AI-Generated Text Detection | 2023 | papers/2023_on_the_possibilities_of_ai_generated_text_detection.pdf | Detection limits/assumptions |
| RADAR: Robust AI-Text Detection via Adversarial Learning | 2023 | papers/2023_radar_robust_ai_text_detection_via_adversarial_learning.pdf | Robust detector training |
| Provable Robust Watermarking for AI-Generated Text | 2023 | papers/2023_provable_robust_watermarking_for_ai_generated_text.pdf | Watermark robustness |
| Are AI-Generated Text Detectors Robust to Adversarial Perturbations? | 2024 | papers/2024_are_ai_generated_text_detectors_robust_to_adversarial_perturbations.pdf | Adversarial robustness study |
| Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors | 2025 | papers/2025_your_language_model_can_secretly_write_like_humans_contrastive_paraphr.pdf | Contrastive paraphrase attack |
| BiMarker: Enhancing Text Watermark Detection for LLMs with Bipolar Watermarks | 2025 | papers/2025_bimarker_enhancing_text_watermark_detection_for_large_language_models_.pdf | Watermark-detection improvement |
| Hidding the Ghostwriters (EMNLP) | 2023 | papers/2023_hidding_the_ghostwriters_an_adversarial_evaluation_of_ai_generated_stu.pdf | Adversarial student-essay detection |
| Hidding the Ghostwriters (arXiv variant) | 2024 | papers/2024_hidding_the_ghostwriters_an_adversarial_evaluation_of_ai_generated_stu.pdf | Alternate release/version |

See `papers/README.md` for details.

### Datasets
Total datasets downloaded: **3**

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| HC3 raw | HuggingFace (`Hello-SimpleAI/HC3`) | 24,322 rows (~82 MB local subset/files) | Human vs AI QA text classification | datasets/hc3_raw/ | JSONL + samples |
| AI Text Detection Pile | HuggingFace (`artem9k/ai-text-detection-pile`) | 1,392,522 rows (~1.9 GB) | Large-scale detector training/eval | datasets/ai_text_detection_pile/ | 7 parquet shards |
| Ghostbuster Reuters | HuggingFace (`acmc/ghostbuster_reuter`) | 7,000 rows (~13 MB) | Attribution/detection robustness | datasets/ghostbuster_reuter/ | Includes generated/model fields |

See `datasets/README.md` for download instructions and loading examples.

### Code Repositories
Total repositories cloned: **3**

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| ai-detection-paraphrases | https://github.com/martiansideofthemoon/ai-detection-paraphrases | DIPPER attack + retrieval defense | code/ai-detection-paraphrases/ | Official repo for core paraphrase-evasion paper |
| detect-gpt | https://github.com/eric-mitchell/detect-gpt | Curvature-based detector baseline | code/detect-gpt/ | Widely used detector baseline |
| lm-watermarking | https://github.com/jwkirchenbauer/lm-watermarking | Watermark generation/detection | code/lm-watermarking/ | Official watermark baseline |

See `code/README.md` for key scripts and setup notes.

### Resource Gathering Notes

#### Search Strategy
1. Attempted `paper-finder` script first.
2. Backend unavailable; switched to OpenAlex API for relevance-ranked fallback search.
3. Downloaded open-access PDFs prioritizing detector-evasion and watermarking papers.
4. Deep-read 3 highest-relevance papers using PDF chunker over all chunks.
5. Pulled benchmark datasets from HuggingFace using snapshot-based download.
6. Cloned canonical code repos for attacks/detectors/watermarking.

#### Selection Criteria
- Direct relevance to iterative rewriting/rejection and detector evasion.
- Preference for papers with available code and empirical detector metrics.
- Inclusion of both attack and defense lines (paraphrase, robust detectors, watermarking).
- Datasets with practical utility for binary AI-vs-human detection and robustness testing.

#### Challenges Encountered
- Local paper-finder service unavailable/hanging.
- arXiv and Semantic Scholar API rate limits (HTTP 429) from this environment.
- One full-text paper URL blocked (JAIR HTTP 403).
- `datasets` script-based loading failed for HC3 due environment behavior.

#### Gaps and Workarounds
- Workaround for literature discovery: OpenAlex API fallback with metadata persistence in `papers/manual_search_results.json`.
- Workaround for HC3 ingestion: snapshot raw JSONL files directly from HuggingFace hub.
- Some methods (e.g., SICO official code link) not directly captured here; paper details still sufficient for baseline reconstruction.

### Recommendations for Experiment Design
1. **Primary dataset(s)**: `hc3_raw` for controlled human-vs-AI benchmarking; `ai_text_detection_pile` for scale.
2. **Baseline methods**: one-shot anti-AI prompt, iterative rejection protocol, paraphrase baseline (DIPPER-style), detector suite (DetectGPT/classifier/watermark).
3. **Evaluation metrics**: ROC-AUC + TPR@1%FPR as primary; semantic similarity/readability/perplexity deltas as guardrails.
4. **Code to adapt/reuse**: `ai-detection-paraphrases` for attack/defense pipeline, `detect-gpt` for detector baseline, `lm-watermarking` for provenance checks.

## Research Execution Update (2026-03-30)

### What Was Executed
- Implemented end-to-end experimental pipeline in `src/run_research.py`.
- Ran smoke validation and full experiment using real OpenAI API generations (`gpt-4.1-mini`).
- Compared one-shot vs iterative rejection rewriting over:
  - Fresh model generations from HC3 prompts (`n=15`)
  - External generated text from Ghostbuster Reuters (`n=15`)
- Evaluated with two independent local detectors trained from AI Text Detection Pile.

### Artifacts Produced
- `planning.md`
- `REPORT.md`
- `results/metrics.json`
- `results/scored_texts.csv`
- `results/stat_tests.csv`
- `results/summary_table.csv`
- `results/pass_rates.csv`
- `results/auc_table.csv`
- `results/trajectory_table.csv`
- `results/plots/*.png`

### High-Level Outcome
- Iterative rejection did not outperform one-shot in this run.
- Stylometric detector showed significant worsening for iterative on external text (FDR-adjusted `p=0.0440`).
