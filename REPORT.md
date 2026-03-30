# REPORT

## 1. Executive Summary
This study tested whether iterative rejection feedback (repeatedly saying a draft is "too AI-sounding") reduces detector scores more than a one-shot request to "sound less like AI." We used real API calls to `gpt-4.1-mini`, two local detectors, and controlled paired comparisons on the same source texts.

Key finding: in this run, iterative rejection did **not** outperform one-shot rewriting. Across both detectors and both source settings, iterative-final scores were slightly or moderately **higher** (more AI-like) than one-shot, and pass rates were unchanged or worse.

Practical implication: conversational rejection loops are not automatically a stronger evasion strategy than a single rewrite instruction; effect direction depends on detector family and setup, so claims of consistent iterative advantage should be treated as conditional, not universal.

## 2. Goal
### Hypothesis tested
Iterative rejection of LLM outputs as "too AI-sounding" leads to text that passes AI detectors better than a single "sound less like AI" instruction.

### Why it matters
AI-text detectors are used for moderation and academic integrity. If simple interaction patterns systematically evade them, deployment risk is higher than many users assume.

### Problem solved
This experiment isolates interaction protocol (one-shot vs iterative rejection) while holding source text and model constant, including a condition with externally sourced AI text not generated in-session.

### Expected impact
Provides evidence on whether iterative rejection is a generally stronger detector-evasion tactic or a detector/model-specific artifact.

## 3. Data Construction
### Dataset Description
- `datasets/hc3_raw/all.jsonl` (HC3): human vs ChatGPT QA corpus, used to sample questions and matched human references.
- `datasets/ghostbuster_reuter/data/train-00000-of-00001.parquet`: externally generated AI news-style texts, used for out-of-context rewrite condition.
- `datasets/ai_text_detection_pile/data/*.parquet`: large human/AI corpus used to train two independent local detectors.

Run-time sampled sizes:
- Fresh-generation condition: 15 prompts from HC3.
- External-text condition: 15 generated texts from Ghostbuster Reuters.
- Detector training set: 6,000 texts (balanced), split 80/20 into train/test.

### Example Samples
| Source | Label/Role | Example (truncated) |
|---|---|---|
| HC3 question | Prompt | "On a donation site ... what keeps people from taking the donation money and running?" |
| Fresh baseline output | AI-generated | "Imagine you have a lemonade stand..." |
| External source text | AI-generated (dataset field `generated=True`) | "Dominion Resources holds initial talks with East Midlands Electricity..." |

### Data Quality
From `results/metrics.json`:
- HC3 rows loaded: 23,492
- HC3 missing question: 0 (0.00%)
- HC3 missing human text: 0 (0.00%)
- HC3 duplicate questions: 0
- External rows after filtering: 5,998
- External duplicate text: 0
- Outliers in detector scores (|z|>3): 0 for both detectors

### Preprocessing Steps
1. Normalize whitespace and strip text.
2. Filter external texts to length >300 chars before sampling.
3. Sample with fixed seed (42).
4. Generate fresh AI answers from prompts using OpenAI API.
5. Produce one-shot rewrite and 4-round iterative rejection rewrites.
6. Score all texts with two detectors.
7. Compute paired statistics and bootstrap CIs.

### Train/Val/Test Splits
- Detector datasets: stratified 80/20 train/test split on human vs AI labels.
- Experimental comparisons: paired within-item analysis (`iterative_final` vs `one_shot`) on the same item IDs.

## 4. Experiment Description
### Methodology
#### High-Level Approach
A controlled paired design with two source settings:
1. Fresh model generations from HC3 prompts.
2. External AI text not generated in the current session.

Each item receives:
- `one_shot`: one rewrite request to sound less AI-like.
- `iterative`: 4 rounds with repeated rejection context: "Rejected: still too AI-sounding. Rewrite again..."

#### Why this method
This directly targets the user claim about rejection context and tests transfer to externally sourced text.

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- openai 2.30.0
- pandas 3.0.1
- numpy 2.4.4
- scikit-learn 1.8.0
- scipy 1.17.1
- matplotlib 3.10.8
- seaborn 0.13.2

#### Algorithms/Models
- Generator: OpenAI `gpt-4.1-mini`
- Detector A (`word_ai_prob`): TF-IDF word 1-2gram + Logistic Regression
- Detector B (`stylometric_ai_prob`): TF-IDF char_wb 3-5gram + Logistic Regression

Detector holdout AUCs:
- Word detector: 0.9239
- Stylometric detector: 0.9538

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| seed | 42 | fixed for reproducibility |
| n_fresh_items | 15 | runtime-constrained planned sample |
| n_external_items | 15 | runtime-constrained planned sample |
| iterative_rounds | 4 | planned ablation depth |
| temperature | 0.7 | fixed generation setting |
| max_tokens_answer | 220 | fixed cap |
| detector_train_size | 6000 | balanced subsample |

#### Training / Analysis Pipeline
1. Sample prompts and external texts.
2. Generate baseline fresh outputs.
3. Create one-shot and iterative rewrites.
4. Fit two detectors on AI Text Detection Pile subsamples.
5. Score all experimental texts.
6. Compute paired t-test, Wilcoxon, Cohen's d, Cliff's delta, bootstrap 95% CI.
7. Apply Benjamini-Hochberg correction.

### Experimental Protocol
#### Reproducibility Information
- Number of runs: 1 full run (plus smoke validation run)
- Seed: 42
- Hardware detected: 2x NVIDIA RTX 3090 (24GB each)
- Torch CUDA usable in this env: No (driver/runtime mismatch), so detector computations ran on CPU
- Execution time: 875.56s (~14.6 min)

#### Evaluation Metrics
- Detector AI probability (continuous)
- Pass rate at threshold 0.5
- AUC vs human references (fresh condition)
- Paired significance tests and effect sizes

### Raw Results
#### Main condition summary
| Source | Condition | Word detector mean | Stylometric detector mean |
|---|---|---:|---:|
| Fresh | baseline | 0.6288 | 0.5902 |
| Fresh | one_shot | 0.5835 | 0.5296 |
| Fresh | iterative_final | 0.5883 | 0.5644 |
| External | original_external | 0.4926 | 0.3930 |
| External | one_shot | 0.5363 | 0.4425 |
| External | iterative_final | 0.5389 | 0.5085 |

#### Paired hypothesis tests (iterative_final - one_shot)
| Detector | Source | Mean diff | p (t-test) | p (FDR) | Cohen's d | 95% CI |
|---|---|---:|---:|---:|---:|---|
| word_ai_prob | fresh | +0.0048 | 0.7009 | 0.8645 | +0.101 | [-0.0181, 0.0281] |
| word_ai_prob | external | +0.0025 | 0.8645 | 0.8645 | +0.045 | [-0.0241, 0.0307] |
| stylometric_ai_prob | fresh | +0.0348 | 0.0414 | 0.0828 | +0.580 | [0.0063, 0.0645] |
| stylometric_ai_prob | external | +0.0660 | 0.0110 | 0.0440 | +0.756 | [0.0241, 0.1088] |

#### Output Locations
- Results JSON/meta: `results/metrics.json`
- Scored rows: `results/scored_texts.csv`
- Stats tables: `results/stat_tests.csv`, `results/summary_table.csv`, `results/pass_rates.csv`, `results/auc_table.csv`, `results/trajectory_table.csv`
- Plots: `results/plots/condition_comparison.png`, `results/plots/iterative_trajectory.png`, `results/plots/auc_vs_human.png`, `results/plots/pass_rates.png`

## 5. Result Analysis
### Key Findings
1. Iterative rejection did not beat one-shot on either detector in mean score; all paired mean deltas were positive (worse for evasion).
2. On stylometric detector, iterative was significantly worse than one-shot for external text after FDR correction (p=0.0440).
3. Pass-rate changes were neutral or negative: no gain on word detector, and -13.3 percentage points on stylometric in both source conditions.
4. AUC vs human did not show iterative weakening detection compared to one-shot; one-shot generally had lower AUC than iterative on both detectors.

### Hypothesis Testing Results
- Null hypothesis (H0): mean paired difference (iterative - one_shot) = 0.
- Alternative (H1, directional expectation): iterative < one_shot (lower AI score).
- Observed: iterative > one_shot on average in all tested pairs.
- Conclusion: hypothesis not supported in this run.

### Comparison to Baselines
- One-shot improved over baseline for fresh texts on both detectors.
- Iterative often partially reversed that gain.
- On external texts, both rewrites increased AI scores relative to original external text, with iterative increasing more on stylometric detector.

### Visualizations
See:
- `results/plots/condition_comparison.png`
- `results/plots/iterative_trajectory.png`
- `results/plots/auc_vs_human.png`
- `results/plots/pass_rates.png`

### Surprises and Insights
- The strongest degradations occurred on external texts under stylometric detection, suggesting iterative loop can push style toward detector-recognized AI patterns.
- Iteration trajectory was non-monotonic but did not trend toward lower AI probability overall.

### Error Analysis
Common failure mode observed in sampled outputs:
- Iterative rounds tended to converge to polished, consistent explanatory style ("clean prose"), which correlates with higher stylometric AI scores.
- External long-form texts were substantially shortened by rewrite prompts (length compression), which may alter detector behavior independently of human-likeness.

### Limitations
- Small experimental sample size (15 + 15 items).
- No commercial detector API (e.g., Pangram) was available in this environment; detectors are local surrogates.
- Single generator model (`gpt-4.1-mini`) and single temperature.
- Rewrite token cap (220) compressed long external documents.
- One full run only; no multi-seed replication yet.

## 6. Conclusions
### Summary
Under this controlled run, iterative rejection context did not outperform a one-shot "less AI-sounding" instruction; instead, it slightly to moderately increased detector AI scores. The hypothesis that rejection-loop context inherently shifts generation into a more detector-evasive regime is not supported here.

### Implications
- Practical: users and evaluators should not assume iterative rejection loops are universally stronger evasion tactics.
- Theoretical: interaction context effects appear detector- and setup-dependent; mechanism claims require broader model/detector triangulation.

### Confidence in Findings
Moderate confidence for this specific setup (clear paired design, two detectors, significant negative effect on one detector/source pair). Broader confidence is limited by sample size, detector choice, and single-model scope.

## 7. Next Steps
### Immediate Follow-ups
1. Replicate with multiple seeds and larger sample size (>=100 paired items per source).
2. Add at least one external commercial detector API to test transfer beyond local classifiers.
3. Add matched-length controls to isolate style shift vs compression artifacts.

### Alternative Approaches
- Compare iterative rejection against explicit feature-targeted prompts (sentence length variability, discourse markers, hedging).
- Evaluate with additional model families (Claude, Gemini, open-weight models).

### Broader Extensions
- Run same protocol on domain-specific genres (student essays, support emails, legal text).
- Study whether iterative rejection improves human preference while harming detector evasion (or vice versa).

### Open Questions
- Which linguistic features are amplified by rejection loops in failing cases?
- Does iterative rejection help only for some model families or detector architectures?
- Is there an optimal number of rounds before style collapses into a detectable pattern?

## References
1. Krishna et al. (2023). *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*.
2. Lu et al. (2024). *Large Language Models can be Guided to Evade AI-Generated Text Detection*.
3. Zhou et al. (2024). *Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack*.
4. Relevant resource index and PDFs in `literature_review.md`, `resources.md`, and `papers/`.
