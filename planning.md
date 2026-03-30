# Research Plan: Iterative Rejection for AI-Text Detector Evasion

## Motivation & Novelty Assessment

### Why This Research Matters
AI-text detectors are increasingly used in education, moderation, and fraud screening, so understanding practical evasion dynamics is directly relevant to policy and product reliability. If a simple interaction pattern (iterative rejection as "too AI-sounding") reliably reduces detector sensitivity, current detector deployments may overestimate robustness and create false confidence for downstream decision makers.

### Gap in Existing Work
The reviewed literature shows that paraphrasing and adversarial rewriting can reduce detection rates, but most studies do not isolate rejection-context phrasing itself as the independent variable. Existing work often mixes prompt engineering, paraphrasers, and detector-aware optimization, leaving an open causal question: whether iterative rejection unlocks controllable anti-detector features beyond one-shot instructions.

### Our Novel Contribution
We directly compare one-shot de-AI instruction versus multi-turn rejection-context rewriting under matched source texts, model, and token budgets. We also test whether gains persist on text not generated in the immediate conversation context, and evaluate transfer across two detector families to distinguish detector overfitting from broader regime shift.

### Experiment Justification
- Experiment 1: One-shot vs iterative rejection on model-generated answers. Why needed: establishes the core effect under controlled generation.
- Experiment 2: Apply both rewriting protocols to externally sourced AI text (from dataset, not this session). Why needed: tests the claim that iterative rejection still works on out-of-context text.
- Experiment 3: Round-by-round trajectory analysis (iteration 1..N). Why needed: determines whether improvements are monotonic and identifies convergence behavior.
- Experiment 4: Cross-detector transfer (pretrained detector + independent detector). Why needed: tests whether effects are detector-specific or general.

## Research Question
Does iterative rejection of outputs as "too AI-sounding" produce lower AI-detection scores and higher detector pass rates than a single instruction to "sound less like AI," and if so, is the gain consistent with feature steering vs broader generation-regime shift?

## Background and Motivation
Prior work (Krishna et al., Lu et al., Zhou et al.) demonstrates detector fragility under paraphrase-style attacks. However, practical user workflows often involve conversational rejection loops rather than explicit adversarial optimization. This project tests that interaction-level mechanism directly using real LLM API outputs and controlled detector evaluation.

## Hypothesis Decomposition
- H1 (Primary): Iterative rejection lowers detector AI probability more than one-shot rewriting.
- H2 (Robustness): Iterative rejection yields higher detector pass rate (below threshold) than one-shot across detectors.
- H3 (Context transfer): The iterative advantage persists on externally sourced AI texts not generated in current dialogue.
- H4 (Mechanism proxy): Improvement from round 1 to later rounds shows a nonlinear trajectory suggestive of regime shift rather than simple linear style polishing.

Independent variables:
- Rewrite condition: `one_shot`, `iterative_round_k`.
- Text source: `fresh_model_generation`, `external_ai_text`.
- Detector: `roberta_detector`, `stylometric_detector`.

Dependent variables:
- Detector AI score (continuous probability).
- Binary pass indicator under preset thresholds.
- Relative score reduction from baseline.

## Proposed Methodology

### Approach
Use a controlled within-item design. For each source text, generate both treatment outputs with the same model and fixed decoding parameters. Evaluate all outputs on two independent detectors and perform paired statistical tests.

### Experimental Steps
1. Sample prompts and source texts from local datasets (HC3 + AI-text corpora) with fixed random seed.  
   Rationale: reproducible, diverse input topics.
2. Generate baseline model outputs with real API calls (single model, fixed temperature/top_p).  
   Rationale: ensures realistic LLM behavior under deployment-like settings.
3. Apply one-shot rewriting prompt to each baseline output.  
   Rationale: strong practical baseline matching user intent.
4. Apply iterative rejection protocol (N rounds, repeated "too AI-sounding" feedback).  
   Rationale: treatment under test.
5. Evaluate all variants with two detectors and compute paired deltas, pass rates, and AUC where labels available.  
   Rationale: measure both score-level and decision-level impact with transfer checks.
6. Run hypothesis tests (paired t-test + Wilcoxon robustness, bootstrap CI, effect size).  
   Rationale: robust inference under potential non-normality.
7. Perform failure analysis on cases where iterative fails or degrades quality.  
   Rationale: constrain claims and identify edge conditions.

### Baselines
- Baseline A: Original model-generated text (no rewrite).
- Baseline B: One-shot "sound less AI-generated" rewrite.
- Comparison condition: Iterative rejection (up to 4 rounds; analyze each round and final).

### Evaluation Metrics
- Primary: Mean detector score delta (`final - baseline`) and pass rate improvement.
- Secondary: AUC by condition on labeled subsets, TPR@target FPR when supported.
- Mechanism proxy: marginal gain by round (`r1→r2`, `r2→r3`, `r3→r4`).
- Quality guardrails: length ratio and lexical diversity changes (to ensure non-trivial rewrites).

### Statistical Analysis Plan
- Significance level: alpha = 0.05.
- Primary paired comparison: iterative-final vs one-shot score per item.
- Tests: paired t-test (if approximately normal), Wilcoxon signed-rank as non-parametric confirmatory test.
- Effect size: Cohen's d for paired differences and Cliff's delta robustness.
- Confidence intervals: 95% bootstrap CI (10,000 resamples).
- Multiple comparisons correction: Benjamini-Hochberg FDR across detector/source combinations.

## Expected Outcomes
Support for hypothesis:
- Iterative-final has significantly lower detector scores than one-shot on both detectors.
- Pass-rate uplift is positive and statistically significant.
- Similar effect appears on external text condition.

Refutation patterns:
- No significant difference or one-shot outperforms iterative.
- Effects collapse on second detector (indicating detector-specific overfit).

## Timeline and Milestones
- Milestone 1 (Planning, 20-30 min): finalize design and preregister tests.
- Milestone 2 (Setup + EDA, 20-30 min): environment, data checks, sample construction.
- Milestone 3 (Implementation, 45-70 min): scripts for API generation, detection, analysis.
- Milestone 4 (Experiments, 45-70 min): run main conditions and store raw outputs.
- Milestone 5 (Analysis + report, 40-60 min): stats, plots, REPORT.md + README.md.
- Buffer (25%): debugging/API retries and reruns.

## Potential Challenges
- API limits or transient failures: implement retries and checkpointing per item.
- Detector mismatch/domain shift: evaluate across two detectors and report disagreement.
- Cost/time: use staged sample size (pilot n=30, then n=120 if stable).
- Semantic drift during rewriting: enforce concise rewrite instructions and check quality proxies.

## Success Criteria
- Completed experiments with real model API calls across all conditions.
- Reproducible scripts and saved raw/aggregated outputs in `results/`.
- Statistically grounded comparison with effect sizes and confidence intervals.
- Clear conclusion on whether iterative rejection materially outperforms one-shot.
