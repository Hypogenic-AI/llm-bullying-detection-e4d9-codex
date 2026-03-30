# Downloaded Papers

This directory contains papers relevant to the hypothesis on iterative rejection prompts and AI-text detector evasion.

## Search notes
- Primary script used: `.claude/skills/paper-finder/scripts/find_papers.py` (backend unavailable in this environment).
- Manual fallback used: OpenAlex API search over detector-evasion/watermarking/paraphrase queries.
- Access issue: JAIR paper "Detecting AI-Generated Text: Factors Influencing Detectability with Current Methods" returned HTTP 403 from this environment.

## Downloaded PDFs (12)

1. **Can AI-Generated Text be Reliably Detected?** (2023)
- File: `2023_can_ai_generated_text_be_reliably_detected.pdf`
- Why relevant: Broad benchmark and failure analysis for detector robustness.

2. **Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense** (2023)
- File: `2023_paraphrasing_evades_detectors_of_ai_generated_text_but_retrieval_is_an.pdf`
- Why relevant: Direct evidence that rewriting/paraphrasing can collapse detector TPR; introduces retrieval defense.

3. **Large Language Models can be Guided to Evade AI-Generated Text Detection** (2023 / TMLR 2024)
- File: `2023_large_language_models_can_be_guided_to_evade_ai_generated_text_detecti.pdf`
- Why relevant: Prompt-only detector evasion (SICO), very close to the rejection-loop hypothesis.

4. **Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack** (2024)
- File: `2024_humanizing_machine_generated_content_evading_ai_text_detection_through.pdf`
- Why relevant: Adversarial perturbation framework under white-box/black-box settings.

5. **On the Possibilities of AI-Generated Text Detection** (2023)
- File: `2023_on_the_possibilities_of_ai_generated_text_detection.pdf`
- Why relevant: Limits and assumptions of current detection methods.

6. **RADAR: Robust AI-Text Detection via Adversarial Learning** (2023)
- File: `2023_radar_robust_ai_text_detection_via_adversarial_learning.pdf`
- Why relevant: Robust detector training against perturbation attacks.

7. **Provable Robust Watermarking for AI-Generated Text** (2023)
- File: `2023_provable_robust_watermarking_for_ai_generated_text.pdf`
- Why relevant: Watermark perspective for resilient provenance detection.

8. **Are AI-Generated Text Detectors Robust to Adversarial Perturbations?** (2024)
- File: `2024_are_ai_generated_text_detectors_robust_to_adversarial_perturbations.pdf`
- Why relevant: Focused adversarial stress test of detectors.

9. **Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors** (2025)
- File: `2025_your_language_model_can_secretly_write_like_humans_contrastive_paraphr.pdf`
- Why relevant: New paraphrase attack framing aligned with iterative style-steering.

10. **BiMarker: Enhancing Text Watermark Detection for Large Language Models with Bipolar Watermarks** (2025)
- File: `2025_bimarker_enhancing_text_watermark_detection_for_large_language_models_.pdf`
- Why relevant: Improved watermark detectability under stronger attacks.

11. **Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection** (2023 EMNLP)
- File: `2023_hidding_the_ghostwriters_an_adversarial_evaluation_of_ai_generated_stu.pdf`
- Why relevant: Education-domain adversarial detector evaluation.

12. **Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection** (2024 arXiv variant)
- File: `2024_hidding_the_ghostwriters_an_adversarial_evaluation_of_ai_generated_stu.pdf`
- Why relevant: Updated/alternate version from arXiv mirror.

## Deep-reading set (chunked)
For detailed method extraction, the following were chunked and read across all chunks:
- `2023_paraphrasing_evades_detectors_of_ai_generated_text_but_retrieval_is_an.pdf`
- `2023_large_language_models_can_be_guided_to_evade_ai_generated_text_detecti.pdf`
- `2024_humanizing_machine_generated_content_evading_ai_text_detection_through.pdf`

Chunk outputs are under `papers/pages/` with manifest files.

## Metadata files
- `papers/manual_search_results.json`: ranked fallback search output.
- `papers/selected_papers.json`: selected download set.
- `papers/papers_metadata.json`: normalized metadata for downloaded PDFs.
