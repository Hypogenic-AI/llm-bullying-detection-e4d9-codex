# Cloned Repositories

## 1) ai-detection-paraphrases
- **URL**: https://github.com/martiansideofthemoon/ai-detection-paraphrases
- **Location**: `code/ai-detection-paraphrases/`
- **Purpose**: Official code for DIPPER paraphrase attacks and retrieval-based defense (NeurIPS 2023).
- **Key files**:
  - `dipper_paraphrases/paraphrase.py`
  - `dipper_paraphrases/paraphrase_minimal.py`
  - `dipper_paraphrases/detect_detectgpt.py`
  - `dipper_paraphrases/detect_retrieval.py`
- **Requirements**: `requirements.txt`; GPU memory constraints for 11B DIPPER noted in repo docs.
- **Use for this hypothesis**: Strong baseline for iterative rewriting attacks and detector-evasion evaluation.

## 2) detect-gpt
- **URL**: https://github.com/eric-mitchell/detect-gpt
- **Location**: `code/detect-gpt/`
- **Purpose**: Reference implementation of DetectGPT curvature-based zero-shot detector.
- **Key files**:
  - `run.py`
  - `custom_datasets.py`
  - `paper_scripts/` (experiment pipelines)
- **Requirements**: `requirements.txt`.
- **Use for this hypothesis**: Important detector baseline for comparing one-shot vs iterative prompt rewriting.

## 3) lm-watermarking
- **URL**: https://github.com/jwkirchenbauer/lm-watermarking
- **Location**: `code/lm-watermarking/`
- **Purpose**: Official watermark generation and detection algorithms.
- **Key files**:
  - `extended_watermark_processor.py`
  - `watermark_processor.py`
  - `demo_watermark.py`
  - `app.py`
- **Requirements**: `requirements.txt`, `pyproject.toml`.
- **Use for this hypothesis**: Add provenance-based detector baseline less dependent on style features.

## Quick validation done
- Repositories cloned successfully.
- README and dependency files inspected.
- No full runs were executed due potential GPU/API dependencies and runtime cost.
