# Downloaded Datasets

This directory contains datasets for LLM-generated text detection and adversarial evasion research.
Data files are intentionally excluded from git via `datasets/.gitignore`.

## Dataset 1: HC3 (Human ChatGPT Comparison Corpus)

### Overview
- **Source**: `Hello-SimpleAI/HC3` (HuggingFace)
- **Location**: `datasets/hc3_raw/`
- **Format**: JSONL
- **Scale in local copy**: `all.jsonl` has 24,322 records
- **Task fit**: Human-vs-AI text discrimination across QA domains
- **License**: See dataset card in `datasets/hc3_raw/README.md`

### Download Instructions

**Recommended (reproducible snapshot):**
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Hello-SimpleAI/HC3",
    repo_type="dataset",
    local_dir="datasets/hc3_raw",
    allow_patterns=["README.md", "all.jsonl", "finance.jsonl", "medicine.jsonl", "open_qa.jsonl", "reddit_eli5.jsonl", "wiki_csai.jsonl"]
)
```

### Loading
```python
import json
with open("datasets/hc3_raw/all.jsonl") as f:
    rows = [json.loads(next(f)) for _ in range(5)]
```

### Sample Data
- `datasets/hc3_raw/samples/sample.json`

## Dataset 2: AI Text Detection Pile

### Overview
- **Source**: `artem9k/ai-text-detection-pile` (HuggingFace)
- **Location**: `datasets/ai_text_detection_pile/`
- **Format**: Parquet shards
- **Scale in local copy**: 1,392,522 rows across 7 shards (~1.9 GB)
- **Columns**: `source`, `id`, `text`
- **Task fit**: Large-scale detector training/evaluation
- **License**: See dataset card in `datasets/ai_text_detection_pile/README.md`

### Download Instructions
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="artem9k/ai-text-detection-pile",
    repo_type="dataset",
    local_dir="datasets/ai_text_detection_pile",
    allow_patterns=["README.md", "data/*.parquet"]
)
```

### Loading
```python
import pandas as pd
df = pd.read_parquet("datasets/ai_text_detection_pile/data/train-00000-of-00007-bc5952582e004d67.parquet")
```

### Sample Data
- `datasets/ai_text_detection_pile/samples/sample.json`

## Dataset 3: Ghostbuster Reuters

### Overview
- **Source**: `acmc/ghostbuster_reuter` (HuggingFace)
- **Location**: `datasets/ghostbuster_reuter/`
- **Format**: Parquet
- **Scale in local copy**: 7,000 rows (~13 MB)
- **Columns**: `text`, `model`, `generated`, `results`
- **Task fit**: Detector analysis under model/domain variation
- **License**: See dataset card in `datasets/ghostbuster_reuter/README.md`

### Download Instructions
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="acmc/ghostbuster_reuter",
    repo_type="dataset",
    local_dir="datasets/ghostbuster_reuter",
    allow_patterns=["README.md", "data/*.parquet"]
)
```

### Loading
```python
import pandas as pd
df = pd.read_parquet("datasets/ghostbuster_reuter/data/train-00000-of-00001.parquet")
```

### Sample Data
- `datasets/ghostbuster_reuter/samples/sample.json`

## Notes
- HuggingFace unauthenticated rate limits may apply; set `HF_TOKEN` for faster/reliable pulls.
- `datasets` package script-loading was blocked for HC3 in this environment, so snapshot-based ingestion was used.
