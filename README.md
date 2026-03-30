# LLM Bullying Detection Research Run

This project tests whether iterative rejection feedback ("too AI-sounding") makes LLM rewrites harder to detect than a one-shot "sound less like AI" instruction. We ran a controlled paired experiment with real OpenAI API outputs and two independent local detectors.

## Key Findings
- In this run, iterative rejection did **not** outperform one-shot rewriting.
- Paired score deltas (`iterative_final - one_shot`) were positive for both detectors in both source settings.
- Stylometric detector showed statistically significant degradation for iterative on external text (FDR-corrected `p=0.0440`).
- Pass rates were unchanged on word detector and dropped by 13.3 points on stylometric detector.

See [REPORT.md](./REPORT.md) for full methodology, statistics, limitations, and interpretation.

## Reproduce
1. Activate environment:
   - `source .venv/bin/activate`
2. Run experiment:
   - `python -u -c "from src.run_research import run, Config; run(Config(n_fresh_items=15, n_external_items=15, iterative_rounds=4, results_dir='results'))"`
3. Review outputs:
   - `results/metrics.json`
   - `results/stat_tests.csv`
   - `results/summary_table.csv`
   - `results/plots/*.png`

## File Structure
- `planning.md`: motivation, novelty, and preregistered plan
- `src/run_research.py`: end-to-end experiment pipeline
- `results/`: generated outputs, tables, metrics, and plots
- `REPORT.md`: full research report
- `literature_review.md`, `resources.md`: pre-gathered resource synthesis
