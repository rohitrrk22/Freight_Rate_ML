# Freight Rate Prediction (ML Engineer assessment)

Predicts the posted rate of 12,000 Nov-Dec 2025 loads from 48,000 labelled Jan-Oct loads.
Model: LightGBM on log(rate), L1 objective, 5 seeds, 45-day recency weights, linear daily-market term, regime-aware `quote_signal`.

## Layout
```
README.md  requirements.txt  score.py (provided)  run_pipeline.py  eda.py
validation_predictions.csv      <- submission file (load_id,predicted_rate)
december_chart_inputs.csv       <- completed December file (predicted_rate filled)
src/freight_rate/{data,features,model,validation}.py
scripts/{diagnostics,noise_regime_test,regime_causality_check,causal_variant_test}.py
tests/check_outputs.py
data/        original assessment CSVs (train_test, validation, template, december inputs)
outputs/     CV tables, fold-level predictions, diagnostics, figures, summary.json
scorer_results/candidate_december.png   <- chart produced by score.py
reports/     Freight_Rate_Report.docx, Freight_Rate_Loom_Deck.pptx, candidate_december.png
```

## Setup and run (tested in a clean Python 3.12 virtualenv)
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python eda.py                                  # figures + data-quality table
python run_pipeline.py                         # CV, ablations, final fit, both prediction files (~25 min, 1 core)
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
python scripts/diagnostics.py                  # quote_signal / half-life / seed comparisons, subgroup errors
python scripts/noise_regime_test.py            # causal noise-regime test for the quote_signal decision
python scripts/regime_causality_check.py       # is the regime knowable from the first 7 days of a month?
python scripts/causal_variant_test.py          # cost of strictly causal market/regime features (build_day_table_causal)
python tests/check_outputs.py                  # pre-submission checks
```
`python run_pipeline.py --reuse-cv` skips the slow CV/ablation sections and reuses the saved tables (~1 min); the final predictions are identical.
`--quick` runs 1-seed CV and writes to `outputs_quick/`; those metrics are NOT the final model's. The final fit always uses 5 seeds.
Place the four assessment CSVs in `data/` as `train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv` (the original December file has a blank `predicted_rate`; the pipeline writes the completed copy to the repo root).

## Validation
Forward-chaining, 5 folds, each trains on the past and scores the next 2 months (Nov-Dec is the future). All learned quantities are fitted inside each fold. Random K-fold appears only as a leakage diagnostic.
Metrics (original dollar scale): MAE, RMSE, MAPE = mean(|pred-actual|/actual), MedAPE, R2; "core" metrics exclude rows with absolute % error >= 50% (about 1.4% of rows, dominated by corrupted labels) and are always shown next to the unfiltered ones.

## Model configurations (every table states its own)
| Results | Evaluation | Configuration |
|---|---|---|
| Comparisons / ablations / causal variants (report Tables 1b, 4, 6, 7) | 5 forward folds | 1 seed, 600 trees, lr 0.04 |
| Noise-regime test (Table 5) | 2 causal splits inside August | 2 seeds, 600 trees |
| Headline CV (Table 9, subgroup tables) | 5 forward folds | 3 seeds, 1,000 trees, lr 0.03, 15 leaves |
| Final predictions | fit on all Jan-Oct | 5 seeds, 1,000 trees, lr 0.03, 15 leaves |

## Verification
The full pipeline and every script were run from scratch in a freshly created virtualenv from `requirements.txt` (Python 3.12, pandas 2.2.3, numpy 2.1.3, LightGBM 4.5.0). Predictions, the December file, CV tables, ablations and diagnostics regenerated with a maximum difference of 0; the scorer chart was pixel-identical. Verified on one machine and software stack.

## Assumptions
- Offline batch scoring: same-day `market_index` rows (and other days of the same month, for the deviation from the monthly mean) are available. Uses features only, never labels. A strictly causal variant (`build_day_table_causal`: trailing 28-day market mean, month-to-date regime) is tested in `scripts/causal_variant_test.py`: five-fold median error is unchanged and the August noise-regime test is slightly worse. Use it if loads must be scored before the rest of the month exists.
- `quote_signal` regime (+, -, noise) comes from the monthly skew of the feature itself; the first 7 days of a month reproduce all 12 classifications.
- A date with no usable `market_index` gets no daily adjustment; missing weight stays NaN (LightGBM native).
