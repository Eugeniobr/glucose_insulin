# Metabolic Twin MVP (Type 1 Diabetes, Experimental)

This is an MVP of a **personalized metabolic digital twin** for research.

## Important Disclaimer

- This project is **experimental**.
- It is for **research and simulation only**.
- It is **not** medical advice and must **not** be used for real insulin dosing decisions.

## Project Structure

```text
metabolic_twin/
  data/
  src/
    config.py
    preprocessing.py
    dataset.py
    model.py
    train.py
    evaluate.py
    simulate.py
    plots.py
  notebooks/
  README.md
```

## Expected Input Columns

Minimum columns in CSV:

- `patient_id`
- `timestamp`
- `glucose`
- `bolus`
- `basal`
- `carbs`

If `bolus`, `basal`, or `carbs` are missing, they are filled with zero.

## What the Pipeline Does

1. Reads CSV.
2. Sorts by patient and time.
3. Resamples each patient to 5-minute frequency.
4. Handles missing values.
5. Builds temporal features:
   - hour sine/cosine
   - day of week
   - glucose lags
   - rolling bolus/carbs windows
   - recent basal
   - time since last meal/bolus
6. Builds windows (2–6h lookback configurable; default 4h).
7. Trains GRU/LSTM for multi-horizon prediction (30/60/120 min).

## Install

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Train

```bash
python metabolic_twin/src/train.py \
  --csv metabolic_twin/data/your_data.csv \
  --out metabolic_twin/artifacts \
  --epochs 40 \
  --split patient \
  --model gru
```

## Evaluate

```bash
python metabolic_twin/src/evaluate.py \
  --csv metabolic_twin/data/your_data.csv \
  --artifacts metabolic_twin/artifacts \
  --plots-dir metabolic_twin/artifacts/plots \
  --split patient
```

Generated plots:
- real vs predicted glucose
- error by horizon
- residual distribution
- patient example

## Counterfactual Simulation

```bash
python metabolic_twin/src/simulate.py \
  --csv metabolic_twin/data/your_data.csv \
  --artifacts metabolic_twin/artifacts \
  --split patient \
  --sample-idx 0 \
  --bolus-mult 1.2 \
  --carbs-mult 0.8
```

This compares baseline forecast against a counterfactual where bolus/carbs are scaled in the input window.

## Notes

- Current version is intentionally simple and modular.
- Next iterations can include hybrid physiological models, neural ODEs, and Bayesian filters.

## Snapshot for Bot Integration

Generate a JSON snapshot consumed by the Telegram bot:

```bash
python metabolic_twin/src/run_inference_snapshot.py \
  --csv "EugênioSilva Rezende_glucose_4-19-2026.csv" \
  --artifacts metabolic_twin/artifacts \
  --output outputs/twin_simulation.json
```

Then use Telegram commands:
- `twin`
- `twin atualizar`
