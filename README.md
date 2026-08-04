# AI Intrusion Detection System

## Summary

This project is a network intrusion detection system (IDS) that uses machine learning to flag malicious traffic in real time. It trains on the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset, a benchmark dataset of real and synthetic network traffic labeled as normal or one of nine attack categories (DoS, Exploits, Fuzzers, Reconnaissance, etc.), and then simulates a live traffic stream, flagging anomalous and malicious batches as they arrive.

This system builds on a project I originally completed in 2024 as the final project for an undergraduate Machine Learning course. That version already trained the same three models (RandomForest, a TensorFlow classifier, and a malicious-activity detector NN) on the UNSW-NB15 dataset, but it stopped there: it evaluated once on a static test set and had no continuous monitoring loop and no retraining, so it couldn't act on new data after the initial training run.

Revisiting it here, I reviewed the original implementation for bugs and design issues, then reworked it to be more correct and more capable: adding the streaming/continuous-monitoring loop and retraining step, fixing data handling edge cases, removing a data leakage bug, replacing fragile global state with a proper class, and documenting the architecture and its gaps clearly for anyone picking this up after me.

It matters because traditional signature-based IDS tools only catch attacks they've already seen. A learned model can generalize to traffic patterns that look statistically abnormal even if no rule was ever written for them. The tradeoff is that it can also be wrong in ways a hand-written rule wouldn't be, which is part of why this project treats the detector as a first-pass filter, not a final verdict (see Design Decisions and `model_card.md`).

All machine learning in this project runs entirely locally: scikit-learn (RandomForest, IsolationForest) and TensorFlow/Keras models are trained and run on your own machine, with no API keys, external services, or hosted model calls required. AI coding assistance was used during development to help review, fix, and document the code (as per the project requirements, see `model_card.md`), but the IDS itself has no runtime dependency on any external AI service.

## Architecture Overview

See [diagrams/archtecture.mmd](diagrams/archtecture.mmd) for the full Mermaid diagram. At a high level:

1. **Data ingestion** (`data_handler.py`): loads the UNSW-NB15 CSVs, drops the `attack_cat` label-leakage column, fills missing values, one-hot encodes categorical fields, and scales numeric features with a `StandardScaler`.
2. **Trainer** (`train.py`, `models.py`): trains three models on the full training set: a `RandomForestClassifier`, a small TensorFlow feed-forward classifier, and a separate TensorFlow "malicious activity" detector.
3. **Detector stack** (retriever-equivalent): an `IsolationForest` anomaly detector and the malicious-activity NN both score each streamed batch of test data.
4. **Streaming loop** (`ids_system.py`): replays the test set in batches (`simulate_data_stream`), checks each batch for anomalies, describes any hits in a table, escalates to a `WARNING` log if the malicious detector agrees, and attempts to retrain the TF monitor model on that batch before continuing.
5. **Evaluator** (`train.py: evaluate_model`, `update_model`): before any retrained model replaces the live one, its accuracy is checked against a held-out validation split (carved out of the test set before streaming starts). If the update would regress accuracy beyond a small tolerance, it's rejected and the model rolls back to its pre-update weights. This closes the "silently retrain on every batch, alert or not" gap the original streaming loop had.
6. **Human-in-the-loop review: still missing.** There is no analyst confirmation step before a `WARNING` alert is logged, and no way to mark an alert as a false positive. This remains a known gap. See Design Decisions.

## Setup Instructions

1. **Clone the repo**
   ```
   git clone https://github.com/KadesChaos/AI110Proj4-AI-Intrusion-Detection-System-v2.git
   cd AI110Proj4-AI-Intrusion-Detection-System-v2
   ```

2. **Use Python 3.9 to 3.12.** TensorFlow does not yet publish wheels for Python 3.13+. Check your version:
   ```
   python --version
   ```
   If you're on a newer interpreter, install a compatible version (e.g. via `pyenv`, or on Windows the `py` launcher: `py -3.11`) and use it in the next step.

3. **Create and activate a virtual environment**
   ```
   py -3.11 -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```
   pip install pandas scikit-learn tensorflow tabulate
   ```

5. **Run the system**
   ```
   python ids_system.py
   ```
   This trains all three models on `UNSW_NB15_training-set.csv`, then starts the continuous monitoring loop against `UNSW_NB15_testing-set.csv`, streaming batches every 10 seconds indefinitely. Stop it with `Ctrl+C`.

   Logs are written both to the console (with color) and to a timestamped file in `~/Downloads/ids_system_<timestamp>.log` (ANSI codes stripped).

## Sample Interactions

The system doesn't take conversational input. Its "input" is a batch of network traffic rows, and its "output" is a detection decision plus a log entry. Below are real examples captured from a run against the UNSW-NB15 test set.

**Example 1: Normal batch, no action**
```
Input:  batch of 100 rows from UNSW_NB15_testing-set.csv (mostly label=0, "Normal")
Output: no anomalies detected by IsolationForest -> batch skipped, no log entry
```

**Example 2: Anomaly detected, not classified as malicious**
```
Input:  batch containing a statistical outlier row (e.g. unusual CPU load / packet timing)
Output:
2026-08-04 18:26:43 - INFO - Data preprocessed successfully.
Malicious activity detected:
+---------+--------------------------+---------+----------+
|   Index | Feature                  |   Value | Status   |
+=========+==========================+=========+==========+
|      89 | CPU Load                 |   -0.72 | OK       |
|      89 | Network Traffic Volume   |    1.38 | OK       |
...
(anomaly logged for review, malicious detector did not flag it above threshold,
 TF monitor model still retrained on this batch)
```

**Example 3: Malicious activity flagged**
```
Input:  batch containing a row scored above the malicious-detector threshold
Output:
2026-08-04 18:26:43 - WARNING - Malicious activity DETECTED! Details:
Malicious activity detected:
+---------+--------------------------+---------+----------+
|   Index | Feature                  |   Value | Status   |
+=========+==========================+=========+==========+
|      97 | CPU Load                 |    0.32 | OK       |
|      97 | Network Traffic Volume   |   -1.51 | OK       |
|      97 | Memory Usage             |    1.69 | OK       |
...
```
This confirms the full path works end to end: stream batch, detect anomaly, cross-check with malicious detector, escalate to a `WARNING` log, then attempt to retrain the monitor model on the new data.

## Reliability Component: Evaluation Gate

`update_model` in `train.py` acts as a self-checking evaluator for the retraining step: before a retrained model replaces the live monitor model, it's re-evaluated against a held-out validation split (carved out of the test set before streaming starts, so it's never trained on). If accuracy would regress beyond a small tolerance, the update is rejected and the model rolls back to its pre-update weights. Without this, the original loop accepted every retraining update unconditionally, alert or not, with no way to catch a batch that made the model worse.

**Input:** a streamed batch that, after retraining on it, would drop the model's held-out validation accuracy.
**Behavior:** `update_model` snapshots weights, retrains, re-evaluates against the validation split, compares against the pre-update baseline.
**Result (captured from a real run):**
```
Update REJECTED: validation accuracy would drop from 0.9611 to 0.8386. Rolled back to previous weights.
Update REJECTED: validation accuracy would drop from 0.9611 to 0.8742. Rolled back to previous weights.
Update REJECTED: validation accuracy would drop from 0.9611 to 0.9380. Rolled back to previous weights.
Model updated with new data. Validation accuracy: 0.9611 -> 0.9614
Model updated with new data. Validation accuracy: 0.9614 -> 0.9620
```

**Input:** a streamed batch that, after retraining, holds or improves held-out validation accuracy.
**Behavior:** same evaluation check, but the new accuracy clears the tolerance threshold.
**Result:** the update is accepted and the new weights are kept, as shown in the last two lines above, accuracy climbed from 0.9611 to 0.9620 across those accepted updates, and continued rising to 0.966 later in the same run.

This is a real functional check, not a cosmetic log message: three consecutive updates were actually rejected and rolled back in that run, and the model that stayed in production was measurably more accurate on held-out data than if every update had been accepted blindly.

## Design Decisions

- **Three separate models instead of one.** RandomForest gives a fast, interpretable baseline. The TensorFlow classifier and the dedicated malicious-activity NN add non-linear pattern detection and let the alerting logic require agreement between an anomaly detector and a classifier before escalating to `WARNING`, rather than trusting a single model's threshold. Trade-off: more training time and more moving parts to keep in sync (e.g. all three need the same preprocessing pipeline).
- **Unsupervised anomaly detection (`IsolationForest`) gates the supervised malicious-activity check.** This mirrors a real SOC (Security Operations Center) workflow: broad, cheap anomaly triage first, narrower and more expensive classification second. Trade-off: a genuinely malicious but statistically "normal-looking" batch could slip past the first gate.
- **`attack_cat` is dropped before encoding.** This column directly determines the binary `label` (anything other than `"Normal"` implies `label=1`), so leaving it in as a one-hot-encoded feature would let the model trivially memorize the label instead of learning from actual traffic features. That's a subtle form of data leakage that would have made offline accuracy look artificially perfect. This was caught during review and fixed in `data_handler.py`.
- **The scaler and feature-column list are encapsulated in a `Preprocessor` class**, not module-level globals. The original code used global mutable state, which meant two pipelines (e.g. testing vs. production, or parallel experiments) in the same process would silently overwrite each other's fitted scaler. Trade-off: a small amount of extra indirection (a class instance) for a meaningful correctness guarantee.
- **Evaluation gate before accepting a retrained model.** `update_model` now snapshots the model's weights, retrains, and re-evaluates against a held-out validation split carved out of the test set before streaming starts. If accuracy would drop by more than a small tolerance, the update is rejected and the previous weights are restored. Trade-off: this costs an extra forward pass per batch and needs a validation split set aside up front, and the accuracy-delta threshold is a blunt instrument (a smarter version might track a moving baseline, or use a metric better suited to imbalanced classes like F1, instead of a fixed tolerance).
- **Human-in-the-loop review is still missing, by design of what's *not yet built*.** There is no analyst confirming a `WARNING` is a true positive before the system logs it, and no feedback mechanism to mark a false positive so the system learns from that correction. This is called out explicitly in the architecture diagram as unfinished work rather than glossed over, because a security system that only self-checks its own accuracy still has no way to catch a mistake a human would have caught (e.g. a `WARNING` that's technically anomalous but operationally benign).

## Testing Summary

- **What worked:** The full pipeline was run end-to-end (training, streaming detection, alerting, retraining) using the real UNSW-NB15 dataset. All three models trained without errors, `IsolationForest` correctly identified injected outlier rows, and the malicious-activity NN correctly escalated at least one batch to a `WARNING` log during a live test run.
- **What didn't work initially:**
  - The original `data_handler.py` used `ffill` alone, which leaves leading rows with unfilled `NaN`s if the very first rows in a batch are missing. Fixed by adding `bfill` as a second pass.
  - The original scaler/feature-column state was global, meaning a second training run in the same process could silently re-fit the scaler and invalidate already-trained models with no warning. Fixed by encapsulating that state in a `Preprocessor` class with an explicit "already fitted" guard.
  - `load_data` originally swallowed `FileNotFoundError` and returned `None`, which surfaced later as a confusing `NoneType` crash deep inside preprocessing instead of a clear error at the source. Fixed by letting the exception propagate.
  - Running on Python 3.14 failed outright because TensorFlow has no published wheel for it yet. Resolved by pinning the environment to Python 3.11.
- **What I learned:** Most of the real bugs weren't in the model code. They were in the boundary conditions of the data pipeline (missing-value edge cases, global state, silently swallowed errors) and in a subtle data leakage column that would have made the model look better than it actually was. Testing against the real dataset, not just synthetic placeholder data, was what surfaced the `attack_cat` leakage issue, since it wasn't visible from reading the code alone.

## Reflection

See `model_card.md` for the graded responsible-AI reflection, including how AI assistance was used during development, one helpful and one flawed AI suggestion encountered, and this system's limitations.
