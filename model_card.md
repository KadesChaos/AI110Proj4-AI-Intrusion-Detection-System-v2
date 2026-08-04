# Model Card & Responsible-AI Reflection

## What the models do

Three models work together in this system:
- **RandomForestClassifier** — baseline binary classifier (normal vs. attack) trained on preprocessed UNSW-NB15 features.
- **TensorFlow feed-forward classifier** — a small dense network (10 hidden units, sigmoid output) doing the same binary classification task, retrained incrementally on streamed batches during monitoring.
- **Malicious-activity detector NN** — a second, slightly larger dense network (20→10→1) trained to flag malicious batches specifically, used to corroborate anomalies found by the `IsolationForest`.

None of these models were used to generate this codebase's logic directly — they are the subject of the project, not the tool used to build it. AI assistance (Claude) was used as a coding collaborator throughout development, described below.

## How AI assistance was used

I worked with Claude (Claude Code) throughout this project as a pair-programming collaborator: reviewing existing starter code for bugs, refactoring based on that review, generating the architecture diagram from the actual code (not from a description I wrote), and running the pipeline end-to-end to verify behavior rather than just to assume it worked. Every fix Claude proposed was reviewed and explicitly approved before being applied — nothing was auto-committed without a decision point.

### A helpful AI suggestion

When I asked for a code review of `data_handler.py`, Claude flagged that `pd.get_dummies` combined with the raw dataset's `attack_cat` column would create features that directly encode the label — since `attack_cat != "Normal"` is exactly what `label == 1` means. This is a real, easy-to-miss data leakage bug: a model trained with that column left in would show excellent offline accuracy while having learned nothing about actual traffic patterns, and would fail immediately on any data that didn't already carry a pre-computed attack category. Removing `attack_cat` before encoding was a small code change with an outsized correctness impact, and it's the kind of bug that's easy to miss just by reading code without inspecting the actual data — it only became obvious once we loaded the real dataset and printed `attack_cat.value_counts()` against `label.value_counts()`.

### A flawed or incomplete AI suggestion

Early on, when asked to "run it and show results," Claude generated a small synthetic CSV with made-up random columns (`cpu_load`, `network_traffic`, etc.) to smoke-test the pipeline before the real UNSW-NB15 files were available. This successfully validated that the code *ran*, but it was a flawed proxy for validating that the system *worked correctly* — random synthetic data has no real relationship between features and labels, so the "malicious activity DETECTED" log lines it produced during that early test were essentially meaningless noise, not evidence of a working detector. It also could have been mistaken for a real validation result if I hadn't asked for the real dataset next. The lesson: a green checkmark on synthetic data proves the code doesn't crash, not that the system does its job — the `attack_cat` leakage bug above only surfaced once real data replaced the synthetic placeholder.

## Limitations of this system

- **No evaluator gate before deployment.** Retrained models (via `update_model`) are put into use immediately after each streamed batch, with no held-out accuracy or drift check. A batch of bad or adversarial data could silently degrade the live model.
- **No human-in-the-loop review.** `WARNING - Malicious activity DETECTED` log lines are the only output — there's no analyst confirmation step, no false-positive/true-positive feedback loop, and no mechanism to override or suppress a bad alert before the system acts on it (by retraining on that data).
- **Evaluated only on the UNSW-NB15 dataset**, which is over a decade old and does not reflect current attack patterns, encrypted traffic norms, or modern protocols. Real-world deployment on live network traffic would likely see substantial distribution shift from this training data.
- **No adversarial robustness testing.** The detectors were not tested against inputs specifically crafted to evade anomaly/classification thresholds, which is a realistic concern for a security-facing system.
- **Streaming simulation, not real streaming.** `simulate_data_stream` replays a static test CSV in a loop — it does not ingest live network traffic, so timing, ordering, and volume characteristics of a real deployment are not represented here.
- **Binary label only.** The system detects "is this malicious" but (after removing `attack_cat` to prevent leakage) does not classify *which* attack category was detected, which would be useful operationally but was out of scope for this fix.
