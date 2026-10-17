"""Create report-ready evidence assets from the project's saved results and figures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).resolve().parent

FIGURES = {
    "week2_wait_length_distribution.png": ROOT / "week 2" / "images" / "figure-02-15-target-distribution-analysis-wait-length.png",
    "week2_temporal_eda.png": ROOT / "week 2" / "images" / "figure-05-17-temporal-analysis.png",
    "week2_chronological_ordering_finding.png": ROOT / "week 2" / "images" / "figure-15-finding-3-the-dataset-is-chronologically-ordered-random-splitting-causes-data-leakage.png",
    "week4_target_class_distribution.png": ROOT / "week 4" / "images" / "fig1_class_distribution.png",
    "week4_daily_caller_by_target.png": ROOT / "week 4" / "images" / "fig2_daily_caller_by_target.png",
    "week4_selected_model_training_history.png": ROOT / "week 4" / "images" / "fig_train_exp2_capacity.png",
    "week4_test_confusion_matrix.png": ROOT / "week 4" / "images" / "fig8_confusion_matrix.png",
    "week4_test_roc_curve.png": ROOT / "week 4" / "images" / "fig10_roc_curve.png",
    "week5_lstm_training_history.png": ROOT / "week 5" / "images" / "figure-01-training-loss-plots.png",
    "week5_test_accuracy_and_confusion_matrix.png": ROOT / "week 5" / "images" / "fig9_test_results.png",
    "week5_test_roc_curve.png": ROOT / "week 5" / "images" / "fig10_roc_curve.png",
    "week6_intent_distribution_and_duplicates.png": ROOT / "week 6" / "images" / "intent_class_distribution_and_duplicates.png",
    "week6_test_confusion_matrices.png": ROOT / "week 6" / "images" / "confusion_matrices.png",
    "week6_test_multiclass_roc_curves.png": ROOT / "week 6" / "images" / "roc_curves.png",
    "week6_training_history.png": ROOT / "week 6" / "images" / "training_loss.png",
}

LANES = [
    {
        "label": "WEEK 2\nEDA ONLY",
        "color": "#EAF2F8",
        "edge": "#2874A6",
        "steps": [
            ("1. Dataset", "Existing Kaggle simulation CSV\n51,708 rows x 9 columns"),
            ("2. Inspect", "Types, missingness,\nduplicates, time parsing"),
            ("3. Explore", "wait_length is the stated\nfuture target; zero-heavy\nand right-skewed"),
            ("4. Split strategy", "Notebook makes a proposed\nday-level 70/15/15 split;\n31,898 / 9,206 / 10,604 rows"),
            ("5. Interpret", "EDA only: no model fit,\npredictions or test scores\nin Week 2"),
        ],
    },
    {
        "label": "WEEK 4\nBINARY FFNN",
        "color": "#E8F8F5",
        "edge": "#148F77",
        "steps": [
            ("1. Dataset", "Same local call-centre CSV;\ntarget = meets_standard"),
            ("2. Inspect / EDA", "Imbalance: 91.83% True,\n8.17% False; time trends"),
            ("3. Preprocess", "Sort by start time; derive\n8 features; drop label leak\nand post-arrival fields"),
            ("4. Split / scale", "Chronological 70/15/15;\nfit scaler on train only"),
            ("5. Train / select", "4 Keras FFNN settings;\nselect on validation\nFalse-class F1"),
            ("6. Predict", "Selected Exp 2; sigmoid\nprobabilities, threshold 0.5"),
            ("7. Evaluate", "Held-out precision/recall/F1,\nROC-AUC, confusion matrix;\ninterpret weak generalization"),
        ],
    },
    {
        "label": "WEEK 5\nSEQUENTIAL LSTM",
        "color": "#FEF5E7",
        "edge": "#CA6F1E",
        "steps": [
            ("1. Dataset", "Same local CSV;\nnext-call meets_standard"),
            ("2. Inspect / EDA", "Validate rows, columns,\nmissingness and order"),
            ("3. Build sequences", "12 arrival features;\nprevious 20 calls per day;\nnever cross dates"),
            ("4. Split / scale", "Final 20% raw rows = future\ntest; train/validation within\ndevelopment; train-only scaler"),
            ("5. Train / select", "6 PyTorch LSTM settings;\nvalidation balanced accuracy;\nweighted BCE"),
            ("6. Predict", "Selected 32-unit LSTM;\nprobabilities thresholded at 0.5"),
            ("7. Evaluate", "Future-test metrics, baseline,\nROC-AUC, confusion matrix;\nmodel below baseline"),
        ],
    },
    {
        "label": "WEEK 6\nINTENT ROUTING",
        "color": "#F4ECF7",
        "edge": "#7D3C98",
        "steps": [
            ("1. Dataset", "Bitext customer-support\nrequests; 26,872 x 5;\n27 intent labels"),
            ("2. Inspect / EDA", "No missing/blank fields;\nnormalize request text;\nidentify duplicate groups"),
            ("3. Preprocess", "Group case/whitespace-\nnormalized duplicates;\nencode labels / tokens"),
            ("4. Split / vectorize", "Stratified group split 70/15/15;\nTF-IDF vocabulary fit on\ntraining text only"),
            ("5. Train models", "TF-IDF MLP and token LSTM;\nselect/checkpoint on\nvalidation macro F1"),
            ("6. Predict", "Score held-out requests\nfor 27 intent classes"),
            ("7. Evaluate", "Macro P/R/F1, OvR macro\nROC-AUC, per-intent scores,\nconfusion matrices"),
        ],
    },
]


def load_results() -> tuple[dict, dict, dict]:
    week4 = json.loads((ROOT / "week 4" / "results.json").read_text(encoding="utf-8"))
    week5 = json.loads((ROOT / "week 5" / "outputs" / "week5_results.json").read_text(encoding="utf-8"))
    week6 = json.loads((ROOT / "week 6" / "outputs" / "week6_results.json").read_text(encoding="utf-8"))
    return week4, week5, week6


def copy_project_figures() -> None:
    for output_name, source in FIGURES.items():
        if not source.is_file():
            raise FileNotFoundError(f"Expected existing project figure is missing: {source}")
        shutil.copy2(source, ASSET_DIR / output_name)


def draw_flowchart() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(20, 11.8))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 19.7)
    ax.set_ylim(0.1, 11.3)
    ax.axis("off")

    ax.text(
        9.85,
        10.95,
        "ACTUAL PROJECT PIPELINES | WEEKS 2, 4, 5 AND 6",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color="#17324D",
    )
    ax.text(
        9.85,
        10.53,
        "Separate targets and experiments are shown as separate lanes; Week 2 is EDA only.",
        ha="center",
        va="center",
        fontsize=11,
        color="#435466",
    )

    left = 2.55
    box_width = 2.26
    gap = 0.19
    box_height = 1.48
    row_centers = [9.05, 6.95, 4.85, 2.75]

    for lane, center_y in zip(LANES, row_centers):
        y = center_y - box_height / 2
        ax.add_patch(
            FancyBboxPatch(
                (0.18, y - 0.12),
                19.30,
                box_height + 0.24,
                boxstyle="round,pad=0.025,rounding_size=0.13",
                facecolor=lane["color"],
                edgecolor=lane["edge"],
                linewidth=1.1,
                alpha=0.43,
                zorder=0,
            )
        )
        ax.text(
            1.35,
            center_y,
            lane["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=lane["edge"],
        )
        steps = lane["steps"]
        for index, (heading, description) in enumerate(steps):
            x = left + index * (box_width + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_width,
                    box_height,
                    boxstyle="round,pad=0.045,rounding_size=0.12",
                    facecolor="white",
                    edgecolor=lane["edge"],
                    linewidth=1.45,
                    zorder=2,
                )
            )
            ax.text(
                x + box_width / 2,
                y + box_height * 0.70,
                heading,
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color=lane["edge"],
                zorder=3,
            )
            ax.text(
                x + box_width / 2,
                y + box_height * 0.37,
                description,
                ha="center",
                va="center",
                fontsize=7.75,
                color="#253746",
                linespacing=1.25,
                zorder=3,
            )
            if index < len(steps) - 1:
                arrow_start = (x + box_width + 0.015, center_y)
                arrow_end = (x + box_width + gap - 0.015, center_y)
                ax.add_patch(
                    FancyArrowPatch(
                        arrow_start,
                        arrow_end,
                        arrowstyle="-|>",
                        mutation_scale=11,
                        color=lane["edge"],
                        linewidth=1.3,
                        zorder=4,
                    )
                )

    ax.text(
        9.85,
        0.52,
        "Week 4/5 predict service-standard outcome; Week 6 predicts request intent. "
        "Week 2 proposes waiting-time prediction but trains no model.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#435466",
    )
    fig.savefig(ASSET_DIR / "actual_project_pipeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "actual_project_pipeline.svg", bbox_inches="tight")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_evidence_markdown(week4: dict, week5: dict, week6: dict) -> None:
    w4m = week4["test_metrics"]
    w5m = week5["test_metrics"]
    w6m = week6["metrics"]
    w4_exp = next(exp for exp in week4["experiments"] if exp["id"] == week4["best_experiment_id"])
    w4_hist = week4["history"][week4["best_experiment_id"]]
    w5_params = week5["selected_parameters"]
    w6_mlp = w6m["Feed-forward MLP"]
    w6_lstm = w6m["PyTorch LSTM"]
    mlp_val = week6["mlp_validation_macro_f1_by_epoch"]
    lstm_val = week6["lstm_validation_macro_f1_by_epoch"]
    mlp_loss = week6["mlp_training_loss"]
    lstm_loss = week6["lstm_training_loss"]

    content = f"""# Week 7 Technical Evidence Pack

This folder contains an implementation-specific pipeline flowchart, copies of existing project figures, verified intermediate results, suggested report snippets, and a screenshot checklist. Numbers below come from the saved notebooks and JSON result artifacts. The copied charts are existing experimental outputs; the flowchart is a process diagram, not a result plot. No screenshot is fabricated.

## Actual pipeline flowchart

![Actual project pipeline](actual_project_pipeline.png)

Vector version: [actual_project_pipeline.svg](actual_project_pipeline.svg).

## Dataset and repository links

- Call-centre simulation dataset source documented in Week 2: <https://www.kaggle.com/datasets/jpmiller/call-centre-queue-simulation>
- Bitext dataset used in Week 6: <https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset>
- Project repository: <https://github.com/BishwadeepRai/Smart-queue-Time-Management-system>

## Pipeline stages and purpose

### Week 2 — call-centre EDA only

1. **Dataset:** Read the existing Kaggle call-centre simulation CSV.
2. **Inspect:** Confirm dimensions, dtypes, missing values, duplicates, and parse timestamps for temporal analysis.
3. **EDA:** Explore `wait_length` as the stated prospective target, plus service duration and time patterns.
4. **Split strategy:** The notebook actually constructs a proposed chronological split at whole-day boundaries: train **31,898 rows**, validation **9,206**, and test **10,604**. This is exploratory split planning, not a model holdout experiment.
5. **Interpret:** Document findings and recommend chronological evaluation. **No Week 2 model is trained, and no test predictions or test metrics exist.** The later Week 4/5 binary classification task is not a fitted wait-time regression.

### Week 4 — feed-forward binary classifier

1. Read the shared CSV, combine date/start-time into a timestamp, and sort chronologically.
2. Inspect target balance and temporal/feature relationships.
3. Create the target from `meets_standard`; engineer eight queue-position/calendar features. Drop `wait_length` (it deterministically defines the label), `call_id`, and fields only known after service.
4. Split chronologically 70/15/15. Fit `StandardScaler` on training features only; transform validation and test.
5. Train four Keras dense-network settings with training class weights; choose using validation missed-class F1, then macro F1.
6. Predict held-out probabilities and threshold at 0.5.
7. Report classification metrics, positive-class ROC-AUC, confusion matrix, and the limitations evident in future-period performance.

### Week 5 — next-call sequence classifier

1. Read and validate the same CSV, parse and stably order call timestamps.
2. Engineer 12 prediction-time arrival/calendar features; exclude `wait_length`, identifiers, and post-arrival outcome fields.
3. Construct {week5["sequence_length"]}-call histories within each day; label each window with the following call's `meets_standard`. Result: **46,488 sequences**, tensor shape **(46,488, 20, 12)**.
4. Keep the final 20% of raw chronological rows as the future test targets; split development targets into train/validation in chronological order. Fit the scaler on **33,131 training-context rows only**.
5. Compare six PyTorch LSTM configurations and select by validation balanced accuracy, then accuracy/loss; train with weighted BCE.
6. Predict the future-period test set at threshold 0.5; compare against the training-majority baseline.
7. Report class-level and macro metrics, ROC-AUC, and confusion matrix. The saved result shows the LSTM did not beat the accuracy baseline.

### Week 6 — duplicate-aware multiclass intent routing

1. Load/cache the Bitext dataset; inspect **{week6["source_rows"]:,} rows x 5 selected fields** and **{week6["number_of_intents"]} intents**.
2. Check missing/empty values, class support and instruction lengths; normalize instructions by case-folding and collapsing whitespace.
3. Group normalized duplicates and verify label consistency; split **groups** into stratified 70/15/15 train/validation/test partitions, so normalized copies do not cross partitions. Fit the TF-IDF vocabulary on training text only. The token LSTM uses its existing training pipeline.
4. Train the existing TF-IDF MLP and token-sequence LSTM; select/checkpoint using validation macro F1.
5. Predict one of 27 intents on held-out requests.
6. Evaluate macro precision/recall/F1, one-vs-rest macro ROC-AUC, per-intent scores and confusion matrices.

### Purpose of the pipeline stages

| Stage | Purpose in this project |
|---|---|
| Dataset acquisition | Identify/load the actual source for that stage; Week 2/4/5 read the shared local CSV, while Week 6 loads or caches Bitext. |
| Inspection | Verify dimensions, schema, time/text validity, missingness, duplicates, class support and chronological order before modelling. |
| EDA | Describe target/feature distributions and project-specific patterns; it is the complete Week 2 deliverable, but only the diagnostic stage in Weeks 4–6. |
| Preprocessing | Transform raw values into prediction-time features or text tokens while excluding target leakage; handle repeated Week 6 requests before splitting. |
| Train/validation/test split | Isolate model fitting, model selection and final evaluation. Preserve chronology for the call-centre tasks and duplicate groups for Week 6. |
| Model training/selection | Fit the existing neural-network candidates and use validation criteria—not test performance—to select a configuration/checkpoint. |
| Predictions | Generate labels and/or class probabilities for held-out examples at the existing thresholds/class outputs. |
| Evaluation | Compare held-out predictions to true labels with the task-appropriate metrics, confusion matrices and ROC/AUC. |
| Interpretation | Explain what the evidence supports and its limits. In particular, Weeks 4/5 generalize poorly on future calls; Week 6's high scores are limited to its synthetic labeled corpus. |

## Intermediate evidence and actual outputs

### Dataset inspection and EDA

| Stage | Actual inspection / EDA output |
|---|---|
| Weeks 2/4/5 call-centre source | **51,708 rows x 9 source columns**; 2021-01-01 to 2021-12-31; 261 operating days. Week 2 notebook reports no missing values in the source fields and zero duplicate source rows. |
| Week 2 target and distributions | Stated target is continuous `wait_length`; mean is approximately **17 seconds**, median **0 seconds**, showing a zero-heavy/right-skewed distribution. This is not the binary target used later. |
| Week 2 proposed split | Whole-day chronological partition: train **31,898 (61.7%)**, validation **9,206 (17.8%)**, test **10,604 (20.5%)**. These are EDA/planning partitions only; Week 2 fits no model. |
| Weeks 4/5 classification target | `meets_standard`: **47,481 True (91.83%)**, **4,227 False (8.17%)**, about **11.23:1** imbalance. The identity `wait_length <= 60` exactly matches this label in all source rows. |
| Week 6 source | **{week6["source_rows"]:,} rows x 5 fields**, **{week6["number_of_intents"]} intents**; class counts range **{week6["class_count_min"]}–{week6["class_count_max"]}**. Missing/empty counts are zero for the five inspected fields. |
| Week 6 normalized duplicates | **{week6["unique_instruction_groups"]:,} unique normalized request groups**, **{week6["repeated_instruction_groups"]:,} repeated groups**, and **{week6["additional_repeated_instruction_rows"]:,} additional repeated rows**. The notebook checks that each normalized group has a single intent label. |

### Splits, preprocessing and model settings

| Stage | Split and preprocessing output |
|---|---|
| Week 4 | Chronological split: train **{week4["splits"]["train"]["n"]:,}** (missed {week4["splits"]["train"]["false"]:,}; {percent(week4["splits"]["train"]["false_pct"] / 100)}), validation **{week4["splits"]["val"]["n"]:,}** (missed {week4["splits"]["val"]["false"]:,}), test **{week4["splits"]["test"]["n"]:,}** (missed {week4["splits"]["test"]["false"]:,}; {percent(week4["splits"]["test"]["false_pct"] / 100)}). Eight features; train-fitted scaler; seed {week4["seed"]}. |
| Week 5 | **46,488 x 20 x 12** sequence tensor. Development has **29,540** training and **7,386** validation targets; future test has **9,562** targets. Training-only majority class is True (94.59%); future-test majority baseline accuracy is **{percent(w5m["train_majority_baseline_accuracy"])}**. |
| Week 6 | Group-stratified split: train **{week6["train_rows"]:,} rows / {week6["train_groups"]:,} groups**, validation **{week6["validation_rows"]:,} / {week6["validation_groups"]:,}**, test **{week6["test_rows"]:,} / {week6["test_groups"]:,}**. TF-IDF fit only on train; 27 classes represented in each partition. Seed {week6["seed"]}. |

| Model | Actual configuration / training evidence |
|---|---|
| Week 4 selected FFNN | Exp 2 — Increased capacity; dense widths **{w4_exp["architecture"]}**, learning rate **{w4_exp["learning_rate"]}**, batch **{w4_exp["batch_size"]}**, max epochs **{w4_exp["epochs"]}**, dropout **{w4_exp["dropout"]}**. Validation accuracy **{w4_exp["val_metrics"]["accuracy"]:.4f}**, missed-class F1 **{w4_exp["val_metrics"]["f1_false"]:.4f}**. Training history contains {len(w4_hist["loss"])} epochs; training loss {w4_hist["loss"][0]:.4f} → {w4_hist["loss"][-1]:.4f}; validation loss {w4_hist["val_loss"][0]:.4f} → {w4_hist["val_loss"][-1]:.4f}. |
| Week 5 selected LSTM | {w5_params["hidden_size"]} hidden units, {w5_params["num_layers"]} layer, learning rate {w5_params["learning_rate"]}, dropout {w5_params["dropout"]}, missed-class loss weight {w5_params["false_class_weight"]}; validation balanced accuracy **53.4976%**, missed-class recall **58.9474%**, and accuracy **49.3095%** versus the validation-majority baseline **88.4240%**. Selected checkpoint epoch 1. The notebook retains the multi-configuration loss curves. |
| Week 6 MLP | TF-IDF unigrams/bigrams, up to 12,000 terms; dense layers 256 and 128 with dropout 0.35/0.25. Training loss over {len(mlp_loss)} epochs: {mlp_loss[0]:.4f} → {mlp_loss[-1]:.4f}; best validation macro F1 **{max(mlp_val):.4f}**. |
| Week 6 LSTM | Token LSTM with 64-dimensional embedding, 64 hidden units and dropout 0.30. Training loss over {len(lstm_loss)} epochs: {lstm_loss[0]:.4f} → {lstm_loss[-1]:.4f}; best validation macro F1 **{max(lstm_val):.4f}**. |

### Held-out results

Metrics are reported per task; scores across these different datasets/targets are **not a direct model comparison**. Binary AUC uses `True`/met-standard as the positive class; Week 6 uses one-vs-rest macro AUC.

**ROC plot convention:** Week 6's saved ROC figure draws micro-averaged multiclass curves. Its numeric `roc_auc_ovr_macro` in `week 6/outputs/week6_results.json` is a separate macro one-vs-rest summary. Label these distinctly in the report; do not call the plotted micro curve macro-averaged.

| Task/model | Accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC | Important interpretation |
|---|---:|---:|---:|---:|---:|---|
| Week 4 selected FFNN | {w4m["accuracy"]:.4f} | {w4m["precision_macro"]:.4f} | {w4m["recall_macro"]:.4f} | {w4m["f1_macro"]:.4f} | {w4m["roc_auc_positive_meets_standard"]:.5f} | Missed recall {w4m["recall_false"]:.4f}; test missed-class prevalence increases from {percent(week4["splits"]["train"]["false_pct"] / 100)} in train to {percent(week4["splits"]["test"]["false_pct"] / 100)} in test. |
| Week 5 selected LSTM | {w5m["accuracy"]:.4f} | {w5m["macro_avg"]["precision"]:.4f} | {w5m["macro_avg"]["recall"]:.4f} | {w5m["macro_avg"]["f1-score"]:.4f} | {w5m["roc_auc"]:.5f} | Below the {percent(w5m["train_majority_baseline_accuracy"])} training-majority test accuracy; missed-standard recall {w5m["missed_standard"]["recall"]:.4f}. |
| Week 6 TF-IDF MLP | {w6_mlp["accuracy"]:.4f} | {w6_mlp["precision_macro"]:.4f} | {w6_mlp["recall_macro"]:.4f} | {w6_mlp["f1_macro"]:.4f} | {w6_mlp["roc_auc_ovr_macro"]:.5f} | High score on this synthetic/automatically labeled dataset; external validity is not established. |
| Week 6 PyTorch LSTM | {w6_lstm["accuracy"]:.4f} | {w6_lstm["precision_macro"]:.4f} | {w6_lstm["recall_macro"]:.4f} | {w6_lstm["f1_macro"]:.4f} | {w6_lstm["roc_auc_ovr_macro"]:.5f} | Evaluated on the same duplicate-aware holdout. |

### Confusion matrices (rows = actual; columns = predicted)

- Week 4 labels `[False, True]`: `{w4m["confusion_matrix"]}`.
- Week 5 labels `[False (missed), True (met)]`: `{week5["confusion_matrix"]}`.
- Week 6: use the saved per-model 27-class matrices in `week 6/outputs/week6_results.json` and the rendered figure below.

## Existing project visuals copied into this folder

These PNGs are copied from the weekly output folders without recreating or altering their experimental data:

| Asset | Evidence to show |
|---|---|
| [week2_wait_length_distribution.png](week2_wait_length_distribution.png) | Week 2 continuous waiting-time target distribution. |
| [week2_temporal_eda.png](week2_temporal_eda.png) | Week 2 temporal exploration. |
| [week2_chronological_ordering_finding.png](week2_chronological_ordering_finding.png) | Week 2 recorded warning/finding about chronology and random splitting. |
| [week4_target_class_distribution.png](week4_target_class_distribution.png) | Week 4 binary class imbalance. |
| [week4_daily_caller_by_target.png](week4_daily_caller_by_target.png) | Project-specific queue-position relationship. |
| [week4_selected_model_training_history.png](week4_selected_model_training_history.png) | Week 4 selected FFNN history. |
| [week4_test_confusion_matrix.png](week4_test_confusion_matrix.png) | Week 4 held-out confusion matrix. |
| [week4_test_roc_curve.png](week4_test_roc_curve.png) | Week 4 held-out ROC curve and AUC. |
| [week5_lstm_training_history.png](week5_lstm_training_history.png) | Week 5 configuration training/validation loss. |
| [week5_test_accuracy_and_confusion_matrix.png](week5_test_accuracy_and_confusion_matrix.png) | Week 5 future test versus baseline and confusion matrix. |
| [week5_test_roc_curve.png](week5_test_roc_curve.png) | Week 5 future-test ROC curve and AUC. |
| [week6_intent_distribution_and_duplicates.png](week6_intent_distribution_and_duplicates.png) | Week 6 class support and duplicate-request EDA. |
| [week6_test_confusion_matrices.png](week6_test_confusion_matrices.png) | Week 6 held-out confusion matrices for MLP and LSTM. |
| [week6_test_multiclass_roc_curves.png](week6_test_multiclass_roc_curves.png) | Week 6 micro-averaged multiclass ROC curves; the saved numeric AUC is macro one-vs-rest. |
| [week6_training_history.png](week6_training_history.png) | Week 6 saved model training histories. |

## Important code snippets to include (short excerpts only)

Use small snippets in the final report and cite the linked source file/cell. Do not paste the full notebook.

### 1. Week 4 chronological split, training-only scaling and model selection

Source: [`week 4/scripts/run_experiments.py`](../week%204/scripts/run_experiments.py).

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURE_COLS])
X_val = scaler.transform(val[FEATURE_COLS])
X_test = scaler.transform(test[FEATURE_COLS])

best = max(experiment_results,
           key=lambda r: (r["val_metrics"]["f1_false"],
                          r["val_metrics"]["f1_macro"]))
```

**Why include it:** shows leakage-safe scaler fitting and that hyperparameter/model selection is performed on validation rather than the held-out test set.

### 2. Week 5 within-day sequence construction

Source: [`week 5/week5_rnn_lstm.ipynb`](../week%205/week5_rnn_lstm.ipynb), sequence preparation cells.

```python
SEQUENCE_LENGTH = 20
window_positions_list, target_positions_list = [], []
for day_positions in raw_df.groupby("date", sort=False).indices.values():
    day_positions = np.asarray(day_positions, dtype=np.int64)
    for next_offset in range(SEQUENCE_LENGTH, len(day_positions)):
        window_positions_list.append(
            day_positions[next_offset - SEQUENCE_LENGTH:next_offset])
        target_positions_list.append(day_positions[next_offset])
```

**Why include it:** explains what one supervised sample means and why sequence boundaries preserve the project's event chronology. Pair it with the actual tensor shape (46,488, 20, 12).

### 3. Week 6 normalized duplicate groups and grouped stratified split

Source: [`week 6/week6_customer_support_intent_classification.ipynb`](../week%206/week6_customer_support_intent_classification.ipynb), split/preprocessing cells.

```python
normalized_instructions = (
    df["instruction"].astype(str).str.casefold()
    .str.replace(r"\\s+", " ", regex=True).str.strip()
)
instruction_groups, unique_instructions = pd.factorize(
    normalized_instructions, sort=False)
if (group_summary["label_count"] != 1).any():
    raise ValueError("A normalized customer instruction maps to multiple intent labels.")
```

**Why include it:** demonstrates the control for exact case/whitespace duplicates that would otherwise inflate held-out performance.

### 4. Week 5 training-only scaling and Week 6 group split

```python
training_feature_positions = np.unique(window_positions[train_idx].reshape(-1))
scaler.fit(feature_matrix[training_feature_positions])

train_groups, holdout_groups = train_test_split(
    all_group_ids, test_size=0.30, random_state=SEED, stratify=group_y)
```

**Why include it:** documents the Week 5 train-context-only scaler and the Week 6 stratified group-level split. In the report, clarify these are snippets from separate notebooks.

### 5. Week 6 train-only TF-IDF vocabulary

```python
X_train_tfidf = vectorizer.fit_transform(texts[train_idx])
X_valid_tfidf = vectorizer.transform(texts[valid_idx])
X_test_tfidf = vectorizer.transform(texts[test_idx])
```

**Why include it:** shows that the vocabulary/statistics are learned from training text only and reused for validation/test.

### 6. Final held-out metrics and ROC

Source: Week 4 experiment runner, Week 5 final evaluation notebook cell, and Week 6 evaluation cells/results JSON.

Include a compact metric computation/evaluation excerpt for the specific task, then show its actual confusion-matrix and ROC figures. For Week 6 describe ROC as one-vs-rest macro AUC; do not present it as binary ROC.

## Recommended screenshots to capture from the real executed project

Screenshots below are **to be captured by the student from the actual notebook/terminal**. They are recommendations, not generated evidence.

| Suggested filename | Capture in the project | What the screenshot should show |
|---|---|---|
| `screenshot_week2_dataset_audit.png` | Week 2 executed notebook, load/quality/date cells | Shape 51,708 x 9, parsed date range, zero missing values and zero duplicate rows. |
| `screenshot_week2_wait_length_eda.png` | Week 2 executed notebook, target distribution/temporal EDA | `wait_length` distribution and a temporal chart; label it as EDA, not model evaluation. |
| `screenshot_week2_proposed_chronological_split.png` | Week 2 split-strategy cell | Day-level proposed train/validation/test row counts; make clear no model was trained in Week 2. |
| `screenshot_week4_preprocessing_split.png` | Week 4 runner/notebook code and split output | Target-leak exclusions, chronological split, train-only scaler fit and 70/15/15 counts. |
| `screenshot_week4_selected_model_history.png` | Week 4 experiment output | Selected Exp 2 configuration and training/validation history. |
| `screenshot_week4_test_evaluation.png` | Week 4 saved final output/figures | Test precision/recall/F1, AUC 0.5404 and matching confusion/ROC figures. |
| `screenshot_week5_sequence_pipeline.png` | Week 5 executed notebook sequence and split cells | (46,488, 20, 12) tensor, within-day windowing, scaler fit rows 33,131 and future-test split. |
| `screenshot_week5_model_selection_history.png` | Week 5 notebook selected-experiment output | Selected 32-unit LSTM, weight 8, validation selection/checkpoint and loss curves. |
| `screenshot_week5_future_test_results.png` | Week 5 final evaluation cells | 74.34% vs 84.06% baseline, missed-class metrics, confusion matrix and AUC 0.4873. |
| `screenshot_week6_duplicate_aware_split.png` | Week 6 executed notebook EDA and split cells | 1,187 duplicate groups, grouped 70/15/15 counts and proof groups are disjoint. |
| `screenshot_week6_model_results.png` | Week 6 notebook held-out evaluation | MLP/LSTM metrics and per-intent report. |
| `screenshot_week6_confusion_roc.png` | Week 6 saved evaluation figures | Both 27-class confusion matrices and micro-averaged ROC curves; distinguish the curve from macro one-vs-rest AUC in the JSON. |
| `screenshot_project_repository_files.png` | VS Code Explorer or GitHub repository page | Actual week folders, report, outputs, and project link. |

Avoid screenshots that hide code context, crop metric labels, or show unexecuted cells as results. Keep raw identifiers/customer-like text out of screenshots where unnecessary.

## Evidence source files

- [`week 4/results.json`](../week%204/results.json)
- [`week 5/outputs/week5_results.json`](../week%205/outputs/week5_results.json)
- [`week 6/outputs/week6_results.json`](../week%206/outputs/week6_results.json)
- Executed notebooks and original image folders under `week 2/`, `week 4/`, `week 5/`, and `week 6/`.

Regenerate copied figures and the pipeline diagram from the repository root:

```powershell
python "report_assets/build_report_assets.py"
```
"""
    (ASSET_DIR / "technical_evidence.md").write_text(content, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    week4, week5, week6 = load_results()
    copy_project_figures()
    draw_flowchart()
    build_evidence_markdown(week4, week5, week6)
    print(f"Saved flowchart, {len(FIGURES)} existing project figures, and evidence guide in {ASSET_DIR}")


if __name__ == "__main__":
    main()
