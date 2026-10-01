# Week 4 — Neural Network Classifier for Call-Centre Service Standard

## AI-Based Smart Queue & Time Management System

Feed-forward neural network (TensorFlow/Keras) that predicts whether an arriving simulated call will meet a 60-second answer standard (`meets_standard`). This is the Week 4 classifier assignment. Later project work may use RNN/LSTM for waiting-time forecasting; that is out of scope here.

---

## Dataset

| Item | Detail |
|------|--------|
| File | `data/simulated_call_centre.csv` |
| Source | [Call Centre Queue Simulation](https://www.kaggle.com/) on Kaggle (Week 2 dataset) |
| License | **CC BY-SA 4.0** (redistribution allowed with attribution and share-alike) |
| Rows × columns | **51,708 × 9** (verified from the file) |
| Period | 2021-01-01 to 2021-12-31 (261 weekdays) |

| Column | Used? |
|--------|--------|
| `call_id` | No (identifier) |
| `date`, `call_started`, `daily_caller` | Yes (context at arrival) |
| `call_answered`, `call_ended`, `wait_length`, `service_length` | No (post-event / leakage) |
| `meets_standard` | Target |

`meets_standard` is exactly `wait_length <= 60` (0 disagreements in 51,708 rows). `wait_length` is therefore **not** an input.

---

## Objective

Binary classification:

- `True` — call met the service standard  
- `False` — call missed the service standard  

---

## Features

`daily_caller`, cyclical hour / weekday / month (`sin`/`cos`), `day_of_month`.

---

## Model

- Type: feed-forward MLP  
- Output: sigmoid  
- Loss: binary cross-entropy  
- Optimiser: Adam  
- Class weights: balanced on the **training** split only  
- Split: chronological 70% / 15% / 15% by `call_started` (avoids mixing December congestion into training)

---

## Experiments (validation)

| Experiment | Architecture | LR | Batch | Epochs | Dropout | Accuracy | F1 (False) |
|------------|--------------|----|-------|--------|---------|----------|------------|
| 1 Baseline | [32] | 0.001 | 32 | 20 | — | 0.2617 | 0.2365 |
| 2 Capacity | [64, 32] | 0.001 | 32 | 30 | — | 0.3394 | 0.2199 |
| **3 Lower LR** | [64, 32] | 0.0005 | 32 | 30 | — | **0.4116** | **0.2391** |
| 4 Dropout | [64, 32] | 0.0005 | 32 | 30 | 0.3 | 0.3124 | 0.2334 |

Selected: **Experiment 3** (highest validation F1 on the minority class).

### Test set (untouched; Exp 3)

| Metric | False (not met) | True (met) |
|--------|-----------------|------------|
| Precision | 0.1607 | 0.8399 |
| Recall | 0.6450 | 0.3560 |
| F1 | 0.2573 | 0.5000 |
| Accuracy | 0.4023 | |
| Macro F1 | 0.3786 | |

Confusion matrix: TN 803, FP 442, FN 4,194, TP 2,318 (n = 7,757).

Accuracy is **not** treated as success. A constant “True” rule would score 83.95% on this test window.

---

## Class imbalance

| Class | Count | % |
|-------|------:|--:|
| True (met) | 47,481 | 91.83 |
| False (not met) | 4,227 | 8.17 |

Ratio 11.23 : 1. Balanced class weights; no SMOTE.

---

## Repository layout

```
.
├── data/
│   └── simulated_call_centre.csv
├── notebooks/
│   └── week4_neural_network.ipynb
├── docs/
│   ├── neural_network_report.md
│   └── figures/
│       ├── fig1_class_distribution.png
│       ├── fig2_daily_caller_by_target.png
│       ├── fig3_daily_caller_boxplot.png
│       ├── fig4_hour_meet_rate.png
│       ├── fig5_weekday_meet_rate.png
│       ├── fig6_month_volume_rate.png
│       ├── fig7_correlation_heatmap.png
│       ├── fig8_confusion_matrix.png
│       ├── fig9_experiment_comparison.png
│       └── fig_train_exp*.png
├── ml_experiments/
│   ├── nn_classifier.py
│   └── run_experiments.py
├── results.json
├── README.md
├── requirements.txt
└── .gitignore
```

Week 2 EDA notebooks remain at the project root for continuity with the same dataset.

---

## How to run

```bash
pip install -r requirements.txt
python ml_experiments/run_experiments.py
jupyter notebook notebooks/week4_neural_network.ipynb
```

Run the notebook **from top to bottom**. Figures are also written to `docs/figures/`.

---

## Publishing to GitHub

Create an empty GitHub repository, then from this project folder:

```bash
git init
git add .
git commit -m "Week 4 neural network implementation"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not invent a URL. Use the address GitHub shows after you create the repo. Put that link on the assignment submission form.

If you already initialised git in this folder, skip `git init` and only add the remote.

---

## Dataset attribution

Simulated call-centre queue data originally published on Kaggle for business/operations analytics teaching, generated with R `simmer` (four agents, weekday 08:00–18:00, ~5-minute mean service, 60-second standard). License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

---

## Course

Bishwadeep Rai — Westcliff University — TECH 405: Artificial Neural Network and Deep Learning — Week 4.
