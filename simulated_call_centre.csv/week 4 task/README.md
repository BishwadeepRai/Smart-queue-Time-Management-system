# Call Centre Neural Network Classifier
## Week 4 Machine Learning Assignment
### AI-Based Smart Queue & Time Management System

---

## Project Description

A feed-forward neural network classifier built with **TensorFlow/Keras** to predict whether an incoming call to a simulated call centre will meet the 60-second service standard (`meets_standard`).

This assignment is part of a larger AI-Based Smart Queue and Time Management System project. Future phases may incorporate RNN/LSTM models to predict continuous queue behaviour; this phase focuses on binary classification.

---

## Dataset

**Source:** Simulated Call Centre Queue dataset (`simulated_call_centre.csv`)  
**Rows:** 51,708  
**Columns:** 9  
**Period:** 2021-01-01 to 2021-12-31  

| Column | Description |
|--------|-------------|
| call_id | Unique call identifier |
| date | Date of the call (YYYY-MM-DD) |
| daily_caller | Caller's queue position on that day |
| call_started | Time call entered the queue |
| call_answered | Time an agent answered (post-event) |
| call_ended | Time call ended (post-event) |
| wait_length | Seconds waited before answer (post-event) |
| service_length | Seconds of service (post-event) |
| meets_standard | **Target** — True if wait ≤ 60 s |

> **Data Leakage Note:** `wait_length`, `service_length`, `call_answered`, and `call_ended` are post-event measurements excluded from model inputs to prevent data leakage.

---

## Objective

Binary classification:  
- `True` → Call waited ≤ 60 seconds (meets standard)  
- `False` → Call waited > 60 seconds (does not meet standard)

---

## Features Used

| Feature | Derivation | Reason |
|---------|------------|--------|
| `daily_caller` | Direct | Queue position at arrival time |
| `hour_sin`, `hour_cos` | Cyclical (24h) | Time-of-day pattern |
| `dow_sin`, `dow_cos` | Cyclical (7d) | Day-of-week pattern |
| `month_sin`, `month_cos` | Cyclical (12m) | Seasonal pattern |
| `day_of_month` | Extracted | Mid-month patterns |

---

## Model

**Type:** Feed-forward Neural Network (Binary Classifier)  
**Framework:** TensorFlow / Keras  
**Architecture:** Dense layers with ReLU activations + Sigmoid output  
**Loss:** Binary Crossentropy  
**Optimiser:** Adam  

---

## Hyperparameter Experiments

| Experiment | Architecture | LR | Batch | Epochs | Dropout |
|------------|-------------|----|-------|--------|---------|
| 1 — Baseline | [32] | 0.001 | 32 | 20 | — |
| 2 — Deeper | [64, 32] | 0.001 | 32 | 30 | — |
| 3 — Lower LR | [64, 32] | 0.0005 | 32 | 40 | — |
| 4 — Dropout | [64, 32] | 0.0005 | 64 | 40 | 0.3 |

---

## Class Imbalance

| Class | Count | % |
|-------|-------|---|
| True (Met) | 47,481 | 91.83% |
| False (Not Met) | 4,227 | 8.17% |

**Handling:** Balanced class weights (`class_weight='balanced'`) applied during training. Stratified splitting used.

---

## Evaluation Metrics

- Confusion Matrix
- Precision (per class)
- Recall (per class)
- F1-Score (per class, macro, weighted)
- Accuracy (reported but not sole metric)

---

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd call-centre-neural-network
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Copy `simulated_call_centre.csv` into the `data/` folder.

### 4. Open the notebook

```bash
jupyter notebook notebooks/week4_neural_network.ipynb
```

Or open in VS Code / JupyterLab and run all cells from top to bottom.

---

## Required Libraries

See `requirements.txt`. Key dependencies:

- Python 3.9+
- TensorFlow 2.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Repository Structure

```
call-centre-neural-network/
|
|-- data/
|   +-- simulated_call_centre.csv        <- Dataset (place here)
|
|-- notebooks/
|   +-- week4_neural_network.ipynb       <- Main notebook
|
|-- README.md
|-- requirements.txt
+-- .gitignore
```

---

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Week 4: Neural network classifier for call centre service standard prediction"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

---

## Dataset Attribution

This dataset is a simulated call centre queue dataset generated for the AI-Based Smart Queue & Time Management System university project.  
If using or sharing this dataset, please attribute it appropriately.

---

## Key Findings

1. **Data leakage** — `wait_length` is directly derived from `meets_standard` (threshold = 60 s). Excluded to prevent leakage.
2. **Class imbalance** (11.2:1 ratio) makes accuracy misleading; F1-score for the False class is the primary metric.
3. **Time patterns** — Hour 13 and Friday show the lowest service standard rates due to congestion.
4. **December** is the highest-volume and lowest-standard month.
5. **Cyclical encoding** of hour, day-of-week, and month improves neural-network learning of periodic patterns.
