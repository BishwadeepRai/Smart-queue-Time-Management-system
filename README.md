<div align="center">

# AI-Based Smart Queue & Time Management System

**TECH 405 | Call Centre Queue Simulation | Sequential Service-Standard Classification**

![Course](https://img.shields.io/badge/Course-TECH%20405-28536B)
![Dataset](https://img.shields.io/badge/Data-Simulated%20Call%20Centre-3A7D78)
![Week 5 model](https://img.shields.io/badge/Week%205-PyTorch%20LSTM-C75C3A)

</div>

## Project Overview

This project studies simulated call-centre queue behavior and whether a call meets the required service standard. Week 5 treats calls as chronological queue events and builds a next-call classifier from recent call history using a real PyTorch LSTM.

The Week 5 model predicts `meets_standard`:

- `True` (1): the call met the standard
- `False` (0): the call missed the standard

The Week 5 deliverables are grouped in [`week 5/`](week%205/), including the [LSTM notebook](week%205/week5_rnn_lstm.ipynb) and a [Week 5 overview](week%205/README.md). The earlier Week 4 feed-forward classifier remains in [`notebooks/week4_neural_network.ipynb`](notebooks/week4_neural_network.ipynb).

## Dataset

- **Dataset:** Call Centre Queue Simulation
- **File:** [`data/simulated_call_centre.csv`](data/simulated_call_centre.csv)
- **Rows and columns:** 51,708 rows × 9 columns
- **Date range:** 2021-01-01 to 2021-12-31, across 261 operating dates
- **Target counts:** 47,481 met standard; 4,227 missed standard
- **Target:** `meets_standard`

The CSV contains ordered call events, not pre-built sequences. The notebook creates overlapping windows of 20 previous calls from the same operating date and uses the following call's target as the label. Windows stay within a day because the simulated queue resets between operating dates.

### Leakage controls

Inspection confirmed that `wait_length <= 60` exactly determines the target in this dataset. `wait_length` is therefore excluded. The model also excludes `call_id`, `call_answered`, `call_ended`, and `service_length` because they are an identifier or information unavailable at call arrival. Inputs use `daily_caller` and time/calendar features derived from the historical calls' start timestamps.

## Week 5 Model

- **Architecture:** `nn.LSTM → nn.Linear`, trained with `BCEWithLogitsLoss`
- **Input:** 20 chronological prior-call feature vectors
- **Output:** binary prediction for the next call
- **Split:** first 80% of ordered raw call rows for development; final 20% as future test data
- **Model selection:** chronological validation data from the development period only
- **Scaling:** fitted on training-window features only
- **Experiments:** hidden size, learning rate, dropout, and class-weight configurations
- **Device:** CUDA when available, otherwise CPU

## Results

On the current fixed split, validation balanced-accuracy selection chose the 32-unit, one-layer LSTM with learning rate 0.001 and false-class loss weight 8.

| Test measure | Result |
|---|---:|
| Training-majority baseline accuracy | 84.06% |
| Selected LSTM accuracy | 74.34% |
| Difference from baseline | -9.73 percentage points |
| Missed-standard recall | 12.93% |

**Interpretation:** the selected LSTM did not beat the majority baseline. It detects some missed-standard calls, but minority-class recall remains low. The dataset is simulated, and these results do not establish that the model generalizes to a real call centre. No official target accuracy was provided for this custom dataset.

The notebook contains the training/validation loss plots, confusion matrix, validation comparison, and four labeled screenshot locations.

## Repository Layout

```text
.
├── data/
│   └── simulated_call_centre.csv
├── docs/
│   └── figures/
├── ml_experiments/
├── notebooks/
│   └── week4_neural_network.ipynb
├── week 2/
├── week 4/
├── week 5/
│   ├── README.md
│   └── week5_rnn_lstm.ipynb
├── README.md
└── requirements.txt
```

## Run Week 5

From the project root:

```powershell
python -m pip install -r requirements.txt
jupyter notebook "week 5/week5_rnn_lstm.ipynb"
```

The notebook selects CUDA automatically when the installed PyTorch build and driver support it. For an NVIDIA GPU, choose the matching build with the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Week 5 figures, training output, evaluation metrics, and the editable report-ready summary are included in the notebook.

## Dataset Redistribution

The dataset is the existing simulation file in this project; no external example dataset is substituted. The project files do not include a separate license or redistribution statement. Confirm permission before publishing the CSV in a public repository; if permission is unclear, do not distribute the raw data.

## Course Context

- **Course:** TECH 405 — Artificial Neural Network and Deep Learning
- **Project:** AI-Based Smart Queue & Time Management System
- **Week 4:** Feed-forward neural-network classification
- **Week 5:** Chronological RNN/LSTM sequence classification
