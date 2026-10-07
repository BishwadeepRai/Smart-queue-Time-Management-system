# Week 5 — Sequential Call-Centre Classification

**Course:** TECH 405 — Artificial Neural Network and Deep Learning
**Project:** AI-Based Smart Queue & Time Management System
**Model:** PyTorch LSTM for next-call service-standard classification

## Project overview

This week's experiment uses ordered call-centre events to predict whether the next call meets the service standard. The target is `meets_standard` (`True` = met the standard; `False` = missed it).

The dataset is shared with the rest of the project at [`../data/simulated_call_centre.csv`](../data/simulated_call_centre.csv). It contains 51,708 rows and nine source columns, covering 2021-01-01 through 2021-12-31 across 261 operating dates.

## Method

- Construct overlapping sequences from the previous 20 calls within the same operating date; use the following call's target as the label.
- Use an `nn.LSTM`, dropout, and a linear output layer trained with `BCEWithLogitsLoss`; no pretrained model is used.
- Split chronologically: the first 80% of raw call rows form the development period, while the final 20% are reserved for the future test period.
- Fit scaling on training-window features only and use validation data for model selection.
- Exclude `wait_length`, which directly determines the target, as well as the call identifier and post-arrival fields.

The full workflow, plots, experiment comparison, confusion matrix, and editable report summary are in [`week5_rnn_lstm.ipynb`](week5_rnn_lstm.ipynb).

## Results

Validation balanced-accuracy selection chose a 32-unit, one-layer LSTM with learning rate 0.001 and missed-class loss weight 8.

| Test measure | Result |
|---|---:|
| Training-majority baseline accuracy | 84.06% |
| Selected LSTM accuracy | 74.34% |
| Difference from baseline | -9.73 percentage points |
| Missed-standard recall | 12.93% |
| Missed-standard F1 | 13.83% |

The selected LSTM did not outperform the majority baseline. The dataset is simulated, no official target-accuracy benchmark was provided, and this course-scale experiment is not a validated operational predictor.

## Run the notebook

From the repository root:

```powershell
python -m pip install -r requirements.txt
jupyter notebook "week 5/week5_rnn_lstm.ipynb"
```

The notebook uses the shared dataset in `data/` and selects CUDA when supported, otherwise CPU. Install the PyTorch build appropriate for the available hardware.
