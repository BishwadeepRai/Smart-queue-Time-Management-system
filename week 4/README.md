# Week 4: Feed-Forward Neural Network Classification

This assignment predicts whether an incoming call meets the 60-second service standard using a TensorFlow/Keras feed-forward neural network.

## Submission files

- [`week4_neural_network.ipynb`](week4_neural_network.ipynb) — executed assignment notebook.
- [`images/`](images/) — saved exploratory, training, and evaluation figures.
- [`results.json`](results.json) — metrics used to document the recorded experiment run.
- [`scripts/`](scripts/) — experiment runner and shared model utilities used to regenerate the saved figures and metrics.
- [`images/fig10_roc_curve.png`](images/fig10_roc_curve.png) — held-out ROC curve for the selected model.

The notebook uses the shared dataset at [`../data/simulated_call_centre.csv`](../data/simulated_call_centre.csv). Install the shared dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

From the repository root, regenerate the saved experiment results and figures with:

```powershell
python "week 4/scripts/run_experiments.py"
```

Experiment output is saved in this folder: figures in `images/` and metrics in `results.json`. The notebook itself contains its executed outputs.

## Recorded evaluation

The current saved run selected **Exp 2 — Increased capacity** using validation
missed-standard F1, with validation macro F1 as the tie-breaker. Its
chronological test-set results are accuracy 0.3753, macro precision 0.5159,
macro recall 0.5246, macro F1 0.3635, and ROC-AUC 0.5404 for the positive
`meets_standard=True` class. Missed-standard recall is 0.7446. The confusion
matrix and complete per-class metrics are in `results.json`.

These results indicate weak future-period generalization (the positive-class
ROC-AUC is close to chance), despite the selected model's missed-class recall.
The dataset's class prevalence changes over time and is simulated; this model
should not be represented as an operationally validated service predictor.
