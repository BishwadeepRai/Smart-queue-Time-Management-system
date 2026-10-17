# Week 4: Feed-Forward Neural Network Classification

This assignment predicts whether an incoming call meets the 60-second service standard using a TensorFlow/Keras feed-forward neural network.

## Submission files

- [`week4_neural_network.ipynb`](week4_neural_network.ipynb) — executed assignment notebook.
- [`images/`](images/) — saved exploratory, training, and evaluation figures.
- [`results.json`](results.json) — metrics used to document the recorded experiment run.
- [`scripts/`](scripts/) — experiment runner and shared model utilities used to regenerate the saved figures and metrics.

The notebook uses the shared dataset at [`../data/simulated_call_centre.csv`](../data/simulated_call_centre.csv). Install the shared dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

From the repository root, regenerate the saved experiment results and figures with:

```powershell
python "week 4/scripts/run_experiments.py"
```

Experiment output is saved in this folder: figures in `images/` and metrics in `results.json`. The notebook itself contains its executed outputs.
