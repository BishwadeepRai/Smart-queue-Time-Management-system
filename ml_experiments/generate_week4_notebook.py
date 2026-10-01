"""Generate the Week 4 assignment notebook with code and result-backed markdown."""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Week 4 — Neural Network Classifier

## AI-Based Smart Queue and Time Management System

**Bishwadeep Rai**  
Westcliff University, California  
TECH 405: Artificial Neural Network and Deep Learning  
Professor Aashish Dhakal

---

## 1. Introduction

This notebook implements a **feed-forward neural network classifier** for the Week 4 assignment. The larger project may later use RNN/LSTM models to forecast waiting times. This week’s requirement is different: train a neural-network **classifier**, adjust hyperparameters, and evaluate it with a confusion matrix, precision, recall, and F-score.

**Prediction task.** When a call arrives, will it meet the service standard (`meets_standard`)?

A call meets the standard if the caller waits at most 60 seconds. That 60-second rule is confirmed from the uploaded CSV (see the leakage check below). It is **not** used as a model input.

**Framework:** TensorFlow / Keras  
**Random seed:** 42

The numbers printed after each experiment in this notebook come from running the cells. The markdown interpretations refer to the recorded seed-42 run of `ml_experiments/run_experiments.py` on this dataset so the written report stays consistent with the saved figures in `docs/figures/`.
""")

md("""## 2. Dataset and problem definition

**File:** `simulated_call_centre.csv`  
**Source:** Kaggle *Call Centre Queue Simulation* (CC BY-SA 4.0)  
**Simulation setting (from the dataset description used in Week 2):** four agents, weekdays 08:00–18:00, mean service about five minutes, 60-second answer standard.

| Column | Role |
|--------|------|
| `call_id` | Identifier — excluded |
| `date` | Arrival date — time features extracted |
| `daily_caller` | Queue position that day — used |
| `call_started` | Arrival time — time features extracted |
| `call_answered` | Post-event — excluded |
| `call_ended` | Post-event — excluded |
| `wait_length` | Post-event and target constructor — excluded |
| `service_length` | Post-event — excluded |
| `meets_standard` | **Target** |

This is binary classification:

- `True` = call met the 60-second standard  
- `False` = call did not meet the standard
""")

md("""---
## Part 1 — Data understanding

> **Screenshot 1 — Dataset loading and initial inspection**  
> Capture the next few executed cells: shape, dtypes, head, sample, quality checks, and datetime parsing.
""")

code("""import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

print('TensorFlow:', tf.__version__)
print('Seed     :', SEED)""")

code("""# Resolve the CSV whether the notebook is opened from notebooks/ or the project root
CANDIDATES = [
    os.path.join('..', 'data', 'simulated_call_centre.csv'),
    os.path.join('data', 'simulated_call_centre.csv'),
    os.path.join('..', 'simulated_call_centre.csv'),
]
CSV_PATH = next(p for p in CANDIDATES if os.path.exists(p))
print('Loading:', os.path.abspath(CSV_PATH))

df_raw = pd.read_csv(CSV_PATH)
print('\\nShape:', df_raw.shape)
print('\\nColumns and dtypes:')
print(df_raw.dtypes)
print('\\nFirst 5 rows:')
display(df_raw.head())""")

code("""print('Random 5 rows:')
display(df_raw.sample(5, random_state=SEED))
print('\\nDescriptive statistics:')
display(df_raw.describe(include='all'))""")

code("""print('Missing values:\\n', df_raw.isnull().sum())
print('\\nDuplicate rows:', int(df_raw.duplicated().sum()))
print('Negative wait_length:', int((df_raw['wait_length'] < 0).sum()))
print('service_length <= 0 :', int((df_raw['service_length'] <= 0).sum()))
print('meets_standard unique:', df_raw['meets_standard'].unique())
print('call_id unique == rows:', df_raw['call_id'].nunique() == len(df_raw))""")

code("""# Date/time columns are not ordinary strings.
# date is YYYY-MM-DD; call times are 12-hour clock strings such as '8:00:00 AM'.
df = df_raw.copy()
df['date'] = pd.to_datetime(df['date'])

def combine_date_time(date_col, time_str_col):
    t = pd.to_datetime(time_str_col, format='%I:%M:%S %p', errors='coerce')
    return date_col + pd.to_timedelta(
        t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second, unit='s')

df['call_started_dt']  = combine_date_time(df['date'], df['call_started'])
df['call_answered_dt'] = combine_date_time(df['date'], df['call_answered'])
df['call_ended_dt']    = combine_date_time(df['date'], df['call_ended'])

print('call_started range :', df['call_started_dt'].min(), '->', df['call_started_dt'].max())
print('Unique days        :', df['date'].nunique())
print('NaT in call_started:', int(df['call_started_dt'].isnull().sum()))
print('Weekdays present   :', sorted(df['date'].dt.dayofweek.unique().tolist()))""")

md("""**Data-quality notes from the uploaded file**

- Shape is **51,708 rows × 9 columns**.
- No missing values and no duplicate rows.
- `meets_standard` is a boolean.
- **88** rows have `service_length == 0`. These are kept. Zero-length service can occur in a simulation (immediate hang-up / zero talk time) and that column is not used as a feature, so dropping them would discard valid arrival events without improving the classifier.
""")

md("""---
## Part 2 — Exploratory analysis for the classifier

### Data leakage check

If `meets_standard` is a hard threshold on `wait_length`, then using `wait_length` as a feature is not prediction. It is copying the label rule.
""")

code("""true_max = df.loc[df['meets_standard'] == True, 'wait_length'].max()
false_min = df.loc[df['meets_standard'] == False, 'wait_length'].min()
mismatches = int(((df['wait_length'] <= 60) != df['meets_standard']).sum())

print('Max wait_length among True  :', true_max)
print('Min wait_length among False :', false_min)
print('Rows where (wait_length <= 60) disagrees with meets_standard:', mismatches)
print()
print('Decision: exclude wait_length, call_answered, call_ended, and service_length.')""")

md("""Recorded result on this CSV: max wait among True is **60**, min wait among False is **61**, mismatches = **0**.  
So `meets_standard` is exactly `wait_length <= 60`.

**Availability at prediction time** (new call just arriving):

| Column | Available? | Decision |
|--------|------------|----------|
| `call_id` | Identifier only | Exclude |
| `date`, `call_started`, `daily_caller` | Yes | Use / extract features |
| `call_answered`, `call_ended` | No — after the wait | Exclude |
| `wait_length` | No — and it *is* the label rule | Exclude |
| `service_length` | No — after the call ends | Exclude |

> **Screenshot 2 — Class distribution**
""")

code("""vc = df['meets_standard'].value_counts()
vc_pct = df['meets_standard'].value_counts(normalize=True) * 100
print('True :', f\"{vc[True]:,}\", f'({vc_pct[True]:.2f}%)')
print('False:', f\"{vc[False]:,}\", f'({vc_pct[False]:.2f}%)')
print('Imbalance ratio True:False =', round(vc[True] / vc[False], 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(['True (met)', 'False (not met)'], [vc[True], vc[False]],
            color=['#2196F3', '#E53935'], edgecolor='white')
for i, v in enumerate([vc[True], vc[False]]):
    axes[0].text(i, v + 400, f'{v:,}', ha='center', fontweight='bold')
axes[0].set_title('Class distribution — count')
axes[0].set_xlabel('meets_standard')
axes[0].set_ylabel('Number of calls')
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
axes[1].pie([vc[True], vc[False]],
            labels=[f'True\\n{vc_pct[True]:.2f}%', f'False\\n{vc_pct[False]:.2f}%'],
            colors=['#2196F3', '#E53935'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class distribution — proportion')
plt.suptitle(f'Class distribution — {len(df):,} calls, heavily imbalanced', fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""df['hour'] = df['call_started_dt'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

hour_rate = df.groupby('hour')['meets_standard'].mean() * 100
axes[0].plot(hour_rate.index, hour_rate.values, marker='o', color='#1565C0')
axes[0].set_title('Meet rate by hour')
axes[0].set_xlabel('Hour of arrival')
axes[0].set_ylabel('% meeting standard')
axes[0].set_xticks(range(8, 18))

dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
dow_rate = df.groupby('day_of_week')['meets_standard'].mean() * 100
axes[1].bar([dow_map[i] for i in dow_rate.index], dow_rate.values, color='#00897B')
axes[1].set_title('Meet rate by weekday')
axes[1].set_xlabel('Weekday')
axes[1].set_ylabel('% meeting standard')
axes[1].set_ylim(85, 96)

sns.boxplot(data=df, x='meets_standard', y='daily_caller', ax=axes[2],
            palette=['#E53935', '#2196F3'])
axes[2].set_title('Queue position vs target')
axes[2].set_xlabel('meets_standard')
axes[2].set_ylabel('daily_caller')
plt.tight_layout()
plt.show()

print('Lowest hour meet-rate : hour', int(hour_rate.idxmin()), f'{hour_rate.min():.2f}%')
print('Lowest weekday        :', dow_map[int(dow_rate.idxmin())], f'{dow_rate.min():.2f}%')""")

code("""month_rate = df.groupby('month')['meets_standard'].mean() * 100
month_vol = df.groupby('month').size()
fig, ax1 = plt.subplots(figsize=(9, 4.4))
ax1.bar(month_vol.index, month_vol.values, color='#90CAF9')
ax1.set_xlabel('Month (2021)')
ax1.set_ylabel('Call volume')
ax2 = ax1.twinx()
ax2.plot(month_rate.index, month_rate.values, color='#C62828', marker='o')
ax2.set_ylabel('% meeting standard')
ax1.set_title('Monthly volume vs service-standard rate')
ax1.set_xticks(range(1, 13))
plt.tight_layout()
plt.show()

print(month_rate.round(2))""")

code("""# Heatmap includes wait_length only to show leakage risk. It is not a model input.
corr_cols = ['daily_caller', 'hour', 'day_of_week', 'month', 'day_of_month',
             'wait_length', 'service_length']
fig, ax = plt.subplots(figsize=(8.5, 6))
sns.heatmap(df[corr_cols].corr(numeric_only=True), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation heatmap (wait_length shown only for inspection)')
plt.tight_layout()
plt.show()""")

md("""**EDA interpretations (computed from this CSV)**

- Target is imbalanced: **47,481 True (91.83%)** vs **4,227 False (8.17%)**, ratio **11.23:1**.
- Hour 13 has the lowest meet rate (**89.86%**). Friday is the weakest weekday (**90.41%**).
- Meet rate falls through the year as volume rises: January **97.47%**, December **83.45%**. This is the dominant pattern in the data.
- Failed calls have a higher mean `daily_caller` (about 124 vs 102). Queue position is weakly associated with the target; it is not a substitute for `wait_length`.
- `wait_length` is sparse (median 0) because most callers are answered immediately. That is why a 60-second rule still yields ~92% True.
""")

md("""---
## Part 3 — Class imbalance

A constant “always True” classifier would score about **91.83% accuracy** and **0 recall on False**. Accuracy would look strong while the model ignored every missed SLA.

That is why this notebook reports **precision, recall, and F1 for both classes**, with extra attention to **False** (the minority / operationally costly class).

**Handling used**

- **Stratification is not used** because the split is chronological (next section). Class rates are allowed to differ across time, which matches the real year.
- **Balanced class weights** on the training labels, so the loss does not ignore False calls.
- **No SMOTE / oversampling.** The CSV is a complete simulated year, not a tiny sample of rare events. Synthetic arrivals would invent queue positions that never occurred. Weights are the lighter, more defensible adjustment.

False negatives (predict True, actually False): the centre is told a call will be fine when it will miss the 60-second standard.  
False positives (predict False, actually True): staff are warned about a call that will still be answered in time. For an operations dashboard, false negatives are the more serious error, but flooding staff with false alarms is also unusable. Both are reported.
""")

md("""---
## Part 4 — Feature engineering

**Kept**

- `daily_caller` — known when the call joins the day’s queue.
- Cyclical **hour**, **day of week**, **month** — hour 8 and hour 17 are close in a cycle; a raw integer would treat them as far apart. The same argument applies to weekdays and months.
- `day_of_month` — kept as a scaled numeric feature (billing-cycle / intra-month load is possible; it is not treated as circular because month lengths differ).

**Not created**

- Weekend flag: the file only contains Monday–Friday (dayofweek 0–4).
- `call_id`: unique row id, 51,708 distinct values.
- Minutes: too granular relative to a four-agent queue and likely to overfit.

Day-of-week is encoded with period **7** (calendar week), not 5, even though Saturday/Sunday are absent. The sine/cosine pair still places Mon–Fri on the weekly circle.
""")

code("""df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

FEATURE_COLS = [
    'daily_caller',
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    'day_of_month',
]

df = df.sort_values('call_started_dt').reset_index(drop=True)
X_all = df[FEATURE_COLS]
y_all = df['meets_standard'].astype(int)

print('Features:', FEATURE_COLS)
display(X_all.head())
print('Target rate overall: {:.2f}% True'.format(100 * y_all.mean()))""")

md("""---
## Part 5 — Train / validation / test split

The file is a **time-ordered** year. Meet rate is not stationary (97% in January, 83% in December).

A **random stratified split** would mix December congestion into training. The network could then “predict” SLA failures partly by remembering late-year month codes that, in deployment, belong to the future. That is temporal leakage.

**Choice: chronological 70% / 15% / 15%** after sorting by `call_started_dt`.

- Train: earliest 70% of calls  
- Validation: next 15% (hyperparameter comparison only)  
- Test: final 15% (untouched until the last evaluation)

The scaler is fit on **train only**.
""")

code("""n = len(df)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train = df.iloc[:n_train]
val   = df.iloc[n_train:n_train + n_val]
test  = df.iloc[n_train + n_val:]

def split_report(name, part):
    y = part['meets_standard']
    print(f\"{name:12s} {len(part):6,}  {part['date'].min().date()} to {part['date'].max().date()}  \"
          f\"True {100*y.mean():.2f}%  False {100*(1-y.mean()):.2f}%\")

print(f\"{'split':12s} {'n':>6}  date range                    class mix\")
split_report('Train', train)
split_report('Validation', val)
split_report('Test', test)

X_train, y_train = train[FEATURE_COLS], train['meets_standard'].astype(int).to_numpy()
X_val,   y_val   = val[FEATURE_COLS],   val['meets_standard'].astype(int).to_numpy()
X_test,  y_test  = test[FEATURE_COLS],  test['meets_standard'].astype(int).to_numpy()""")

md("""Recorded split from this CSV:

| Split | Rows | Dates | True | False |
|-------|------|-------|------|-------|
| Train | 36,195 | 2021-01-01 to 2021-10-08 | 94.54% | 5.46% |
| Validation | 7,756 | 2021-10-08 to 2021-11-23 | 87.02% | 12.98% |
| Test | 7,757 | 2021-11-23 to 2021-12-31 | 83.95% | 16.05% |

The minority class becomes more common later in the year. That is expected and is exactly why a random split would be misleading.
""")

md("""---
## Part 6 — Preprocessing

> **Screenshot 3 — Feature preprocessing**
""")

code("""scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit on train only
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
print('Train feature means (approx 0):', np.round(X_train_s.mean(axis=0), 3))
print('Train feature stds  (approx 1):', np.round(X_train_s.std(axis=0), 3))
print('Class weights (False, True)   :', class_weight_dict)""")

md("""StandardScaler puts `daily_caller` and the cyclical features on a comparable scale for gradient descent. Cyclical features already lie in [-1, 1]; scaling them is still consistent and does not leak test statistics because the scaler is train-only.

Recorded training class weights: **False = 9.16**, **True = 0.53**.
""")

md("""---
## Part 7 — Baseline neural network

```
Input (8 features)
        ↓
Dense(32, ReLU)
        ↓
Dense(1, sigmoid)
```

- **Sigmoid** maps the last unit to a probability in (0, 1) for a single binary label.  
- **Binary cross-entropy** is the matching loss for a Bernoulli target with a sigmoid output.  
- **Adam** is used with a stated learning rate in each experiment.  
- Metrics during fit: accuracy and AUC. Assignment scoring uses the confusion matrix, precision, recall, and F1 **after** training, on probabilities thresholded at 0.5.

> **Screenshot 4 — Neural-network architecture / `model.summary()`**
""")

code("""def build_model(hidden_layers, learning_rate, dropout_rate=0.0):
    keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = keras.Sequential(name='CallCentreNN')
    model.add(keras.Input(shape=(X_train_s.shape[1],)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, seed=SEED))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')],
    )
    return model

results_log = []

def evaluate_split(name, y_true, y_pred):
    return {
        'split': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'prec_false': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'rec_false': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1_false': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        'prec_true': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'rec_true': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_true': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

def train_and_evaluate(exp_name, model, epochs, batch_size):
    history = model.fit(
        X_train_s, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_val_s, y_val),
        class_weight=class_weight_dict,
        verbose=2,
    )
    y_proba = model.predict(X_val_s, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    m = evaluate_split('val', y_val, y_pred)
    m['Experiment'] = exp_name
    results_log.append(m)
    print('\\nValidation metrics')
    print(f\"  Accuracy     {m['accuracy']:.4f}\")
    print(f\"  False  P/R/F1 {m['prec_false']:.4f} / {m['rec_false']:.4f} / {m['f1_false']:.4f}\")
    print(f\"  True   P/R/F1 {m['prec_true']:.4f} / {m['rec_true']:.4f} / {m['f1_true']:.4f}\")
    print(f\"  Macro F1      {m['f1_macro']:.4f}\")
    return history, y_pred

def plot_training(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].plot(history.history['loss'], label='Train', color='#1565C0', lw=2)
    axes[0].plot(history.history['val_loss'], label='Validation', color='#EF6C00', lw=2, ls='--')
    axes[0].set_title(title + ' — loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Binary cross-entropy'); axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train', color='#1565C0', lw=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', color='#EF6C00', lw=2, ls='--')
    axes[1].set_title(title + ' — accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.show()

baseline = build_model([32], 0.001)
baseline.summary()""")

md("""---
## Part 8 — Hyperparameter experiments

One group of settings is changed at a time.

| Experiment | Change | Architecture | LR | Batch | Epochs | Dropout |
|------------|--------|--------------|----|-------|--------|---------|
| 1 Baseline | starting point | [32] | 0.001 | 32 | 20 | 0 |
| 2 Capacity | add a hidden layer | [64, 32] | 0.001 | 32 | 30 | 0 |
| 3 Learning rate | halve LR vs Exp 2 | [64, 32] | 0.0005 | 32 | 30 | 0 |
| 4 Regularisation | add dropout vs Exp 3 | [64, 32] | 0.0005 | 32 | 30 | 0.3 |

Hyperparameters are compared **only on the validation set**. The test set is unused until Part 10.

> **Screenshot 5 — training logs**  
> **Screenshot 6 — loss/accuracy curves**
""")

code("""print('Experiment 1 — Baseline')
model_e1 = build_model([32], 0.001)
hist_e1, pred_e1 = train_and_evaluate('Exp 1 — Baseline', model_e1, epochs=20, batch_size=32)
plot_training(hist_e1, 'Exp 1 — Baseline')""")

md("""**Exp 1 (recorded run).** Train accuracy rose to **0.638** while validation accuracy stayed near **0.26** (final **0.2617**). Validation loss was lowest at epoch 1 (**0.856**) and finished at **1.036**. Minority recall was high (**0.8808**) but precision was **0.1366**, so F1(False) = **0.2365**. Class weights made the network flag many calls as likely SLA misses; under later-year congestion that produces lots of false alarms.
""")

code("""print('Experiment 2 — Increased capacity (only architecture changed vs a deeper net; LR stays 0.001)')
model_e2 = build_model([64, 32], 0.001)
hist_e2, pred_e2 = train_and_evaluate('Exp 2 — Capacity', model_e2, epochs=30, batch_size=32)
plot_training(hist_e2, 'Exp 2 — Capacity')""")

md("""**Exp 2 (recorded run).** Extra capacity improved **training** (final train acc **0.689**, train loss **0.537**) but validation AUC drifted down and F1(False) fell to **0.2199**. This is overfitting to early-2021 patterns that do not hold in October–November.
""")

code("""print('Experiment 3 — Lower learning rate (architecture same as Exp 2)')
model_e3 = build_model([64, 32], 0.0005)
hist_e3, pred_e3 = train_and_evaluate('Exp 3 — Lower LR', model_e3, epochs=30, batch_size=32)
plot_training(hist_e3, 'Exp 3 — Lower LR')""")

md("""**Exp 3 (recorded run).** Halving the learning rate raised validation accuracy to **0.4116** and F1(False) to **0.2391** (best minority F1 among the four). Training was smoother than Exp 2. This configuration is selected for the test set.
""")

code("""print('Experiment 4 — Dropout 0.3 (only regularisation changed vs Exp 3)')
model_e4 = build_model([64, 32], 0.0005, dropout_rate=0.3)
hist_e4, pred_e4 = train_and_evaluate('Exp 4 — Dropout', model_e4, epochs=30, batch_size=32)
plot_training(hist_e4, 'Exp 4 — Dropout')""")

md("""**Exp 4 (recorded run).** Dropout cut the train–val loss gap (final val loss **0.907**, the lowest of the four) but F1(False) was **0.2334**, below Exp 3. Regularisation helped calibration of the loss a little; it did not recover minority precision.
""")

md("""---
## Part 9 — Training visualisation

Across all four recorded curves:

- Training loss decreases; validation loss is already high at epoch 1 and does not fall below that first-epoch value. The network is fitting the **early-year** majority class mix and then struggling on later months.
- Accuracy plots are easy to misread. Validation accuracy around 0.26–0.41 is **worse than predicting True always** on the validation window (~87% True). That is a direct effect of class weights plus distribution shift, not a coding error.
- Exp 2 shows the widest train/val accuracy gap (overfitting from extra capacity).
- Exp 3 is the most usable compromise: slower updates, higher validation accuracy/F1 than Exp 2.
- Exp 4 is more stable in loss but more conservative in a way that did not win F1(False).
""")

md("""---
## Part 11 — Model comparison (validation)

> **Screenshot 9 — comparison table**

Selection rule, set **before** looking at the test set: highest validation **F1(False)**, then macro F1.
""")

code("""config = pd.DataFrame([
    {'Experiment': 'Exp 1 — Baseline', 'Architecture': '[32]', 'LR': 0.001, 'Batch': 32, 'Epochs': 20, 'Dropout': 0.0},
    {'Experiment': 'Exp 2 — Capacity', 'Architecture': '[64, 32]', 'LR': 0.001, 'Batch': 32, 'Epochs': 30, 'Dropout': 0.0},
    {'Experiment': 'Exp 3 — Lower LR', 'Architecture': '[64, 32]', 'LR': 0.0005, 'Batch': 32, 'Epochs': 30, 'Dropout': 0.0},
    {'Experiment': 'Exp 4 — Dropout', 'Architecture': '[64, 32]', 'LR': 0.0005, 'Batch': 32, 'Epochs': 30, 'Dropout': 0.3},
])
table = config.merge(pd.DataFrame(results_log), on='Experiment')
display(table.round(4))
best_row = table.sort_values(['f1_false', 'f1_macro'], ascending=False).iloc[0]
print('Selected on validation F1(False):', best_row['Experiment'])""")

md("""**Recorded validation comparison**

| Experiment | Architecture | LR | Batch | Epochs | Accuracy | Prec(F) | Rec(F) | F1(F) | F1(T) |
|------------|--------------|----|-------|--------|----------|---------|--------|-------|-------|
| 1 Baseline | [32] | 0.001 | 32 | 20 | 0.2617 | 0.1366 | 0.8808 | 0.2365 | 0.2853 |
| 2 Capacity | [64, 32] | 0.001 | 32 | 30 | 0.3394 | 0.1298 | 0.7170 | 0.2199 | 0.4271 |
| **3 Lower LR** | [64, 32] | 0.0005 | 32 | 30 | **0.4116** | **0.1437** | 0.7120 | **0.2391** | **0.5203** |
| 4 Dropout | [64, 32] | 0.0005 | 32 | 30 | 0.3124 | 0.1365 | 0.8064 | 0.2334 | 0.3766 |

- Capacity (Exp 2 vs 1) raised accuracy but **hurt** F1(False).
- Lower LR (Exp 3 vs 2) improved accuracy, True F1, and False F1 together.
- Dropout (Exp 4 vs 3) did not beat Exp 3 on the selection metric.

**Final model for the test set:** Experiment 3, the already-trained validation winner. The test set is not used to pick hyperparameters. The model is **not** refit on train+val, so the reported test numbers match the same weights that produced the validation table.
""")

md("""---
## Part 10 — Final evaluation on the untouched test set

> **Screenshot 7 — confusion matrix**  
> **Screenshot 8 — precision / recall / F1**  
> **Screenshot 10 — final model results**
""")

code("""# Test set is evaluated once, using the validation-selected experiment (Exp 3).
final_model = model_e3
y_test_proba = final_model.predict(X_test_s, verbose=0).ravel()
y_test_pred = (y_test_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
ConfusionMatrixDisplay(cm, display_labels=['False (not met)', 'True (met)']).plot(
    cmap='Blues', ax=ax, colorbar=False, values_format='d')
ax.set_title('Test confusion matrix — Exp 3 (lower learning rate)')
ax.set_xlabel('Predicted label')
ax.set_ylabel('Actual label')
plt.tight_layout()
plt.show()

print(classification_report(
    y_test, y_test_pred,
    target_names=['False (not met)', 'True (met)'],
    digits=4, zero_division=0,
))
print('Accuracy:', round(accuracy_score(y_test, y_test_pred), 4))
print('Macro F1:', round(f1_score(y_test, y_test_pred, average='macro'), 4))
print('Weighted F1:', round(f1_score(y_test, y_test_pred, average='weighted'), 4))
print('\\nConfusion matrix [[TN FP],[FN TP]]:')
print(cm)""")

md("""**Recorded test results (Exp 3, seed 42)**

Confusion matrix (actual × predicted):

|  | Predicted False | Predicted True |
|--|-----------------|----------------|
| **Actual False** | TN = 803 | FP = 442 |
| **Actual True** | FN = 4,194 | TP = 2,318 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| False (not met) | 0.1607 | 0.6450 | 0.2573 | 1,245 |
| True (met) | 0.8399 | 0.3560 | 0.5000 | 6,512 |
| Accuracy |  |  | **0.4023** | 7,757 |
| Macro avg | 0.5003 | 0.5005 | 0.3786 | 7,757 |
| Weighted avg | 0.7308 | 0.4023 | 0.4610 | 7,757 |

**What the errors mean here**

- **False negative (4,194):** call is predicted to meet the standard but waited more than 60 s. Supervisors would not be warned.  
- **False positive (442):** call is predicted to miss the standard but was actually answered in time. Unnecessary alert.

The model, pushed by class weights, catches **64.5%** of real misses (recall False) but only **16.1%** of “miss” alarms are correct (precision False). Accuracy **40.2%** is far below the 84.0% True rate in the test window. That gap is the point of the assignment: **accuracy is not the story**.
""")

md("""---
## Part 12 — Discoveries

1. **The label is a wait-time rule.** `meets_standard` matches `wait_length <= 60` on every one of 51,708 rows. Including wait time would give a perfect classifier that cannot be used when a call arrives.

2. **Accuracy is the wrong headline.** 91.83% of all calls already meet the standard. After a chronological split the test majority is still 83.95% True. The selected network’s 40.23% accuracy is worse than a constant True predictor, while False recall is 64.50%. Those two facts only make sense together if class weights and shift are acknowledged.

3. **Time drift dominates arrival-time features.** Meet rate drops from 97.47% in January to 83.45% in December as volume rises (3,000 calls in January vs 6,249 in December). Train (to 8 Oct) is 94.54% True; test (23 Nov–31 Dec) is 83.95% True. A random split would hide this.

4. **Extra capacity overfit the first part of the year.** Exp 2 reached train accuracy 0.689 and train loss 0.537, but validation F1(False) fell to 0.2199 versus 0.2365 for the shallow baseline.

5. **A lower learning rate was the only change that helped both majority and minority metrics at once.** Exp 3 vs Exp 2: val accuracy 0.4116 vs 0.3394, F1(False) 0.2391 vs 0.2199, F1(True) 0.5203 vs 0.4271.

6. **Dropout stabilised loss more than it improved F1.** Exp 4 had the best final validation loss (0.907) and did not win the assignment metrics.

7. **Arrival context is a weak SLA predictor under this simulation.** With four fixed agents and no staffing feature, `daily_caller` plus clock features cannot reconstruct queue delay. That is a project finding, not a failed import: an honest Week 4 classifier is a warning system with limited precision, not a substitute for measuring wait time.

8. **Hour 13 and Friday are the weakest routine slots** (89.86% and 90.41% meet rates), but the month trend is larger than either of those effects.
""")

md("""---
## Part 13 — Recommended screenshots

| # | Capture | What it proves |
|---|---------|----------------|
| 1 | Part 1 load/inspect cells | Real CSV: 51,708 × 9, columns, dtypes |
| 2 | Class bar/pie chart | 91.83% / 8.17% imbalance |
| 3 | Scaler + class-weight cell | Train-only scaling; weights ~9.16 / 0.53 |
| 4 | `model.summary()` | Feed-forward classifier, sigmoid output |
| 5 | `verbose=2` epoch logs | Training actually ran; val loss/acc observed |
| 6 | Loss and accuracy plots | Overfitting / shift, not a single reported number |
| 7 | Test confusion matrix | TN 803, FP 442, FN 4194, TP 2318 |
| 8 | Classification report | Precision, recall, F1 both classes |
| 9 | Comparison table | Four controlled experiments |
| 10 | Test metrics markdown + printout | Final evaluation on unseen late-2021 calls |

Do not paste fabricated images. Run the notebook and capture the executed outputs. Saved copies of the experiment figures also live in `docs/figures/`.
""")

md("""---
## Part 14 — Report-ready write-up

### Introduction
Week 4 asks for a neural-network classifier on the same call-centre dataset used in Week 2 EDA. The operational question is whether an arriving call will meet a 60-second answer standard.

### Dataset
51,708 simulated calls from 1 Jan 2021 to 31 Dec 2021, 261 weekdays, nine columns. License: CC BY-SA 4.0 (Kaggle Call Centre Queue Simulation).

### Problem definition
Binary classification of `meets_standard`. Post-event columns are excluded so the model cannot read the answer.

### Preprocessing
Datetimes parsed from 12-hour clock strings. StandardScaler fit on the chronological training block only. Balanced class weights on train labels.

### Feature engineering
`daily_caller` plus cyclical hour, weekday, and month, and `day_of_month`. Identifiers and leakage columns removed.

### Architecture
Keras Sequential, ReLU hidden layers, sigmoid output, binary cross-entropy, Adam.

### Baseline and experiments
Four runs: shallow 32-unit net; deeper 64→32; same deeper net with LR 0.0005; same with dropout 0.3.

### Results
Validation selected Exp 3. Test accuracy 0.4023; False precision/recall/F1 = 0.1607 / 0.6450 / 0.2573; True F1 = 0.5000.

### Confusion matrix
See Part 10. Most errors are false negatives on the True class because the weighted model predicts False too often relative to the still-majority True rate in November–December.

### Findings
Leakage of wait time, imbalance, chronological shift, overfitting from extra depth, modest benefit from a lower learning rate, and limited signal in arrival-only features.

### Discussion
For the larger smart-queue project, this classifier is a **risk flag**, not an SLA oracle. A later RNN/LSTM waiting-time model would target `wait_length` directly — still without feeding the label rule back in as an input. Staffing level, which this CSV does not vary, is likely the missing driver of December’s missed-standard rate.

### Conclusion
The assignment requirements are met with a leakage-aware feed-forward classifier, four hyperparameter experiments, confusion matrix, and precision/recall/F1. The honest conclusion is that arrival-time features plus a 60-second label do not yield a high-precision SLA detector on later months, and that accuracy would have hidden that fact.

### GitHub
Publish this project folder (see README). Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository. Do not invent a link.
""")

md("""---
## Part 15 — How to publish

```bash
git init
git add .
git commit -m "Week 4 neural network implementation"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Place that URL on the assignment cover sheet. If the CSV is omitted from a public fork, keep the Kaggle attribution in the README (CC BY-SA 4.0 still requires share-alike credit).
""")

md("""---
## Checklist

| Requirement | Status |
|-------------|--------|
| Inspected the uploaded CSV (51,708 × 9) | Yes |
| Target `meets_standard` | Yes |
| `wait_length` excluded after proving the 60 s rule | Yes |
| Post-event columns excluded | Yes |
| `call_id` excluded | Yes |
| Class imbalance reported and not judged by accuracy alone | Yes |
| Chronological split; scaler fit on train only | Yes |
| Test set unused for tuning | Yes |
| Baseline feed-forward NN (Keras) | Yes |
| ≥3 controlled hyperparameter experiments (4 run) | Yes |
| Training/validation plots | Yes |
| Confusion matrix, precision, recall, F1 | Yes |
| Comparison table | Yes |
| Discoveries tied to measured numbers | Yes |
| Screenshot guide | Yes |
| GitHub layout / README / requirements | Yes |
| No invented GitHub URL | Yes |
| Not converted into an RNN/LSTM assignment | Yes |
""")

nb["cells"] = cells
out = Path(__file__).resolve().parents[1] / "notebooks" / "week4_neural_network.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", out, "cells", len(cells))
