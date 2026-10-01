"""Run Week 4 EDA, training experiments, and evaluation on the real call-centre CSV."""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml_experiments"))

from nn_classifier import SEED, build_classifier, classification_metrics, predict_labels, set_seeds

CSV_PATH = ROOT / "data" / "simulated_call_centre.csv"
FIG_DIR = ROOT / "docs" / "figures"
RESULTS_PATH = ROOT / "results.json"

FEATURE_COLS = [
    "daily_caller",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "day_of_month",
]


def style_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="whitegrid", context="notebook")


def combine_date_time(date_col: pd.Series, time_str_col: pd.Series) -> pd.Series:
    t = pd.to_datetime(time_str_col, format="%I:%M:%S %p", errors="coerce")
    return date_col + pd.to_timedelta(
        t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second, unit="s"
    )


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["call_started_dt"] = combine_date_time(df["date"], df["call_started"])
    df["call_answered_dt"] = combine_date_time(df["date"], df["call_answered"])
    df["call_ended_dt"] = combine_date_time(df["date"], df["call_ended"])
    df["hour"] = df["call_started_dt"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["y"] = df["meets_standard"].astype(int)
    return df.sort_values("call_started_dt").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()


def plot_eda(df: pd.DataFrame) -> dict:
    vc = df["meets_standard"].value_counts()
    vc_pct = df["meets_standard"].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["True (met)", "False (not met)"]
    counts = [int(vc[True]), int(vc[False])]
    axes[0].bar(labels, counts, color=["#2196F3", "#E53935"], edgecolor="white")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 400, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Class distribution — count")
    axes[0].set_xlabel("meets_standard")
    axes[0].set_ylabel("Number of calls")
    axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].pie(
        counts,
        labels=[f"True\n{vc_pct[True]:.2f}%", f"False\n{vc_pct[False]:.2f}%"],
        colors=["#2196F3", "#E53935"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class distribution — proportion")
    fig.suptitle(f"Class distribution — {len(df):,} calls, heavily imbalanced", fontweight="bold")
    savefig("fig1_class_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.histplot(df, x="daily_caller", hue="meets_standard", bins=40, ax=ax, palette=["#E53935", "#2196F3"])
    ax.set_title("Daily caller position by service-standard outcome")
    ax.set_xlabel("daily_caller (queue position that day)")
    ax.set_ylabel("Count")
    savefig("fig2_daily_caller_by_target.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(data=df, x="meets_standard", y="daily_caller", ax=ax, palette=["#E53935", "#2196F3"])
    ax.set_title("Queue position vs meets_standard")
    ax.set_xlabel("meets_standard")
    ax.set_ylabel("daily_caller")
    savefig("fig3_daily_caller_boxplot.png")

    hour_rate = df.groupby("hour")["meets_standard"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hour_rate.index, hour_rate.values, marker="o", color="#1565C0", lw=2)
    ax.set_title("Service-standard rate by hour of arrival")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of calls meeting standard")
    ax.set_xticks(range(8, 18))
    savefig("fig4_hour_meet_rate.png")

    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    dow_rate = df.groupby("day_of_week")["meets_standard"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([dow_map[i] for i in dow_rate.index], dow_rate.values, color="#00897B")
    ax.set_title("Service-standard rate by weekday")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("% of calls meeting standard")
    ax.set_ylim(85, 96)
    savefig("fig5_weekday_meet_rate.png")

    month_rate = df.groupby("month")["meets_standard"].mean() * 100
    month_vol = df.groupby("month").size()
    fig, ax1 = plt.subplots(figsize=(9, 4.4))
    ax1.bar(month_vol.index, month_vol.values, color="#90CAF9", label="Call volume")
    ax1.set_xlabel("Month (2021)")
    ax1.set_ylabel("Call volume")
    ax2 = ax1.twinx()
    ax2.plot(month_rate.index, month_rate.values, color="#C62828", marker="o", lw=2, label="Meet rate")
    ax2.set_ylabel("% meeting standard")
    ax1.set_title("Monthly volume vs service-standard rate")
    ax1.set_xticks(range(1, 13))
    savefig("fig6_month_volume_rate.png")

    corr_cols = ["daily_caller", "hour", "day_of_week", "month", "day_of_month", "wait_length", "service_length"]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (leakage columns included for inspection only)")
    savefig("fig7_correlation_heatmap.png")

    return {
        "n_rows": int(len(df)),
        "n_cols_raw": 9,
        "true_count": int(vc[True]),
        "false_count": int(vc[False]),
        "true_pct": float(vc_pct[True]),
        "false_pct": float(vc_pct[False]),
        "imbalance_ratio": float(vc[True] / vc[False]),
        "hour_meet_rate": {str(k): float(v) for k, v in hour_rate.items()},
        "dow_meet_rate": {dow_map[k]: float(v) for k, v in dow_rate.items()},
        "month_meet_rate": {str(k): float(v) for k, v in month_rate.items()},
        "wait_true_max": int(df.loc[df["meets_standard"], "wait_length"].max()),
        "wait_false_min": int(df.loc[~df["meets_standard"], "wait_length"].min()),
        "leakage_mismatches": int(((df["wait_length"] <= 60) != df["meets_standard"]).sum()),
        "service_length_zero": int((df["service_length"] <= 0).sum()),
        "duplicates": int(df.duplicated().sum()),
        "missing": int(df.isnull().sum().sum()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "unique_days": int(df["date"].nunique()),
    }


def plot_training(history, title: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].plot(history.history["loss"], label="Train", color="#1565C0", lw=2)
    axes[0].plot(history.history["val_loss"], label="Validation", color="#EF6C00", lw=2, ls="--")
    axes[0].set_title(f"{title} — loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Train", color="#1565C0", lw=2)
    axes[1].plot(history.history["val_accuracy"], label="Validation", color="#EF6C00", lw=2, ls="--")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    savefig(filename)


def plot_confusion(cm, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["False (not met)", "True (met)"],
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    savefig(filename)


def split_summary(name: str, part: pd.DataFrame) -> dict:
    y = part["y"]
    return {
        "name": name,
        "n": int(len(part)),
        "true": int((y == 1).sum()),
        "false": int((y == 0).sum()),
        "true_pct": float((y == 1).mean() * 100),
        "false_pct": float((y == 0).mean() * 100),
        "date_min": str(part["date"].min().date()),
        "date_max": str(part["date"].max().date()),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    style_plots()
    set_seeds(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare()
    eda = plot_eda(df)

    train, val, test = chronological_split(df)
    splits = {
        "train": split_summary("train", train),
        "val": split_summary("validation", val),
        "test": split_summary("test", test),
    }

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_val = scaler.transform(val[FEATURE_COLS])
    X_test = scaler.transform(test[FEATURE_COLS])
    y_train = train["y"].to_numpy()
    y_val = val["y"].to_numpy()
    y_test = test["y"].to_numpy()

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    experiments = [
        {
            "id": "exp1_baseline",
            "name": "Exp 1 — Baseline",
            "architecture": [32],
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 20,
            "dropout": 0.0,
        },
        {
            "id": "exp2_capacity",
            "name": "Exp 2 — Increased capacity",
            "architecture": [64, 32],
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 30,
            "dropout": 0.0,
        },
        {
            "id": "exp3_lower_lr",
            "name": "Exp 3 — Lower learning rate",
            "architecture": [64, 32],
            "learning_rate": 0.0005,
            "batch_size": 32,
            "epochs": 30,
            "dropout": 0.0,
        },
        {
            "id": "exp4_dropout",
            "name": "Exp 4 — Dropout regularisation",
            "architecture": [64, 32],
            "learning_rate": 0.0005,
            "batch_size": 32,
            "epochs": 30,
            "dropout": 0.3,
        },
    ]

    experiment_results = []
    histories = {}
    models = {}

    for spec in experiments:
        print(f"\n=== Training {spec['name']} ===", flush=True)
        model = build_classifier(
            n_features=X_train.shape[1],
            hidden_layers=spec["architecture"],
            learning_rate=spec["learning_rate"],
            dropout_rate=spec["dropout"],
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=spec["epochs"],
            batch_size=spec["batch_size"],
            class_weight=class_weight,
            verbose=2,
        )
        plot_training(
            history,
            spec["name"],
            f"fig_train_{spec['id']}.png",
        )
        y_pred, _ = predict_labels(model, X_val)
        metrics = classification_metrics(y_val, y_pred)
        result = {
            **spec,
            "val_metrics": metrics,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_train_acc": float(history.history["accuracy"][-1]),
            "final_val_acc": float(history.history["val_accuracy"][-1]),
            "min_val_loss": float(min(history.history["val_loss"])),
            "min_val_loss_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        }
        experiment_results.append(result)
        histories[spec["id"]] = history.history
        models[spec["id"]] = model
        print(
            f"Val acc={metrics['accuracy']:.4f}  F1(False)={metrics['f1_false']:.4f}  "
            f"Rec(False)={metrics['recall_false']:.4f}",
            flush=True,
        )

    best = max(experiment_results, key=lambda r: (r["val_metrics"]["f1_false"], r["val_metrics"]["f1_macro"]))
    best_id = best["id"]
    best_model = models[best_id]
    y_test_pred, y_test_proba = predict_labels(best_model, X_test)
    test_metrics = classification_metrics(y_test, y_test_pred)
    plot_confusion(
        test_metrics["confusion_matrix"],
        f"Test confusion matrix — {best['name']}",
        "fig8_confusion_matrix.png",
    )

    # Comparison bar chart of minority-class F1 on validation
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = [r["name"] for r in experiment_results]
    f1s = [r["val_metrics"]["f1_false"] for r in experiment_results]
    accs = [r["val_metrics"]["accuracy"] for r in experiment_results]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, accs, w, label="Val accuracy", color="#90CAF9")
    ax.bar(x + w / 2, f1s, w, label="Val F1 (False)", color="#EF5350")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation accuracy vs minority-class F1")
    ax.legend()
    savefig("fig9_experiment_comparison.png")

    payload = {
        "seed": SEED,
        "csv_path": str(CSV_PATH),
        "features": FEATURE_COLS,
        "excluded_features": {
            "call_id": "Identifier with no operational signal.",
            "wait_length": "Deterministic constructor of meets_standard (threshold 60 s).",
            "call_answered": "Known only after an agent answers.",
            "call_ended": "Known only after the call finishes.",
            "service_length": "Known only after service completes.",
        },
        "split": "chronological 70/15/15 by call_started_dt",
        "class_weight": class_weight,
        "eda": eda,
        "splits": splits,
        "experiments": [
            {
                k: v
                for k, v in r.items()
                if k != "architecture"
            }
            | {"architecture": r["architecture"]}
            for r in experiment_results
        ],
        "best_experiment_id": best_id,
        "best_experiment_name": best["name"],
        "test_metrics": test_metrics,
        "test_positive_rate_predicted": float(y_test_pred.mean()),
        "history": histories,
    }

    # Convert remaining numpy types
    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
    print("\nSaved", RESULTS_PATH)
    print("Best experiment:", best["name"])
    print("Test accuracy:", round(test_metrics["accuracy"], 4))
    print("Test F1 False:", round(test_metrics["f1_false"], 4))
    print(classification_report(y_test, y_test_pred, target_names=["False", "True"], digits=4, zero_division=0))


if __name__ == "__main__":
    main()
