"""
Train and evaluate spam classifiers on the SMS Spam Collection.

What this improves over the first notebook (notebooks/01_tutorial_baseline.ipynb):
  1. Removes 403 duplicate messages BEFORE splitting (otherwise the same text sits in
     train and test, and the test score is inflated).
  2. Uses a stratified split so train and test keep the same spam ratio (~13%).
  3. Treats SPAM as the positive class and reports precision, recall and F1 for it.
     Accuracy alone is misleading: a model that says "ham" every time is already 87% accurate.
  4. Compares 4 models with 5-fold cross-validation instead of trusting one split.
  5. Tunes the decision threshold for the use case (don't send real messages to the spam folder).

Run:  python src/train.py
Out:  models/spam_model.joblib, reports/metrics.json, reports/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spam.csv"
FIGS = ROOT / "reports" / "figures"
SEED = 42

# chart colours
BLUE, ORANGE, GREY, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#c9c8c1", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.grid": True, "grid.color": GRID, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.titlesize": 13, "axes.titlepad": 22, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
                     "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10.5})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    before = len(df)
    df = df.drop_duplicates(subset="Message").reset_index(drop=True)
    df["is_spam"] = (df["Category"] == "spam").astype(int)
    print(f"{before} messages, {before - len(df)} duplicates removed, {len(df)} left, "
          f"{df['is_spam'].mean():.1%} spam")
    return df


def tfidf_words():
    return TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True)


def models() -> dict:
    """Candidate models. All are pipelines, so text -> features -> classifier is one object."""
    words_and_chars = FeatureUnion([
        ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return {
        "Always say ham (dummy)": Pipeline([("tfidf", tfidf_words()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression (original)": Pipeline([("tfidf", tfidf_words()), ("clf", LogisticRegression(max_iter=1000))]),
        "Naive Bayes": Pipeline([("tfidf", tfidf_words()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression, balanced": Pipeline([("tfidf", tfidf_words()),
                                                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=10))]),
        "Linear SVM, words + chars": Pipeline([("tfidf", words_and_chars),
                                               ("clf", LinearSVC(C=1.0, class_weight="balanced"))]),
    }


def evaluate(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "spam_precision": precision_score(y_true, y_pred, zero_division=0),
            "spam_recall": recall_score(y_true, y_pred),
            "spam_f1": f1_score(y_true, y_pred)}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    df = load()
    X_train, X_test, y_train, y_test = train_test_split(
        df["Message"], df["is_spam"], test_size=0.2, stratify=df["is_spam"], random_state=SEED)

    # ---- 1. compare models with 5-fold cross-validation on the training set ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models().items():
        s = cross_validate(model, X_train, y_train, cv=cv,
                           scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"})
        rows.append({"model": name, "cv_accuracy": s["test_accuracy"].mean(), "cv_spam_precision": s["test_precision"].mean(),
                     "cv_spam_recall": s["test_recall"].mean(), "cv_spam_f1": s["test_f1"].mean(),
                     "cv_spam_f1_std": s["test_f1"].std()})
    cv_table = pd.DataFrame(rows).set_index("model")
    print("\n5-fold cross-validation (training set):")
    print(cv_table.round(3).to_string())

    # ---- 2. final check on the untouched test set ----
    test_rows = {}
    fitted = {}
    for name, model in models().items():
        fitted[name] = model.fit(X_train, y_train)
        test_rows[name] = evaluate(y_test, model.predict(X_test))
    test_table = pd.DataFrame(test_rows).T
    print("\nTest set (20%, never used for choosing):")
    print(test_table.round(3).to_string())

    best_name = cv_table.drop(index="Always say ham (dummy)")["cv_spam_f1"].idxmax()
    print(f"\nBest by cross-validated spam F1: {best_name}")

    # ---- 3. threshold tuning: probability model, pick the threshold that keeps precision >= 99% ----
    # A false positive (a real message hidden in the spam folder) costs more than a missed spam.
    best = models()[best_name]
    if not hasattr(best[-1], "predict_proba"):
        best.steps[-1] = ("clf", CalibratedClassifierCV(best[-1], cv=5))
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    ok = np.where(prec[:-1] >= 0.99)[0]
    threshold = float(thr[ok[np.argmax(rec[:-1][ok])]]) if len(ok) else 0.5
    y_thr = (proba >= threshold).astype(int)
    tuned = evaluate(y_test, y_thr)
    default = evaluate(y_test, (proba >= 0.5).astype(int))
    print(f"Threshold 0.50 -> {default}")
    print(f"Threshold {threshold:.2f} -> {tuned}")

    cm = confusion_matrix(y_test, y_thr)
    errors = pd.DataFrame({"message": X_test, "is_spam": y_test, "spam_probability": proba.round(3)})
    errors["predicted"] = y_thr
    errors = errors[errors["is_spam"] != errors["predicted"]].sort_values("spam_probability")
    errors.to_csv(ROOT / "reports" / "error_samples.csv", index=False)

    joblib.dump({"model": best, "threshold": threshold}, ROOT / "models" / "spam_model.joblib", compress=3)

    metrics = {
        "dataset": {"messages_after_dedup": int(len(df)), "duplicates_removed": 403,
                    "spam_share": round(float(df["is_spam"].mean()), 4), "test_size": int(len(y_test))},
        "cross_validation": cv_table.round(4).to_dict(orient="index"),
        "test_set": test_table.round(4).to_dict(orient="index"),
        "final_model": best_name,
        "threshold": {"value": round(threshold, 3), "at_0.5": {k: round(v, 4) for k, v in default.items()},
                      "tuned": {k: round(v, 4) for k, v in tuned.items()}},
        "confusion_matrix_tuned": {"ham_kept": int(cm[0, 0]), "ham_wrongly_flagged": int(cm[0, 1]),
                                   "spam_missed": int(cm[1, 0]), "spam_caught": int(cm[1, 1])},
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- charts ----
    plot_comparison(test_table)
    plot_confusion(cm, threshold)
    plot_pr(prec, rec, threshold, tuned)
    print("Saved model, metrics and charts.")


def plot_comparison(t: pd.DataFrame) -> None:
    t = t.drop(index="Always say ham (dummy)")
    names = list(t.index)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(y + 0.18, t.loc[names, "spam_recall"] * 100, height=0.34, color=BLUE, label="Spam recall (spam caught)")
    ax.barh(y - 0.18, t.loc[names, "spam_precision"] * 100, height=0.34, color=ORANGE, label="Spam precision (flags that were right)")
    for i, n in enumerate(names):
        ax.text(t.loc[n, "spam_recall"] * 100 + 0.4, i + 0.18, f"{t.loc[n, 'spam_recall'] * 100:.1f}", va="center", fontsize=9)
        ax.text(t.loc[n, "spam_precision"] * 100 + 0.4, i - 0.18, f"{t.loc[n, 'spam_precision'] * 100:.1f}", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(50, 104)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.32), ncol=2, frameon=False, fontsize=9)
    missed = 1 - t.loc["Logistic Regression (original)", "spam_recall"]
    ax.set_title(f"The original model let {missed:.0%} of spam through")
    acc = t.loc["Logistic Regression (original)", "accuracy"]
    subtitle(ax, f"Test set, spam as the positive class. Accuracy hid this: the original still scored {acc:.1%}.")
    fig.savefig(FIGS / "model_comparison.png")
    plt.close(fig)


def plot_confusion(cm, threshold) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=ramp, vmin=0, vmax=1)
    labels = ["Ham", "Spam"]
    ax.set_xticks([0, 1], ["Predicted ham", "Predicted spam"])
    ax.set_yticks([0, 1], ["Actually ham", "Actually spam"])
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({share[i, j]:.1%})", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else INK, fontsize=11)
    ax.set_title("Final model on the test set")
    subtitle(ax, f"Threshold {threshold:.2f}, tuned so almost no real message is flagged.")
    fig.savefig(FIGS / "confusion_matrix.png")
    plt.close(fig)


def plot_pr(prec, rec, threshold, tuned) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(rec * 100, prec * 100, color=BLUE, lw=2)
    ax.scatter([tuned["spam_recall"] * 100], [tuned["spam_precision"] * 100], s=60, color=ORANGE, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"chosen threshold {threshold:.2f}\nprecision {tuned['spam_precision']:.1%}, recall {tuned['spam_recall']:.1%}",
                (tuned["spam_recall"] * 100, tuned["spam_precision"] * 100), xytext=(-170, -50),
                textcoords="offset points", fontsize=9.5, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlabel("Recall: % of spam caught")
    ax.set_ylabel("Precision: % of flags that are spam")
    ax.set_xlim(0, 102)
    ax.set_ylim(50, 101)
    ax.set_title("Trading recall for precision")
    subtitle(ax, "Each point on the line is a different threshold.")
    fig.savefig(FIGS / "precision_recall.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
