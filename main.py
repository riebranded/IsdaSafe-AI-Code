import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import joblib

# ----------------------------- CONFIG ----------------------------------------
CSV_PATH      = None          # e.g. "rf_training_data.csv"; None = upload prompt
TARGET_COL    = "species"     # label column
TEST_SIZE     = 0.25          # fraction held out for the test set
RANDOM_SEED   = 42

# Random Forest hyperparameters
N_ESTIMATORS  = 300
MAX_DEPTH     = None          # None = grow until pure; set an int to limit
MIN_LEAF      = 1
CLASS_WEIGHT  = "balanced"    # keeps classes fair if counts drift; None to disable

DO_CROSS_VAL  = True          # 5-fold CV on the full data (slower but robust)
MODEL_PATH    = "fish_rf_model.joblib"
CM_PATH       = "confusion_matrix.png"
CLASS_ACC_PATH = "class_accuracy.png"   # per-class accuracy bar chart
# -----------------------------------------------------------------------------


def load_csv():
    if CSV_PATH:
        return pd.read_csv(CSV_PATH)
    try:
        from google.colab import files  # type: ignore
        print("Upload your training CSV (rf_training_data.csv)...")
        up = files.upload()
        return pd.read_csv(next(iter(up)))
    except ImportError:
        raise SystemExit("Not in Colab: set CSV_PATH to your training file.")


def main():
    df = load_csv()

    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column '{TARGET_COL}' not found. "
                         f"Columns present: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    print("Rows:", len(df))
    print("Features:", feature_cols)
    print("\nClass balance:")
    print(y.value_counts(), "\n")

    # ---- split -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    # ---- train -----------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ---- evaluate --------------------------------------------------------
    acc = model.score(X_test, y_test)
    print("=" * 55)
    print(f"Test accuracy: {acc:.4f}")
    print("=" * 55)
    print("\nClassification report (test set):\n")
    print(classification_report(y_test, model.predict(X_test)))

    print("Feature importances:")
    for f, imp in sorted(zip(feature_cols, model.feature_importances_),
                         key=lambda t: -t[1]):
        print(f"  {f:14s} {imp:.3f}")

    if DO_CROSS_VAL:
        print("\nRunning 5-fold cross-validation...")
        cv = cross_val_score(model, X, y, cv=5, n_jobs=-1)
        print(f"CV accuracy: {cv.mean():.4f}  (+/- {cv.std():.4f})")

    # ---- confusion matrix plot ------------------------------------------
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    print(f"\nSaved confusion matrix -> {CM_PATH}")

    # ---- per-class accuracy plot ----------------------------------------
    # per-class accuracy = correct predictions for a species / its test rows
    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy (test set):")
    for lab, v in zip(labels, per_class):
        print(f"  {lab:18s} {v*100:.1f}%")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, per_class * 100, color="#2a7ab0", width=0.55)
    ax.axhline(acc * 100, ls="--", color="gray", lw=1.2,
               label=f"Overall accuracy {acc*100:.1f}%")
    for b, v in zip(bars, per_class):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 - 3, f"{v*100:.1f}%",
                ha="center", va="top", fontsize=11, color="white",
                fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(CLASS_ACC_PATH, dpi=150)
    print(f"Saved per-class accuracy chart -> {CLASS_ACC_PATH}")

    # ---- save model bundle ----------------------------------------------
    bundle = {
        "model": model,
        "features": feature_cols,
        "classes": list(model.classes_),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    # ---- quick demo prediction ------------------------------------------
    demo = pd.DataFrame([[27.0, 7.5, 4.8, 0.10]], columns=feature_cols)
    pred = model.predict(demo)[0]
    proba = dict(zip(model.classes_, model.predict_proba(demo)[0].round(3)))
    print("\nDemo prediction for", demo.iloc[0].to_dict())
    print("  ->", pred)
    print("  probabilities:", proba)

    # ---- download in Colab ----------------------------------------------
    try:
        from google.colab import files  # type: ignore
        files.download(MODEL_PATH)
        files.download(CM_PATH)
        files.download(CLASS_ACC_PATH)
    except ImportError:
        pass


if __name__ == "__main__":
    main()