"""
Logistic regression baseline -- the PS explicitly requires comparing the
world model against this. Same windowed features, but zero temporal
awareness: every window is classified on its own, using nothing but that
window's own feature vector. This is deliberately the simplest possible
model, so any improvement the world model claims is measured against a
real floor, not a strawman -- and its lead time is 0 by construction,
which is the whole point of the comparison.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def flatten_for_baseline(X, y_attack):
    """X: [n_samples, seq_len, n_features] -> [n_samples*seq_len, n_features].
    Every window becomes its own independent training example -- this IS
    the 'per-flow classification, no context' baseline the PS contrasts
    the world model against, not an approximation of it."""
    n_samples, seq_len, n_features = X.shape
    return X.reshape(-1, n_features), y_attack.reshape(-1)


def train_baseline(X_train, y_train):
    X_flat, y_flat = flatten_for_baseline(X_train, y_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_flat, y_flat)
    return clf


def evaluate_baseline(clf, X_test, y_test):
    X_flat, y_flat = flatten_for_baseline(X_test, y_test)
    preds = clf.predict(X_flat)
    return {
        "f1": f1_score(y_flat, preds, zero_division=0),
        "report": classification_report(y_flat, preds, output_dict=True, zero_division=0),
    }


if __name__ == "__main__":
    from data import make_dataset

    print("smoke-testing the baseline on synthetic data...\n")
    X, y_attack, y_stage = make_dataset(n_samples=400, seq_len=40, seed=1)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_attack[:split], y_attack[split:]

    clf = train_baseline(X_train, y_train)
    result = evaluate_baseline(clf, X_test, y_test)

    print(f"baseline F1 (attack vs benign, per-window, no temporal context): {result['f1']:.4f}")
    print("\nthis is the number the world model's F1 needs to beat on the same "
          "test split, and its lead time is 0 by definition -- a same-window "
          "classifier structurally cannot warn before the window it's looking at.")
