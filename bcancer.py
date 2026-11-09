"""Train and compare models on the breast cancer dataset, tracking every run
with MLflow so hyperparameters, metrics, and the fitted model are recorded.

After running this, explore results with:   uv run mlflow ui
"""

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

# random_state pins the train/test split so every model below is trained and
# scored on *exactly* the same data. Without it, each run gets a different
# random split, which would make accuracy differences between models
# meaningless (you wouldn't know if a model did better, or just got an
# easier test set).
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    train_size=0.9,
)

mlflow.set_experiment("bcancer")
# autolog() captures model hyperparameters, training-time metrics, and the
# fitted model itself as an MLflow artifact for every scikit-learn model
# fit below - no need to log those by hand.
mlflow.sklearn.autolog()


def evaluate_and_log(run_name, model):
    """Fit `model`, log held-out test metrics to MLflow, and return it."""
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_family", type(model).__name__)

        print(f"{run_name:28s} test_accuracy={metrics['test_accuracy']:.4f}")
        return metrics["test_accuracy"], model


results = {}

# --- Baseline: Logistic Regression ------------------------------------------
results["logreg_baseline"] = evaluate_and_log(
    "logreg_baseline",
    LogisticRegression(max_iter=5000),
)

# --- Decision trees, each exploring a different hyperparameter -------------

# 1) max_depth caps how many yes/no questions ("splits") the tree can ask
#    before making a decision. A shallow tree (depth=3) is simple and
#    resists overfitting ("high bias"), but may be too simple to capture
#    real patterns in the data.
results["tree_max_depth_3"] = evaluate_and_log(
    "tree_max_depth_3",
    DecisionTreeClassifier(max_depth=3, random_state=42),
)

# 2) max_depth=None (the default) lets the tree grow until every leaf is
#    pure or has fewer than 2 samples. This usually fits the training data
#    almost perfectly ("low bias") but risks overfitting noise in the
#    training set ("high variance") -> worse generalization to new data.
results["tree_unrestricted"] = evaluate_and_log(
    "tree_unrestricted",
    DecisionTreeClassifier(max_depth=None, random_state=42),
)

# 3) min_samples_leaf requires every leaf to contain at least N training
#    examples. Combined with a moderate max_depth, this is a common way to
#    fight overfitting without hard-capping the tree's depth.
results["tree_min_samples_leaf_5"] = evaluate_and_log(
    "tree_min_samples_leaf_5",
    DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
)

# 4) criterion controls the impurity measure used to score each candidate
#    split: "gini" (default) vs. "entropy" (information gain). They usually
#    produce similar trees but can diverge on some datasets. min_samples_split
#    requires at least N samples at a node before it's even considered for
#    splitting (a coarser cousin of min_samples_leaf).
results["tree_entropy_min_split_20"] = evaluate_and_log(
    "tree_entropy_min_split_20",
    DecisionTreeClassifier(criterion="entropy", min_samples_split=20, random_state=42),
)

# --- Pick the best model (by test accuracy) for the serving app ------------
best_name, (best_acc, best_model) = max(results.items(), key=lambda kv: kv[1][0])
print(f"\nBest model: {best_name} (test_accuracy={best_acc:.4f})")

joblib.dump(best_model, "bcancer_model.joblib")
