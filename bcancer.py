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

# --- Decision trees: realistic hyperparameter combos --------------------
# In practice you rarely tune one hyperparameter in isolation - you combine
# a few that each fight overfitting in a different way. Each config below
# is a plausible "real" setting, not just a single-knob demo.
tree_configs = {
    # A conservative, shallow tree: low max_depth *and* a minimum leaf size
    # together keep the tree simple on two fronts at once. Good baseline
    # for "will a simple tree do?" before reaching for anything fancier.
    "tree_shallow_regularized": dict(
        max_depth=3,
        min_samples_leaf=10,
        criterion="gini",
    ),
    # A moderate-depth tree with both min_samples_split (how many samples
    # a node needs before it's even considered for splitting) and
    # min_samples_leaf (how many samples each resulting leaf must keep).
    # Using both together is a common way to prevent the tree from
    # carving out tiny, noise-driven leaves while still letting it grow
    # deep enough to find real structure.
    "tree_moderate_depth": dict(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        criterion="gini",
    ),
    # A deeper tree using "entropy" (information gain) instead of the
    # default "gini" impurity, paired with a leaf-size floor so the extra
    # depth doesn't just memorize outliers.
    "tree_deep_entropy": dict(
        max_depth=8,
        min_samples_leaf=8,
        criterion="entropy",
    ),
    # No depth cap at all - instead we lean entirely on sample-count
    # constraints (min_samples_split / min_samples_leaf) to control
    # complexity. This is a different philosophy than capping max_depth:
    # it lets the tree go as deep as it needs in "easy" regions of the
    # data, while still refusing to split on just a handful of points.
    "tree_unrestricted_depth_sample_regularized": dict(
        max_depth=None,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion="gini",
    ),
    # Cost-complexity pruning (ccp_alpha): grow the tree fully, then prune
    # back any branch whose added accuracy doesn't justify its added
    # complexity, weighted by ccp_alpha. Higher ccp_alpha -> more
    # aggressive pruning -> simpler tree. This is a *post*-hoc technique,
    # distinct from pre-stopping rules like max_depth or min_samples_leaf.
    "tree_ccp_pruned": dict(
        max_depth=None,
        ccp_alpha=0.01,
    ),
    # max_features restricts each split to consider only a random subset
    # of features (here, sqrt(n_features)) rather than all of them. This
    # injects randomness per split - the same idea random forests build
    # on - which can reduce overfitting at the cost of a possibly weaker
    # single tree.
    "tree_feature_subsampling": dict(
        max_depth=6,
        max_features="sqrt",
        min_samples_leaf=5,
        criterion="entropy",
    ),
}

for name, params in tree_configs.items():
    results[name] = evaluate_and_log(
        name,
        DecisionTreeClassifier(random_state=42, **params),
    )

# --- Pick the best model (by test accuracy) for the serving app ------------
best_name, (best_acc, best_model) = max(results.items(), key=lambda kv: kv[1][0])
print(f"\nBest model: {best_name} (test_accuracy={best_acc:.4f})")

joblib.dump(best_model, "bcancer_model.joblib")
