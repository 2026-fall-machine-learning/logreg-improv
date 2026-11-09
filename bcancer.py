"""
Train and compare models on the breast cancer dataset, tracking every run
with MLflow so the params/metrics can be browsed and compared later with
`uv run mlflow ui`.
"""

import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42  # fixed so every model below trains/tests on the identical split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    train_size=0.9,
)

mlflow.set_experiment("bcancer")

# autolog() hooks into scikit-learn's .fit(): every time a model is trained
# inside an active run, MLflow automatically records its hyperparameters,
# training metrics, and the fitted model itself as an artifact - no manual
# mlflow.log_param() calls needed for any of that.
mlflow.sklearn.autolog()

# --- Baseline: logistic regression (same model as before) ------------------
with mlflow.start_run(run_name="logreg-baseline"):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"[logreg-baseline] test accuracy: {test_accuracy:.4f}")

# keep serving the logistic regression model from bcancer_app.py unchanged
joblib.dump(model, "bcancer_model.joblib")

# --- Decision trees with different hyperparameters --------------------------
# Hyperparameters are settings we choose *before* training (unlike the
# weights/splits the model learns from data). For DecisionTreeClassifier,
# these have the biggest effect on the classic overfitting-vs-underfitting
# trade-off:
#
# - max_depth: deepest the tree may grow. None = keep splitting until every
#   leaf is pure -> can memorize the training set (overfit). Small values
#   (e.g. 3) force a simple tree -> may underfit.
# - min_samples_leaf: minimum number of training samples a leaf must contain.
#   Higher values smooth the decision boundary and reduce overfitting.
# - criterion: impurity measure used to pick each split ("gini" vs
#   "entropy"/information gain). Usually a minor effect, but worth comparing.
tree_configs = [
    {
        "run_name": "tree-shallow",
        "params": {"max_depth": 3},
        "note": "max_depth=3 keeps the tree very simple - easy to interpret, "
        "but likely to underfit.",
    },
    {
        "run_name": "tree-unrestricted",
        "params": {"max_depth": None},
        "note": "max_depth=None (the default) lets the tree grow until "
        "leaves are pure - flexible, but prone to overfitting on 30 "
        "features.",
    },
    {
        "run_name": "tree-min-leaf-5",
        "params": {"max_depth": None, "min_samples_leaf": 5},
        "note": "min_samples_leaf=5 forces every leaf to cover at least 5 "
        "training samples, reining in the overfitting from an unrestricted "
        "depth.",
    },
    {
        "run_name": "tree-entropy",
        "params": {"max_depth": 5, "criterion": "entropy"},
        "note": "criterion='entropy' swaps the split-quality measure from "
        "Gini impurity to information gain; results are usually close to "
        "gini but worth comparing.",
    },
]

for config in tree_configs:
    print(f"\n[{config['run_name']}] params={config['params']}")
    print(f"  -> {config['note']}")

    with mlflow.start_run(run_name=config["run_name"]):
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, **config["params"])
        tree.fit(X_train, y_train)

        test_accuracy = tree.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.set_tag("note", config["note"])
        print(f"  -> test accuracy: {test_accuracy:.4f}")

print("\nRun `uv run mlflow ui` and open http://127.0.0.1:5000 to compare all runs.")
