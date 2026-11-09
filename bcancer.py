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

# Track whichever trained model gets the best held-out accuracy across *all*
# runs below, so we save that one (not just whichever trained last) for
# bcancer_app.py to serve.
best_model = None
best_run_name = None
best_accuracy = -1.0


def consider(run_name, candidate, accuracy):
    global best_model, best_run_name, best_accuracy
    if accuracy > best_accuracy:
        best_model, best_run_name, best_accuracy = candidate, run_name, accuracy


# --- Baseline: logistic regression (same model as before) ------------------
with mlflow.start_run(run_name="logreg-baseline"):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"[logreg-baseline] test accuracy: {test_accuracy:.4f}")
    consider("logreg-baseline", model, test_accuracy)

# --- Decision trees: a more realistic hyperparameter search -----------------
# Real tuning rarely changes one knob at a time - these configs combine
# several hyperparameters at once, the way you'd actually set up a
# DecisionTreeClassifier for this dataset:
#
# - max_depth: deepest the tree may grow. None = keep splitting until every
#   leaf is pure -> can memorize the training set (overfit). Small values
#   (3-6) keep it simple -> may underfit.
# - min_samples_leaf / min_samples_split: minimum samples a leaf must contain,
#   and minimum samples required before even considering a split. Together
#   they control how much evidence the tree needs before it commits to a
#   split - higher values smooth the decision boundary and fight overfitting.
# - criterion: impurity measure used to score a split - "gini" vs "entropy"
#   (information gain). Usually a small effect vs. the knobs above.
# - ccp_alpha: cost-complexity pruning strength. 0.0 = no pruning; grow the
#   tree first, then prune back any branch that doesn't improve accuracy by
#   at least `ccp_alpha` - a more principled alternative to guessing
#   max_depth up front.
# - max_features: how many of the 30 features to consider at each split.
#   "sqrt" only looks at a random subset per split (the trick random forests
#   use internally), adding regularization; None considers all of them.
# - class_weight: this dataset is mildly imbalanced (212 malignant vs. 357
#   benign samples). "balanced" reweights the loss so the minority class
#   (malignant) isn't under-prioritized just because it's less common.
tree_configs = [
    {
        "run_name": "tree-shallow-balanced",
        "params": {"max_depth": 4, "class_weight": "balanced"},
        "note": "Shallow, interpretable tree with class_weight='balanced' "
        "to avoid under-prioritizing the malignant class.",
    },
    {
        "run_name": "tree-cost-complexity-pruned",
        "params": {"max_depth": None, "ccp_alpha": 0.01},
        "note": "Let the tree grow fully, then prune it back with "
        "ccp_alpha - usually a better regularizer than a hand-picked "
        "max_depth.",
    },
    {
        "run_name": "tree-conservative",
        "params": {
            "max_depth": 5,
            "min_samples_leaf": 5,
            "min_samples_split": 10,
            "ccp_alpha": 0.005,
        },
        "note": "Stacks several mild regularizers together (depth cap + "
        "leaf/split minimums + light pruning) - closer to what a cautious, "
        "production-minded tuning pass looks like.",
    },
    {
        "run_name": "tree-entropy-balanced",
        "params": {
            "max_depth": 6,
            "criterion": "entropy",
            "class_weight": "balanced",
        },
        "note": "Swaps in information gain instead of Gini impurity, "
        "combined with class balancing.",
    },
    {
        "run_name": "tree-random-feature-subset",
        "params": {"max_depth": None, "max_features": "sqrt"},
        "note": "Only considers a random sqrt(30)~=5 features per split, "
        "adding randomness/regularization to a single tree.",
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
        consider(config["run_name"], tree, test_accuracy)

print(f"\nBest model: {best_run_name} (test accuracy: {best_accuracy:.4f})")
joblib.dump(best_model, "bcancer_model.joblib")
print("Saved best model to bcancer_model.joblib")

print("Run `uv run mlflow ui` and open http://127.0.0.1:5000 to compare all runs.")
