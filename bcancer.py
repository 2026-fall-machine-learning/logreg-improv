import itertools

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.9,
    # Fixed so every model below is compared on the *same* split - without
    # this, differences in accuracy could just be luck of the shuffle
    # rather than the model/hyperparameters actually being better.
    random_state=42,
)

mlflow.set_experiment("breast-cancer-classification")


def run_experiment(name: str, model, params: dict):
    """Fit `model`, log params/metrics/model to MLflow, return it with its metrics."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        # cloudpickle (plain pickle under the hood) instead of MLflow's
        # newer default "skops" format, which refuses to serialize
        # DecisionTree/RandomForest models as "untrusted" for security
        # reasons. Fine for our own local, trusted experiments.
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )

        print(f"{name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return model, metrics


results = []

# --- Baseline: the logistic regression we already had ---
lr_params = {"max_iter": 5000}
results.append(
    run_experiment("logistic-regression", LogisticRegression(**lr_params), lr_params)
)

# --- Decision trees: a small but realistic grid search ---
# In practice you rarely tune hyperparameters one at a time - you search
# combinations, because their effects interact (e.g. a deep tree is much
# less overfit-prone once min_samples_leaf is also raised). This grid
# covers the knobs that matter most for a DecisionTreeClassifier:
tree_param_grid = {
    # How many splits deep the tree may grow from root to leaf.
    # None = unlimited (grow until leaves are pure or tiny) - the classic
    # way a decision tree overfits.
    "max_depth": [3, 5, 8, None],
    # Minimum samples required in a leaf. Raising this smooths the
    # decision boundary by refusing to carve out tiny regions around a
    # handful of (possibly noisy) training points.
    "min_samples_leaf": [1, 5, 10],
    # Impurity measure used to pick the best split at each node. "gini" is
    # the scikit-learn default; "entropy" is the information-gain
    # criterion from classic ID3/C4.5. They usually agree but not always.
    "criterion": ["gini", "entropy"],
}

tree_param_combos = [
    dict(zip(tree_param_grid.keys(), values))
    for values in itertools.product(*tree_param_grid.values())
]


def tree_run_name(params: dict) -> str:
    depth = params["max_depth"] if params["max_depth"] is not None else "none"
    return f"tree_depth={depth}_leaf={params['min_samples_leaf']}_crit={params['criterion']}"


for params in tree_param_combos:
    results.append(
        run_experiment(
            tree_run_name(params),
            DecisionTreeClassifier(random_state=42, **params),
            params,
        )
    )

# --- A couple of extra, less commonly-grid-searched hyperparameters ---
# These interact oddly enough with the grid above (e.g. pruning after an
# unlimited max_depth) that they're usually explored separately rather
# than as another axis of the main grid.
results.append(
    run_experiment(
        "tree_ccp_alpha-pruned",
        # ccp_alpha: cost-complexity pruning. After growing, the tree
        # prunes back any subtree whose complexity isn't "worth" the
        # impurity reduction it buys - a principled alternative/complement
        # to max_depth for controlling overfitting. 0.0 (default) = no
        # pruning.
        DecisionTreeClassifier(random_state=42, ccp_alpha=0.01),
        {"ccp_alpha": 0.01},
    )
)
results.append(
    run_experiment(
        "tree_class_weight-balanced",
        # class_weight="balanced": reweights classes inversely to their
        # frequency. The breast cancer dataset is mildly imbalanced
        # (~63% benign / 37% malignant); this matters more on imbalanced
        # datasets where you care about not missing the minority class.
        DecisionTreeClassifier(
            random_state=42, max_depth=5, class_weight="balanced"
        ),
        {"max_depth": 5, "class_weight": "balanced"},
    )
)

best_model, best_metrics = max(results, key=lambda r: r[1]["accuracy"])
print(f"\nBest model accuracy: {best_metrics['accuracy']:.3f}")

joblib.dump(best_model, "bcancer_model.joblib")
