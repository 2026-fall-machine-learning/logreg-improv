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

# --- Decision trees with different hyperparameters ---
# Hyperparameters are settings you choose *before* training (unlike the
# learned coefficients) that control how the tree is built:
tree_configs = [
    # No constraints: the tree keeps splitting until every leaf is pure.
    # On a small dataset (569 rows) this tends to overfit the training set.
    ("tree-unconstrained", {}),
    # max_depth: how many splits deep the tree can go from root to leaf.
    # Shallow trees (3-5) are simpler/more interpretable and resist
    # overfitting, usually at a small cost in accuracy.
    ("tree-max_depth-3", {"max_depth": 3}),
    ("tree-max_depth-5", {"max_depth": 5}),
    # min_samples_leaf: minimum number of training samples allowed in a
    # leaf. Raising it smooths the decision boundary by refusing to carve
    # out tiny regions around a handful of noisy points.
    ("tree-min_samples_leaf-5", {"max_depth": 5, "min_samples_leaf": 5}),
    # min_samples_split: minimum samples a node must have before it's even
    # allowed to split further - another lever against overfitting.
    ("tree-min_samples_split-10", {"min_samples_split": 10}),
    # criterion: the impurity measure used to pick the best split at each
    # node. "gini" (default) and "entropy" (information gain, classic
    # ID3/C4.5) usually produce similar trees.
    ("tree-criterion-entropy", {"max_depth": 5, "criterion": "entropy"}),
]

for name, params in tree_configs:
    results.append(
        run_experiment(name, DecisionTreeClassifier(random_state=42, **params), params)
    )

best_model, best_metrics = max(results, key=lambda r: r[1]["accuracy"])
print(f"\nBest model accuracy: {best_metrics['accuracy']:.3f}")

joblib.dump(best_model, "bcancer_model.joblib")
