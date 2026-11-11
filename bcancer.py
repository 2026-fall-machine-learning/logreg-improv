import itertools

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

# --- Random forest: a grid search over a bagged ensemble of trees ---
# A random forest trains many decision trees, each on a bootstrapped
# (random-with-replacement) sample of the rows and a random subset of
# features per split, then averages/votes their predictions. That extra
# randomness is what usually lets forests beat a single tree - individual
# trees overfit in different ways, and averaging cancels much of that out.
# It has all the same tree hyperparameters (max_depth, min_samples_leaf,
# criterion, ...) plus a few more of its own:
forest_param_grid = {
    # n_estimators: how many trees to grow in the forest. More trees
    # generally keeps helping (less variance from averaging) but with
    # diminishing returns and higher training/prediction cost.
    "n_estimators": [100, 300],
    # max_depth: same as for a single tree, but here it caps every tree in
    # the forest. None lets each tree grow fully; forests resist
    # overfitting better than lone deep trees because of the averaging.
    "max_depth": [5, 10, None],
    # max_features: how many features are considered at each split.
    # "sqrt" (= sqrt(30) ~ 5 features here) is the forest default and is
    # what decorrelates the trees from each other; "log2" considers even
    # fewer. Smaller values -> more diverse trees, possibly less accurate
    # individually, but potentially a better ensemble.
    "max_features": ["sqrt", "log2"],
}

forest_param_combos = [
    dict(zip(forest_param_grid.keys(), values))
    for values in itertools.product(*forest_param_grid.values())
]


def forest_run_name(params: dict) -> str:
    depth = params["max_depth"] if params["max_depth"] is not None else "none"
    return f"forest_n={params['n_estimators']}_depth={depth}_feat={params['max_features']}"


for params in forest_param_combos:
    results.append(
        run_experiment(
            forest_run_name(params),
            RandomForestClassifier(random_state=42, n_jobs=-1, **params),
            params,
        )
    )

# --- A couple of extra, less commonly-grid-searched forest hyperparameters ---
results.append(
    run_experiment(
        "forest_bootstrap-false",
        # bootstrap=False: trains every tree on the *entire* training set
        # (no resampling) instead of a bootstrap sample. Each tree still
        # differs via random feature subsets at each split, but the
        # ensemble loses some of its diversity - usually a slight
        # accuracy/variance trade-off versus the default bootstrap=True.
        RandomForestClassifier(
            random_state=42, n_jobs=-1, n_estimators=300, bootstrap=False
        ),
        {"n_estimators": 300, "bootstrap": False},
    )
)
results.append(
    run_experiment(
        "forest_class_weight-balanced",
        # class_weight="balanced": same idea as for the single tree -
        # reweights classes inversely to frequency, to counter the mild
        # benign/malignant imbalance in this dataset.
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
        ),
        {"n_estimators": 300, "max_depth": 10, "class_weight": "balanced"},
    )
)

# --- Gradient boosting: trees built sequentially, not in parallel ---
# Unlike a forest (many independent trees averaged/voted together),
# boosting builds trees one at a time, where each new (shallow) tree is
# trained to correct the errors (residuals) of the ensemble so far,
# scaled down by `learning_rate` before being added in. It's a different
# bias/variance strategy: many weak learners added incrementally, versus
# many strong-ish learners averaged.
boost_param_grid = {
    # n_estimators: how many boosting stages (trees) to add in sequence.
    # Unlike a forest, too many stages *can* overfit - boosting keeps
    # chasing residuals even after test performance has plateaued.
    "n_estimators": [100, 300],
    # learning_rate: how much each new tree's correction counts (shrinks
    # its contribution). Lower rates need more estimators to converge but
    # generalize better; it trades off directly against n_estimators.
    "learning_rate": [0.01, 0.1, 0.3],
    # max_depth: boosting trees are usually kept shallow (stumps to a
    # handful of levels) since they're each only meant to be a "weak
    # learner" nudging the ensemble - unlike the deep trees that can work
    # fine in a forest.
    "max_depth": [2, 3],
}

boost_param_combos = [
    dict(zip(boost_param_grid.keys(), values))
    for values in itertools.product(*boost_param_grid.values())
]


def boost_run_name(params: dict) -> str:
    return (
        f"boost_n={params['n_estimators']}"
        f"_lr={params['learning_rate']}"
        f"_depth={params['max_depth']}"
    )


for params in boost_param_combos:
    results.append(
        run_experiment(
            boost_run_name(params),
            GradientBoostingClassifier(random_state=42, **params),
            params,
        )
    )

# --- A couple of extra, less commonly-grid-searched boosting hyperparameters ---
results.append(
    run_experiment(
        "boost_subsample-0.8",
        # subsample < 1.0 turns this into *stochastic* gradient boosting:
        # each tree trains on a random fraction of the rows (like a
        # forest's bootstrap, but without replacement). Adds randomness
        # that can reduce overfitting and speed up training.
        GradientBoostingClassifier(
            random_state=42, n_estimators=200, learning_rate=0.1, subsample=0.8
        ),
        {"n_estimators": 200, "learning_rate": 0.1, "subsample": 0.8},
    )
)
results.append(
    run_experiment(
        "boost_min_samples_leaf-10",
        # min_samples_leaf: same meaning as for a single tree - applied to
        # each of the (shallow) trees boosting builds. Raising it keeps
        # the individual weak learners from latching onto tiny, noisy
        # groups of residuals.
        GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            min_samples_leaf=10,
        ),
        {"n_estimators": 200, "learning_rate": 0.1, "min_samples_leaf": 10},
    )
)

best_model, best_metrics = max(results, key=lambda r: r[1]["accuracy"])
print(f"\nBest model accuracy: {best_metrics['accuracy']:.3f}")

joblib.dump(best_model, "bcancer_model.joblib")
