"""
Train and compare models on the breast cancer dataset, tracking every run
with MLflow so the params/metrics can be browsed and compared later with
`uv run mlflow ui`.
"""

import matplotlib

# mlflow.sklearn.autolog() renders a confusion matrix / ROC curve / PR curve
# PNG per run via matplotlib. On Windows, matplotlib's default backend is the
# interactive TkAgg (GUI) backend; creating dozens of Tk figures in one
# script with no GUI ever shown eventually crashes with
# "RuntimeError: main thread is not in main loop". "Agg" renders straight to
# image files with no GUI involved at all, which is all autolog needs. Must
# be set before matplotlib.pyplot is imported anywhere (including inside
# mlflow), so this has to stay the first import in the file.
matplotlib.use("Agg")

import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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


# --- Logistic regression: its own experiment --------------------------------
# Kept separate for the same reason as every model family below: C, penalty,
# and solver aren't comparable to a tree's max_depth or a forest's
# n_estimators, so mixing them into one experiment's run table would just be
# a lot of empty/irrelevant columns either way.
mlflow.set_experiment("bcancer-logreg")

with mlflow.start_run(run_name="logreg-baseline"):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    test_accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"[logreg-baseline] test accuracy: {test_accuracy:.4f}")
    consider("logreg-baseline", model, test_accuracy)

# --- Decision trees: a more realistic hyperparameter search -----------------
# Own experiment too ("bcancer-decision-tree") - same reasoning as above.
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

mlflow.set_experiment("bcancer-decision-tree")

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


# --- Random forest: a separate experiment ----------------------------------
# A random forest trains many decision trees on bootstrapped (randomly
# resampled with replacement) subsets of the data and random feature subsets
# per split, then averages their votes. That averaging is what tames a
# single tree's tendency to overfit, generally without needing much pruning.
# Logged under its own MLflow experiment ("bcancer-random-forest") so the
# existing "bcancer" experiment (logreg + decision trees above) is left
# completely untouched.
#
# New hyperparameters beyond what DecisionTreeClassifier has:
# - n_estimators: how many trees to grow and average over. More trees =
#   lower variance (smoother, more stable predictions) at the cost of more
#   compute; returns diminish after a few hundred for a dataset this size.
# - bootstrap: whether each tree trains on a bootstrap resample of the
#   training data (True, the usual "bagging" approach) instead of the full
#   set every time - this is what makes the trees diverse enough to be worth
#   averaging.
# - max_samples: when bootstrap=True, the fraction (or count) of samples
#   drawn per tree. <1.0 makes each tree see less data -> more diverse trees,
#   similar in spirit to max_features but sampling rows instead of columns.
# - oob_score: "out-of-bag" score. Each bootstrap resample leaves ~37% of
#   training rows unused for that tree; oob_score=True evaluates each row
#   using only the trees that didn't see it during training - a free
#   validation estimate without needing a separate held-out split.
#
# It also reuses max_depth, min_samples_leaf, min_samples_split, criterion,
# max_features, and class_weight from the decision tree, applied to every
# tree in the forest.
mlflow.set_experiment("bcancer-random-forest")

forest_configs = [
    {
        "run_name": "rf-baseline",
        "params": {"n_estimators": 100},
        "note": "Default-ish random forest: 100 trees, no depth cap - the "
        "usual starting point before tuning anything.",
    },
    {
        "run_name": "rf-many-shallow-trees",
        "params": {"n_estimators": 300, "max_depth": 4},
        "note": "More trees (300), each individually shallow/weak - leans "
        "on averaging over quantity rather than letting any one tree "
        "overfit.",
    },
    {
        "run_name": "rf-regularized",
        "params": {
            "n_estimators": 200,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        "note": "Combines per-tree regularizers (min_samples_leaf, "
        "max_features='sqrt') with the forest's own averaging effect.",
    },
    {
        "run_name": "rf-balanced",
        "params": {"n_estimators": 200, "class_weight": "balanced"},
        "note": "class_weight='balanced' applied per-tree, to avoid "
        "under-prioritizing the malignant class the way we did for the "
        "single-tree run.",
    },
    {
        "run_name": "rf-subsampled-oob",
        "params": {
            "n_estimators": 300,
            "max_samples": 0.7,
            "oob_score": True,
        },
        "note": "Each tree only sees 70% of the training rows (max_samples) "
        "and oob_score=True reports an accuracy estimate from the "
        "left-out 30% per tree, as a free sanity check against the "
        "held-out test accuracy below.",
    },
]

for config in forest_configs:
    print(f"\n[{config['run_name']}] params={config['params']}")
    print(f"  -> {config['note']}")

    with mlflow.start_run(run_name=config["run_name"]):
        forest = RandomForestClassifier(random_state=RANDOM_STATE, **config["params"])
        forest.fit(X_train, y_train)

        test_accuracy = forest.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        if config["params"].get("oob_score"):
            mlflow.log_metric("oob_score", forest.oob_score_)
            print(f"  -> oob_score: {forest.oob_score_:.4f}")
        mlflow.set_tag("note", config["note"])
        print(f"  -> test accuracy: {test_accuracy:.4f}")
        consider(config["run_name"], forest, test_accuracy)


# --- Gradient boosting: yet another separate experiment --------------------
# Instead of averaging many independent trees (bagging, what the random
# forest above does), gradient boosting builds trees *sequentially*: each
# new tree is trained to correct the errors the ensemble-so-far is still
# making. That usually squeezes out more accuracy than a forest, but is more
# prone to overfitting if you let it run too long or too aggressively.
# Logged under its own experiment ("bcancer-gradient-boosting") - the
# "bcancer" and "bcancer-random-forest" experiments above are untouched.
#
# New hyperparameters beyond what the tree/forest already introduced:
# - learning_rate: shrinks each new tree's contribution before adding it to
#   the ensemble. Lower values (e.g. 0.01-0.05) need more trees but
#   generalize better; higher values (e.g. 0.3+) learn fast but overfit fast.
# - n_estimators: how many trees to add *in sequence*. Unlike a forest, more
#   isn't free here - each one keeps chasing the previous errors, so too
#   many (especially with a high learning_rate) will eventually overfit the
#   training set.
# - subsample: fraction of training rows used to fit *each* tree (sampled
#   without replacement, unlike bagging's bootstrap). <1.0 ("stochastic
#   gradient boosting") trades a bit of per-tree accuracy for less
#   correlation between trees and better generalization.
# - max_depth: boosted trees are usually kept shallow (2-4) on purpose -
#   each one only needs to capture a small residual pattern, not the whole
#   problem, so deep boosted trees overfit very quickly.
mlflow.set_experiment("bcancer-gradient-boosting")

boosting_configs = [
    {
        "run_name": "gb-baseline",
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "note": "scikit-learn's defaults: 100 shallow trees, moderate "
        "learning rate - a reasonable starting point before tuning.",
    },
    {
        "run_name": "gb-slow-and-steady",
        "params": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 3},
        "note": "Low learning_rate + many more trees - takes smaller, more "
        "cautious steps toward fitting the residuals, usually generalizes "
        "at least as well as the baseline.",
    },
    {
        "run_name": "gb-stochastic",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "max_depth": 3,
        },
        "note": "subsample=0.7 means each tree only sees 70% of the "
        "training rows - adds randomness between trees (stochastic "
        "gradient boosting), often reducing overfitting.",
    },
    {
        "run_name": "gb-aggressive-overfit-risk",
        "params": {"n_estimators": 300, "learning_rate": 0.3, "max_depth": 5},
        "note": "High learning_rate + deeper trees + many estimators - "
        "learns the training set fast and is the config here most likely "
        "to overfit; a useful contrast against the slower configs above.",
    },
]

for config in boosting_configs:
    print(f"\n[{config['run_name']}] params={config['params']}")
    print(f"  -> {config['note']}")

    with mlflow.start_run(run_name=config["run_name"]):
        booster = GradientBoostingClassifier(random_state=RANDOM_STATE, **config["params"])
        booster.fit(X_train, y_train)

        train_accuracy = booster.score(X_train, y_train)
        test_accuracy = booster.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.set_tag("note", config["note"])
        print(f"  -> train accuracy: {train_accuracy:.4f}, test accuracy: {test_accuracy:.4f}")
        consider(config["run_name"], booster, test_accuracy)

print(f"\nBest model: {best_run_name} (test accuracy: {best_accuracy:.4f})")
joblib.dump(best_model, "bcancer_model.joblib")
print("Saved best model to bcancer_model.joblib")

print("Run `uv run mlflow ui` and open http://127.0.0.1:5000 to compare all runs.")
