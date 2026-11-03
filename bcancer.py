import joblib
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    train_size=0.9,
)

# A handful of decision tree configs to compare. max_depth controls how deep
# the tree can grow (None = unlimited, which tends to overfit); criterion
# controls the splitting rule used at each node.
param_grid = [
    {"max_depth": None, "criterion": "gini"},
    {"max_depth": 3, "criterion": "gini"},
    {"max_depth": 5, "criterion": "gini"},
    {"max_depth": 5, "criterion": "entropy"},
]

mlflow.set_experiment("bcancer-decision-tree")

best_model = None
best_accuracy = -1.0

for params in param_grid:
    with mlflow.start_run():
        mlflow.log_params(params)

        model = DecisionTreeClassifier(random_state=42, **params)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # Trust the tree's internal node-storage type since we trained this
        # model ourselves in this process (skops flags it by default as it
        # can be unsafe to load from an untrusted/third-party source).
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"params={params} -> accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

print("Best accuracy:", best_accuracy)
joblib.dump(best_model, "bcancer_model.joblib")
