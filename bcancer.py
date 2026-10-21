from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    #random_state=42,
    #test_size=0.2,
    train_size=0.9,
)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

print("accuracy:", model.score(X_test, y_test))
