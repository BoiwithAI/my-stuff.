from sklearn.neural_network import MLPClassifier
model = MLPClassifier
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
model.fit(X, y)
