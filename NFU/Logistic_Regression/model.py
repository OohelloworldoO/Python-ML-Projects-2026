import numpy as np


class LogisticRegression:
    def __init__(
        self,
        lr: float = 0.01,
        num_iter: int = 100000,
        fit_intercept: bool = True,
        verbose: bool = False,
    ) -> None:
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _loss(self, h: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-9
        h = np.clip(h, eps, 1 - eps)
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)

            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose and i % 10000 == 0:
                loss = self._loss(h, y)
                print(f"iter={i}, loss={loss:.6f}")

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")

        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.round(self.predict_prob(X)).astype(int)