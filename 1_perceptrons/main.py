import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    w: np.array
    b: float

    def __init__(
        self,
        dimensions: int,
        w: np.array = None,
        b: float = None,
    ):
        self.w = w if w is not None else np.random.normal(size=dimensions)
        self.b = b or 0

    def predict(self, x: np.array) -> np.array:
        return np.sign(x @ self.w + self.b)

    def calculate_least_squared_loss(self, x: np.array, y: np.array) -> float:
        y_pred = self.predict(x)
        return sum((y - y_pred) ** 2)

    def calculate_smooth_loss(self, x: np.array, y: np.array) -> float:
        y_pred = self.predict(x)
        return sum((y - y_pred) @ x)


def _debug_plot(X, y):
    def _plot(x, label, ax):
        ax.plot(
            x[:, 0],
            x[:, 1],
            label=label,
            ls="None",
            marker="*",
        )

    fig, _ax = plt.subplots()
    _plot(X[y > 0, :], "1", _ax)
    _plot(X[y < 0, :], "0", _ax)

    fig.savefig(__file__ + ".plot_1.png")


def _make_training_data(
    n_records: int,
    dimensions: int,
    loc: np.array = None,
    hpn: np.array = None,
):
    """Make linearly separable training data."""

    # mean of training data
    loc = loc if loc is not None else np.random.uniform(size=dimensions)

    # hyper plane normal that divides the points
    hpn = hpn if hpn is not None else np.random.uniform(size=dimensions)

    # generate centred observations
    x_norm = np.random.normal(size=(n_records, dimensions))

    y = np.sign(x_norm @ hpn)
    x = x_norm + loc

    return x, y, hpn, loc


def main():
    n_dims = 9
    n_samples = 100
    x, y, hpn, loc = _make_training_data(
        n_samples,
        n_dims,
    )

    # make a perceptron with random weights and 0 bias
    random_perceptron = Perceptron(n_dims)

    # Using the hyper plane to get the perfect weights/bias for the perceptron
    perfect_perceptron = Perceptron(n_dims, w=hpn, b=-np.dot(hpn, loc))

    print(random_perceptron.calculate_least_squared_loss(x, y))
    print(perfect_perceptron.calculate_least_squared_loss(x, y))

    print(random_perceptron.calculate_smooth_loss(x, y))
    print(perfect_perceptron.calculate_smooth_loss(x, y))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
