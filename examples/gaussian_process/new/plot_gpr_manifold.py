
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels_non_stationary import ManifoldKernel
from sklearn.metrics import mean_squared_error
from sklearn.learning_curve import learning_curve


np.random.seed(0)

n_samples = 100
n_features = 5
n_dim_manifold = 2

# Generate data
def f(X_nn):
    return X_nn[:, 0]**2 + X_nn[:, 1]**2

X = np.random.uniform(-5, 5, (n_samples, n_features))
A = np.random.random((n_features, n_dim_manifold))
X_nn = X.dot(A)
y = f(X_nn)

# Standard Gaussian Process
kernel = (1e-10, 1.0, 100) * RBF([(0.1, 1, 100.0) for i in range(n_features)])
gp = GaussianProcessRegressor(kernel=kernel)

# Manifold Gaussian Process
kernel_nn = (1e-10, 10.0, 100) \
    * ManifoldKernel(base_kernel=RBF(5.0),
                     architecture=((n_features, n_dim_manifold),),
                     transfer_fct="linear", max_nn_weight=1)
gp_nn = GaussianProcessRegressor(kernel=kernel_nn, y_err=1e-2)

# Plot learning curve
def plot_learning_curve(estimators, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    colors = ["r", "g", "b"]
    for color, estimator in zip(colors, estimators.keys()):
        train_sizes, train_scores, test_scores = \
            learning_curve(estimators[estimator], X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring="mean_squared_error")
        test_scores_median = np.median(test_scores, axis=1)
        test_scores_min = np.min(test_scores, axis=1)
        test_scores_max = np.max(test_scores, axis=1)

        plt.fill_between(train_sizes, test_scores_min,
                         test_scores_max, alpha=0.1, color=color)
        plt.plot(train_sizes, test_scores_median, 'o-', color=color,
                 label=estimator)

    plt.grid()
    plt.title(title)
    plt.yscale("symlog")
    plt.xlabel("Training examples")
    plt.ylabel("-MSE")
    plt.legend(loc="best")

plot_learning_curve({"GP": gp, "GP NN": gp_nn}, "Test", X, y, cv=10)
plt.show()
