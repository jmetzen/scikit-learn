
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels_non_stationary import ManifoldKernel
from sklearn.metrics import mean_squared_error
from sklearn.learning_curve import learning_curve


np.random.seed(0)

n_samples = 100
n_features = 5
n_dim_manifold = 2

# Generate data
def f(X_nn):
    #return X_nn[:, 0] * X_nn[:, 1]
    return X_nn[:, 0]**2 + X_nn[:, 1]**2

X = np.random.uniform(-5, 5, (n_samples, n_features))
A = np.random.random((n_features, n_dim_manifold))
X_nn = X.dot(A)
y = f(X_nn)

# Standard Gaussian Process
kernel = (1e-10, 1.0, 100) * RBF([(0.1, 1, 100.0) for i in range(n_features)])
gp = GaussianProcessRegressor(kernel=kernel)

# Manifold Gaussian Process
kernel_nn = (1e-10, 1.0, 100) \
    * ManifoldKernel(base_kernel=RBF((1.0, 5.0, 100.0)),
                     architecture=((n_features, n_dim_manifold),),
                     transfer_fct="linear", max_nn_weight=0.1) \
    + WhiteKernel((1e-10, 1e-3, 1e-1))
gp_nn = GaussianProcessRegressor(kernel=kernel_nn, y_err=0)

# Fit GPs and create scatter plot on test data
gp.fit(X, y)
gp_nn.fit(X, y)

X_test = np.random.uniform(-5, 5, (1000, n_features))
X_nn_test = X_test.dot(A)
y_test = f(X_nn_test)
plt.figure(0)
plt.scatter(y_test, gp.predict(X_test), c='b', label="GP RBF")
plt.scatter(y_test, gp_nn.predict(X_test), c='r', label="GP NN")
plt.xlabel("True")
plt.xlabel("Predicted")
plt.legend(loc=0)
plt.title("Scatter plot on test data")

plt.figure(1)
X_gp_nn_test = gp_nn.kernel_.k1.k2._project_manifold(X_test)
plt.scatter(X_gp_nn_test[:, 0], X_gp_nn_test[:, 1], c=y_test)
plt.colorbar()
plt.title("Learned 2D Manifold")


# Plot learning curve
def plot_learning_curve(estimators, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
    plt.title("Learning curve")

plt.figure(2)
plot_learning_curve({"GP": gp, "GP NN": gp_nn}, "Test", X, y, cv=10)
plt.show()
