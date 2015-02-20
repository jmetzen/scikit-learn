"""Gaussian process regression (GPR) prior and posterior

This example illustrates the prior and posterior of a GPR. Mean, standard
deviation, and 10 samples are shown for both prior and posterior.
"""
print __doc__

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegression
from sklearn.gaussian_process.kernels import RBF


# Specify Gaussian Process
kernel = (1e-10, 1.0, 100) * RBF(param_space=(1e-10, 1.0, None))
gp = GaussianProcessRegression(kernel=kernel)

# Plot prior
plt.figure(0, figsize=(8, 8))
plt.subplot(2, 1, 1)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
y_samples = gp.sample(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, color='b', lw=1)
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Prior, kernel:  %s" % kernel)

# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(0, 5, 10)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)
gp.fit(X, y)

# Plot posterior
plt.subplot(2, 1, 2)
X_ = np.linspace(0, 5, 100)
y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')

y_samples = gp.sample(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, color='b', lw=1)
plt.scatter(X[:, 0], y, c='r', s=50, zorder=10)
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.title("Posterior, kernel: %s" % kernel)
plt.tight_layout()
plt.show()
