
import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegression
from sklearn.gaussian_process.kernels import Kernel, RBF


class ManifoldKernel(Kernel):

    def __init__(self, base_kernel, architecture, transfer_fct="tanh",
                 max_nn_weight=5):
        self.base_kernel = base_kernel

        self.architecture = architecture
        self.transfer_fct = transfer_fct

        n_outputs, theta_nn_size = determine_network_layout(architecture)

        theta0 = \
            list(np.random.uniform(-max_nn_weight, max_nn_weight,
                                   theta_nn_size))
        thetaL = [-max_nn_weight] * theta_nn_size
        thetaU = [max_nn_weight] * theta_nn_size

        param_space = np.vstack((thetaL, theta0, thetaU)).T

        self._parse_param_space(param_space)

    @property
    def params(self):
        return self.theta

    @params.setter
    def params(self, theta):
        self.theta = np.asarray(theta, dtype=np.float)
        if self.theta.ndim == 2:
            self.theta = self.theta[:, 0]

        # XXX:
        if np.any(self.theta == 0):
            self.theta[np.where(self.theta == 0)] \
                += np.random.random((self.theta == 0).sum()) * 2e-5 - 1e-5

    def auto_correlation(self, X, eval_gradient=False):
        X_nn = self._project_manifold(X)
        K = self.base_kernel.auto_correlation(X_nn)
        if not eval_gradient:
            return K
        else:
            # XXX: Analytic expression for gradient based on chain rule and
            #      backpropagation?
            K_gradient = np.empty((K.shape[0], K.shape[1],
                                   self.theta.shape[0]))
            for i in range(self.theta.shape[0]):
                eps = np.zeros_like(self.theta)
                eps[i] += 1e-5
                X_nn_i = self._project_manifold(X, self.theta + eps)
                K_i = self.base_kernel.auto_correlation(X_nn_i)
                K_gradient[..., i] = (K_i - K) / 1e-5

            return K, K_gradient


    def cross_correlation(self, X1, X2):
        X1_nn = self._project_manifold(X1)
        X2_nn = self._project_manifold(X2)
        return self.base_kernel.cross_correlation(X1_nn, X2_nn)

    def _project_manifold(self, X, theta=None):
        # Lazila fetch transfer function (to keep object pickable)
        if self.transfer_fct == "tanh":
            transfer_fct = np.tanh
        elif self.transfer_fct == "sin":
            transfer_fct = np.sin
        elif self.transfer_fct == "relu":
            transfer_fct = lambda x: np.maximum(0, x)
        elif hasattr(self.transfer_fct, "__call__"):
            transfer_fct = self.transfer_fct

        if theta is None:
            theta = self.theta

        y = []
        for subnet in self.architecture:
            y.append(X[:, :subnet[0]])
            for layer in range(len(subnet) - 1):
                W = theta[:subnet[layer]*subnet[layer+1]]
                W = W.reshape((subnet[layer], subnet[layer+1]))
                b = theta[subnet[layer]*subnet[layer+1]:
                                 (subnet[layer]+1)*subnet[layer+1]]
                a = y[-1].dot(W) + b
                y[-1] = transfer_fct(a)

                # chop off weights of this layer
                theta = theta[(subnet[layer]+1)*subnet[layer+1]:]

            X = X[:, subnet[0]:]  # chop off used input dimensions

        return np.hstack(y)


def determine_network_layout(architecture):
    """ Determine number of outputs and params of given architecture."""
    n_outputs = 0
    n_params = 0
    for subnet in architecture:
        for layer in range(len(subnet) - 1):
            n_params += (subnet[layer] + 1) * subnet[layer+1]

        n_outputs += subnet[-1]

    return n_outputs, n_params


np.random.seed(0)

# Specify Gaussian Process
kernel = (1e-10, 1.0, 100) \
    * ManifoldKernel(base_kernel=RBF(0.1), architecture=((1, 2),),
                     transfer_fct="tanh", max_nn_weight=1)
gp = GaussianProcessRegression(kernel=kernel)


X_ = np.linspace(-7.5, 7.5, 100)

# Visualization of prior
X_nn = kernel.k2._project_manifold(X_[:, None])
pylab.figure(0, figsize=(8, 8))
pylab.subplot(2, 1, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i])
pylab.title("Prior mapping to manifold")

pylab.subplot(2, 1, 2)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9)
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.xlim(-7.5, 7.5)
pylab.ylim(-3, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(-5, 5, 30)[:, None]
y = np.sin(X[:, 0]) + (X[:, 0] > 0)
old_params = kernel.params
gp.fit(X, y)

# Visualization of posterior
X_nn = kernel.k2._project_manifold(X_[:, None])

pylab.figure(1, figsize=(8, 8))
pylab.subplot(2, 1, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i])
pylab.title("Posterior mapping to manifold")

pylab.subplot(2, 1, 2)
y_mean, y_cov = gp.predict(X_[:, None], return_cov=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9)
pylab.fill_between(X_, y_mean - np.sqrt(np.diag(y_cov)),
                   y_mean + np.sqrt(np.diag(y_cov)),
                   alpha=0.5, color='k')
y_samples = gp.sample(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.xlim(-7.5, 7.5)
#pylab.ylim(-3, 3)
pylab.title("Posterior samples")

pylab.show()