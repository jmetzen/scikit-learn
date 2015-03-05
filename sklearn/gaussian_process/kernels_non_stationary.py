import numpy as np

from sklearn.gaussian_process.kernels import Kernel


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

    def __call__(self, X, Y=None, eval_gradient=False):
        X_nn = self._project_manifold(X)
        if Y is None:
            K = self.base_kernel(X_nn)
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
                    K_i = self.base_kernel(X_nn_i)
                    K_gradient[..., i] = (K_i - K) / 1e-5

                return K, K_gradient
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            Y_nn = self._project_manifold(Y)
            return self.base_kernel(X_nn, Y_nn)

    def _project_manifold(self, X, theta=None):
        # Lazily fetch transfer function (to keep object pickable)
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
