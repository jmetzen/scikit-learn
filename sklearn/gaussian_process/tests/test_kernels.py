"""Testing for kernels for Gaussian processes."""

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Licence: BSD 3 clause

from collections import Hashable
import inspect

import numpy as np

from scipy.optimize import approx_fprime

from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.gaussian_process.kernels \
    import (RBF, RationalQuadratic, ExpSineSquared, DotProduct,
            ConstantKernel, WhiteKernel, PairwiseKernel, KernelOperator,
            Exponentiation)
from sklearn.base import clone

from sklearn.utils.testing import (assert_equal, assert_almost_equal,
    assert_not_equal, assert_array_equal)


X = np.random.normal(0, 1, (10, 2))

kernels = [RBF(l=2.0), RBF(l_bounds=(0.5, 2.0)),
           ConstantKernel(c=10.0),
           2.0 * RBF(l=0.5), RBF(l=2.0) + WhiteKernel(c=1.0),
           RationalQuadratic(alpha=1.0, l=1.0),
           ExpSineSquared(l=1.0, p=1.0),
           DotProduct(sigma_0=1.0), DotProduct(sigma_0=1.0) ** 2]
for metric in PAIRWISE_KERNEL_FUNCTIONS:
    if metric in ["additive_chi2", "chi2"]:
        continue
    kernels.append(PairwiseKernel(gamma=1.0, metric=metric))


def test_kernel_gradient():
    """ Compare analytic and numeric gradient of kernels. """
    for kernel in kernels:
        K, K_gradient = kernel(X, eval_gradient=True)

        assert_equal(K_gradient.shape[0], X.shape[0])
        assert_equal(K_gradient.shape[1], X.shape[0])
        assert_equal(K_gradient.shape[2], kernel.theta.shape[0])

        K_gradient_approx = np.empty_like(K_gradient)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                def eval_kernel_ij_for_theta(theta):
                    kernel_clone = kernel.clone_with_theta(theta)
                    K = kernel_clone(X, eval_gradient=False)
                    return K[i, j]
                K_gradient_approx[i, j] = \
                    approx_fprime(kernel.theta, eval_kernel_ij_for_theta,
                                  1e-10)

        assert_almost_equal(K_gradient, K_gradient_approx, 4)


def test_kernel_theta():
    """ Check that parameter vector theta of kernel is set correctly. """
    for kernel in kernels:
        if isinstance(kernel, KernelOperator) \
            or isinstance(kernel, Exponentiation):  # skip non-basic kernels
            continue
        theta = kernel.theta
        _, K_gradient = kernel(X, eval_gradient=True)

        # Determine kernel parameters that contribute to theta
        args, varargs, kw, default = \
            inspect.getargspec(kernel.__class__.__init__)
        theta_vars = map(lambda s: s.rstrip("_bounds"),
                         filter(lambda s: s.endswith("_bounds"), args))
        assert_equal(kernel.theta_vars, theta_vars)

        # Check that values returned in theta are consistent with
        # hyperparameter values
        for i, theta_var in enumerate(theta_vars):
            assert_equal(theta[i], getattr(kernel, theta_var))

        # Fixed kernel parameters must be excluded from theta and gradient.
        for i, theta_var in enumerate(theta_vars):
            # create copy with certain hyperparameter fixed
            params = kernel.get_params()
            params[theta_var + "_bounds"] = "fixed"
            kernel_class = kernel.__class__
            new_kernel = kernel_class(**params)
            # Check that theta and K_gradient are identical with the fixed
            # dimension left out
            _, K_gradient_new = new_kernel(X, eval_gradient=True)
            assert_equal(theta.shape[0], new_kernel.theta.shape[0] + 1)
            assert_equal(K_gradient.shape[2], K_gradient_new.shape[2] + 1)
            if i > 0:
                assert_equal(theta[:i], new_kernel.theta[:i])
                assert_array_equal(K_gradient[..., :i],
                                   K_gradient_new[..., :i])
            if i + 1 < len(theta_vars):
                assert_equal(theta[i+1:], new_kernel.theta[i:])
                assert_array_equal(K_gradient[..., i+1:],
                                   K_gradient_new[..., i:])

        # Check that values of theta are modified correctly
        for i, theta_var in enumerate(theta_vars):
            theta[i] = 42
            kernel.theta = theta
            assert_equal(getattr(kernel, theta_var), 42)

            setattr(kernel, theta_var, 43)
            assert_equal(kernel.theta[i], 43)


def test_auto_vs_cross():
    """ Auto-correlation and cross-correlation should be consistent. """
    for kernel in kernels:
        K_auto = kernel(X)
        K_cross = kernel(X, X)
        assert_almost_equal(K_auto, K_cross, 5)


def test_kernel_diag():
    """ Test that diag method of kernel returns consistent results. """
    for kernel in kernels:
        K_call_diag = np.diag(kernel(X))
        K_diag = kernel.diag(X)
        assert_almost_equal(K_call_diag, K_diag, 5)


def test_kernel_operator_commutative():
    """ Adding kernels and multiplying kernels should be commutative. """
    # Check addition
    assert_almost_equal((RBF(2.0) + 1.0)(X),
                        (1.0 + RBF(2.0))(X))

    # Check multiplication
    assert_almost_equal((3.0 * RBF(2.0))(X),
                        (RBF(2.0) * 3.0)(X))


def test_kernel_anisotropic():
    """ Anisotropic kernel should be consistent with isotropic kernels."""
    K = RBF([0.5, 2.0])(X)
    X1 = np.array(X)
    X1[:, 0] *= 4
    K1 = RBF(2.0)(X1)
    assert_almost_equal(K, K1)

    X2 = np.array(X)
    X2[:, 1] /= 4
    K2 = RBF(0.5)(X2)
    assert_almost_equal(K, K2)


def test_kernel_stationary():
    """ Test stationarity of kernels."""
    for kernel in kernels:
        if not kernel.is_stationary():
            continue
        K = kernel(X, X + 1)
        assert_almost_equal(K[0, 0], np.diag(K))


def test_kernel_clone():
    """ Test that sklearn's clone works correctly on kernels. """
    for kernel in kernels:
        kernel_cloned = clone(kernel)

        assert_equal(kernel, kernel_cloned)
        assert_not_equal(id(kernel), id(kernel_cloned))
        for attr in kernel.__dict__.keys():
            attr_value = getattr(kernel, attr)
            attr_value_cloned = getattr(kernel_cloned, attr)
            if np.iterable(attr_value):
                assert_array_equal(attr_value, attr_value_cloned)
            else:
                assert_equal(attr_value, attr_value_cloned)
            if not isinstance(attr_value, Hashable):
                # modifiable attributes must not be identical
                assert_not_equal(id(attr_value), id(attr_value_cloned))
