# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np
from scipy import sparse

from sklearn.utils.testing import assert_array_almost_equal, assert_equal
from sklearn.utils.testing import assert_almost_equal
from nose.tools import assert_true

from sklearn.datasets import make_classification, make_blobs
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import sigmoid_calibration, _SigmoidCalibration
from sklearn.calibration import calibration_curve


def test_calibration():
    """Test calibration objects with isotonic and sigmoid"""
    n_samples = 500
    X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                               random_state=42)
    sample_weight = np.random.random(y.size)

    X -= X.min()  # MultinomialNB only allows positive X

    # split train and test
    X_train, y_train, sw_train = \
        X[:n_samples], y[:n_samples], sample_weight[:n_samples]
    X_test, y_test, sw_test = \
        X[n_samples:], y[n_samples:], sample_weight[n_samples:]

    # Naive-Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train, sw_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]

    # Naive Bayes with calibration
    for this_X_train, this_X_test in [(X_train, X_test),
                                      (sparse.csr_matrix(X_train),
                                       sparse.csr_matrix(X_test))]:
        for method in ['isotonic', 'sigmoid']:
            pc_clf = CalibratedClassifierCV(clf, method=method, cv=2)
            pc_clf.fit(this_X_train, y_train, sample_weight=sw_train)
            prob_pos_pc_clf = pc_clf.predict_proba(this_X_test)[:, 1]

            assert_true(brier_score_loss(y_test, prob_pos_clf) >
                        brier_score_loss(y_test, prob_pos_pc_clf))

    # test multi-class setting with classifier that implements
    # only decision function
    clf = LinearSVC()
    X, y = make_blobs(n_samples=100, n_features=2, random_state=42,
                      cluster_std=3.0)
    X_train, y_train = X[::2], y[::2]
    X_test, y_test = X[1::2], y[1::2]
    ir_clf = CalibratedClassifierCV(clf, method='isotonic', cv=2)
    ir_clf.fit(X_train, y_train)
    probas = ir_clf.predict_proba(X_test)
    assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X_test)))
    assert_true(clf.fit(X_train, y_train).score(X_test, y_test) <=
                ir_clf.fit(X_train, y_train).score(X_test, y_test))


def test_sigmoid_calibration():
    """Test calibration values with Platt sigmoid model"""
    exF = np.array([5, -4, 1.0])
    exY = np.array([1, -1, -1])
    # computed from my python port of the C++ code in LibSVM
    AB_lin_libsvm = np.array([-0.20261354391187855, 0.65236314980010512])
    assert_array_almost_equal(AB_lin_libsvm, sigmoid_calibration(exF, exY), 3)
    lin_prob = 1. / (1. + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
    sk_prob = _SigmoidCalibration().fit(exF, exY).predict(exF)
    assert_array_almost_equal(lin_prob, sk_prob, 6)


def test_calibration_curve():
    """Check calibration_curve function"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0., 0.1, 0.2, 0.8, 0.9, 1.])
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=2)
    prob_true_unnormalized, prob_pred_unnormalized = \
        calibration_curve(y_true, y_pred * 2, n_bins=2, normalize=True)
    assert_equal(len(prob_true), len(prob_pred))
    assert_equal(len(prob_true), 2)
    assert_almost_equal(prob_true, [0, 1])
    assert_almost_equal(prob_pred, [0.1, 0.9])
    assert_almost_equal(prob_true, prob_true_unnormalized)
    assert_almost_equal(prob_pred, prob_pred_unnormalized)
