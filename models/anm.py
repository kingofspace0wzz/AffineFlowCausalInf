# ------------------------------------------
# Additive noise models (Hoyer at al 2009)
# ------------------------------------------
#
#
#
# this code is taken from cdt toolbox as it was failing

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = p1[:, np.newaxis]
        p2 = p2[:, np.newaxis]

    size1 = p1.shape
    size2 = p2.shape

    G = np.sum(p1 * p1, axis=1)[:, np.newaxis]
    H = np.sum(p2 * p2, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    # Set kernel size to median distance between points, if no kernel specified
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def FastHsicTestGamma(X, Y, sig=np.array([-1, -1]), maxpnt=500):
    """This function implements the HSIC independence test using a Gamma approximation
     to the test threshold. Use at most maxpnt points to save time.

    :param X: contains dx columns, m rows. Each row is an i.i.d sample
    :param Y: contains dy columns, m rows. Each row is an i.i.d sample
    :param sig: [0] (resp [1]) is kernel size for x(resp y) (set to median distance if -1)
    :return: test statistic

    """

    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat


def normalized_hsic(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    h = FastHsicTestGamma(x, y)

    return h


class ANM:
    """ANM algorithm.

    **Description**: The Additive noise model is one of the most popular
    approaches for pairwise causality. It bases on the fitness of the data to
    the additive noise model on one direction and the rejection of the model
    on the other direction.

    **Data Type**: Continuous

    **Assumptions**: Assuming that :math:`x\\rightarrow y` then we suppose that
    the data follows an additive noise model, i.e. :math:`y=f(x)+E`.
    E being a noise variable and f a deterministic function.
    The causal inference bases itself on the independence
    between x and e.
    It is proven that in such case if the data is generated using an additive noise model, the model would only be able
    to fit in the true causal direction.

    .. note::
       Ref : Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters, Jonas and Schölkopf, Bernhard,
       "Nonlinear causal discovery with additive noise models", NIPS 2009
       https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models.pdf
    """

    def __init__(self):
        """Init the model."""
        pass

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
            data (np.ndarray): 2-column np.ndarray of variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = data[:, 0], data[:, 1]
        a = scale(a).reshape((-1, 1))
        b = scale(b).reshape((-1, 1))

        p = self.anm_score(b, a) - self.anm_score(a, b)
        causal_dir = 'x->y' if p > 0 else 'y->x'
        return p, causal_dir

    def anm_score(self, x, y):
        """Compute the fitness score of the ANM model in the x->y direction.

        Args:
            x (numpy.ndarray): Variable seen as cause
            y (numpy.ndarray): Variable seen as effect

        Returns:
            float: ANM fit score
        """
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        indepscore = normalized_hsic(y_predict - y, x)

        return indepscore
