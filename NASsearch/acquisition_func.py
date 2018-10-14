""" module for acquisition function"""
import numpy as np
from scipy.stats import norm

from NASsearch.gp import GaussianProcess


class AcquisitionFunc:
    """ class for acquisition function
    expected improvement in this case
    """
    def __init__(self, X_train, y_train, current_optimal, mode, trade_off):
        """
        :param mode: pi: probability of improvement, ei: expected improvement, lcb: lower confident bound
        :param trade_off: a parameter to control the trade off between exploiting and exploring
        :param model_type: gp: gaussian process, rf: random forest
        """
        self.X_train = X_train
        self.y_train = y_train
        self.current_optimal = current_optimal
        self.mode = mode or "ei"
        self.trade_off = trade_off or 0.01
        self.model = GaussianProcess(80)

    def compute(self, X_test):
        self.model.fit(self.X_train, self.y_train)
        y_mean, y_variance, y_std = self.model.predict([X_test])
        print(y_mean, " ", y_variance)
        # y_variance = y_std ** 2
        z = (y_mean - self.current_optimal - self.trade_off) / y_std

        if self.mode == "ei":
            if y_std < 0.0000000001:
                return 0
            result = y_std * (z * norm.cdf(z) + norm.pdf(z))
        elif self.mode == "pi":
            result = norm.cdf(z)
        else:
            result = - (y_mean - self.trade_off * y_std)
        return np.squeeze(result)
