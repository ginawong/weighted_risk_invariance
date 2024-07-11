import math

class AverageMeter:
    @staticmethod
    def alpha_from_tau(tau):
        return math.exp(-1.0 / tau)

    def __init__(self, alpha=None, tau=None, drop_first=False):
        """
        Keeps a running total with optionally limited memory. This is known as exponential smoothing. Some math
        is provided to help you choose alpha.

        Average calculated as
            running_average = alpha*running_average + (1-alpha)*sample

        Assuming each sample is IID with mean mu and standard deviation sigma, then after sufficient time has passed
        the mean of the average meter will be mu and the standard deviation is sigma*sqrt((1-alpha)/(1+alpha)). Based
        on this, if we want the standard deviation of the average to be sigma*(1/N) for some N then we should choose
        alpha=(N**2-1)/(N**2+1).

        The time constant (tau) of an exponential filter is the number of updates before the average meter is expected
        to reach (1 - 1/e) * mu = 0.632 * mu when initialized with running_average=0. This can be thought of as the
        delay in the filter or something like the width of a window. It's relation to alpha is alpha = exp(-1/tau).
        Note that this meter initializes running_average with the first sample value, rather than 0, so in reality the
        expected value of the average meter is always mu (still assuming IID). In a real system the average may be a
        non-stationary statistics (for example training loss) so choosing a alpha with a reasonable time constant is
        still important.

        Some reasonable values for alpha

        alpha = 0.9 results in
            sigma = 0.23 * sigma
            tau = 10

        alpha = 0.98 results in
            sigma_meter = 0.1 * sigma
            tau = 50

        alpha = 0.995 results in
            sigma_meter = 0.05 * sigma
            tau = 200

        Args
            alpha (None or float): Range 0 < alpha < 1. The closer to 1 the more accurate the estimate but
                the more delayed the estimate. If alpha and tau are None then the average meter simply keeps a running
                total and returns the current average. If not None then tau must be None.
            tau (None or float): Range tau > 0. The larger tau is the more delay in the statistics, can be thought of
                as proportional to the number of samples being averaged over. If alpha and tau are None then the
                average meter simply keeps a running total and returns the current average. If not None then alpha must
                be None.
            drop_first (bool): If True then ignore the first call to update. Useful in, for example, measuring data
                loading times since the first call to the loader takes much longer than subsequent calls.
        """
        if tau is not None and alpha is not None:
            raise RuntimeError("Cannot specify both tau and alpha")

        if tau is not None:
            self._alpha = self.alpha_from_tau(tau)
        else:
            self._alpha = alpha

        self._drop_first = drop_first

        self._first = None
        self._value = None
        self._running_value = None
        self._count = None
        self.reset()

    def update(self, value, batch_size=1):
        if batch_size == 0:
            return

        if self._drop_first and self._first:
            self._first = False
            return

        if self._alpha is not None:
            self._value = value
            w = self._alpha ** batch_size
            self._running_value = w * self._running_value + (1.0 - w) * value \
                if self._running_value is not None else value
            self._count += batch_size
        else:
            self._value = value
            if self._running_value is not None:
                self._running_value += self._value * batch_size
            else:
                self._running_value = self._value * batch_size
            self._count += batch_size

    @property
    def average(self):
        if self._alpha is not None:
            return self._running_value if self._running_value is not None else 0.0
        elif self._running_value is None:
            return 0
        else:
            return self._running_value / self._count if self._count > 0 else 0.0

    @property
    def value(self):
        return self._value if self._value is not None else 0

    @property
    def count(self):
        return self._count

    def reset(self):
        self._value = None
        self._running_value = None
        self._count = 0
        self._first = True
