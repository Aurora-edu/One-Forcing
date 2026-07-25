"""Small statistical helpers shared by rebuttal result aggregators."""

import math
import statistics

from scipy.stats import t as student_t


def mean_std_ci95(values):
    """Return mean, sample standard deviation, and a two-sided 95% t interval.

    A normal 1.96 multiplier substantially understates uncertainty for the
    three-training-seed setting used by the rebuttal experiments.
    """

    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, None, None
    sample_std = statistics.stdev(values)
    critical = float(student_t.ppf(0.975, df=len(values) - 1))
    ci95 = critical * sample_std / math.sqrt(len(values))
    return mean, sample_std, ci95
