import math

import numpy as np


def hellinger(p, q):
    return math.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2) / 2)


def psi(p, q):
    return np.sum((p - q) * np.log(p / q))


def kl_divergence(p, q):
    kl = np.sum(p * np.log(p / q))
    return kl


def js_divergence(p, q):
    m = (p + q) / 2
    pm = kl_divergence(p, m)
    qm = kl_divergence(q, m)
    jsd = (pm + qm) / 2
    return jsd


def ks(p, q):
    return np.max(np.abs(np.cumsum(p) - np.cumsum(q)))
