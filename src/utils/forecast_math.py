import numpy as np

def logit(p: float) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

def inv_logit(l: float) -> float:
    return 1 / (1 + np.exp(-l)) 