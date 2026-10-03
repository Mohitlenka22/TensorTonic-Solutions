import math
import numpy as np
from scipy.special import erf

def exact_erf_gelu(x: float) -> float:
    return (x / 2) * (1 + erf(x / math.sqrt(2)))

def gelu(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    return exact_erf_gelu(np.array(x, dtype=float))