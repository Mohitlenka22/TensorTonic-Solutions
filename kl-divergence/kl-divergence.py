import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.asarray(p)
    q = np.asarray(q)
    q = np.clip(q, eps, None)
    # Get indices where p == 0
    indices = np.where(p == 0)
    p[indices]=1; q[indices]=1
    kl = (p * np.log(p / q)).sum()
    return float(kl)