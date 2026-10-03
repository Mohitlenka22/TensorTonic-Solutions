import numpy as np

def sig(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))
    
    
def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    if not isinstance(x, list):
        return float(sig(x))
    else:
        y = np.array(x, dtype=float)
        shape_ = y.shape
        y = y.flatten()
        y = sig(y)
        return y.reshape(shape_)
        