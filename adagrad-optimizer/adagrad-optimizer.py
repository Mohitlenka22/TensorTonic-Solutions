import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    new_G = np.asarray(g)**2 + np.asarray(G)
    new_W = np.asarray(w) - lr * (np.asarray(g) / (new_G + eps)**(1/2))

    return {
        "new_w": new_W,
        "new_G": new_G
    }