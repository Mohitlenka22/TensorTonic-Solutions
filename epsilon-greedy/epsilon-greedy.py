import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    # Write code here
    # set seed
    rng=np.random.default_rng(seed=seed)
    t = rng.random() # [0, 1)
    if t < epsilon:
        e = rng.choice(q_values)
        return q_values.index(e)
    else:
        return int(np.argmax(q_values))