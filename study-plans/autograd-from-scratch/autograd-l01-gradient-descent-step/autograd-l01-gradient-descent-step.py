import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    new_values = []
    for x, y in zip(values, gradients):
        new_values.append(float(x - learning_rate*y))

    l_pred = 0
    for i in range(len(gradients)):
        l_pred += gradients[i] * (new_values[i] - values[i])

    return (new_values, float(l_pred))
        
