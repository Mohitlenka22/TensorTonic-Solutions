import numpy as np

def f_x(coefficients: list, x: float):
    n = len(coefficients)
    x_k = np.array([x**i for i in range(n)], dtype=np.float64)
    c_k = np.array(coefficients, dtype=np.float64)
    return (x_k * c_k).sum(axis=0)

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    fx = f_x(coefficients, x)
    fh = f_x(coefficients, x+h)
    slope = (fh - fx) / h
    # print()
    return (float(fx), float(fh), float(slope))
    