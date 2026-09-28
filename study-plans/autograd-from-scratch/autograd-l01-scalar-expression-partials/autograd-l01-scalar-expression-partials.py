def d(a: float, b: float, c: float) -> float:
    return float(a*b + c)

def d_(dh: float, dx: float, h: float) -> float:
    return float((dh - dx) / h)
    
def scalar_expression_partials(a: float, b: float, c: float, h: float) -> tuple[float, float, float, float]:
    """
    Returns the expression value and numerical partials for a, b, and c.
    """
    value = d(a, b, c)
    a_ = d_(d(a+h, b, c), value, h)
    b_ = d_(d(a, b+h, c), value, h)
    c_ = d_(d(a, b, c+h), value, h)

    return (value, a_, b_, c_)
    
