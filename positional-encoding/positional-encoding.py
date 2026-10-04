import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))
    half_d_model = (d_model // 2) # To handle odd d_model
    
    pos = np.arange(seq_len).reshape(-1, 1)
    denom = 1.0 / np.pow(base, np.arange(0, d_model, 2) / d_model)
    
    pe[:, 0::2] = np.sin(pos * denom)
    pe[:, 1::2] = np.cos(pos * denom[:half_d_model])

    return pe
    