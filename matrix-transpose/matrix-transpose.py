import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    B = np.array(A)
    size = B.shape[::-1] # shape of transpose matrix
    BT = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            BT[i, j] = B[j, i]

    return BT
    
    
