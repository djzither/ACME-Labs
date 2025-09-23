
# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Derek>
Math 345>
<9/16/2025>
"""

import numpy as np

def prob1():
    """ Define the matrices A and B as arrays. Return the matrix product AB. """
    # we make matrices and multiply them
    A = np.array([[3,-1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    return A @ B

def prob2():
    """ Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A. """
    #we will return the funciton that defines the following np.array
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    return -1 * (A @ A @ A) + 9 * (A @ A) - 15 * A

def prob3():
    """ Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    # uses python special commands to make the arrays
    A = np.triu(np.ones(7))
    B_5 = np.full((7, 7), 5)
    B = np.tril(B_5, k=0)
    B = np.where(np.tril(np.ones((7, 7), dtype=bool)), -1, B_5)
    # do some operations
    Final = A @ B @ A
    return Final.astype(np.int64)

def prob4(A):
    """ Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    # use mask to make all negatives 0
    A_copy = np.copy(A)
    mask = A_copy < 0
    A_copy[mask] = 0
    return A_copy

def prob5():
    """ Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    #make all arrays
    A = np.array([[0, 2, 4],[1, 3, 5]])
    B = np.tril(np.full((3, 3), 3))
    C = np.diag([-2] * 3)
    I = np.eye(3)
    # make the block
    M = np.block([[np.zeros((3, 3)), A.T, I], [A, np.zeros((2,2)), np.zeros((2,3))], [B, np.zeros((3, 2)), C]])
    return M
    
def prob6(A):
    """ Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    # sum the rows
    sum_row = A.sum(axis=1, keepdims = True)
    #normalize by the sum
    normalized = A / sum_row
    return normalized

def prob7():
    """ Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """
    #get grid
    grid = np.load("grid.npy")
    #gets the products
    horiz = grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:]
    vert = grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]
    diag_down_right = grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]
    diag_down_left = grid[3:, :-3] * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:]
    
    #gets max of products
    max_product = np.max([
    np.max(horiz),
    np.max(vert),
    np.max(diag_down_right),
    np.max(diag_down_left)
    ])
    return max_product

if __name__ == "__main__":
    print(prob2())
    print(prob3())
    print(prob4(np.array([-3,-1,3])))
    print(prob5())
    A = np.array([[1,1,0],[0,1,0],[1,1,1]])
    print(prob6(A))
    print(prob7())