# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Derek>
<Math 345 Section 2>
<9/9/2025>
"""

import numpy as np
import scipy
from scipy import linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    # initializes for gram schmidt
    m, n = np.shape(A)
    Q = A.copy()
    R = np.zeros((n, n))
    # preforms the gram schmidt algorithm
    for i in range(n):
        R[i, i] = scipy.linalg.norm(Q[:,i])
        Q[:,i] = Q[:,i] / R[i,i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], Q[:, j])
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]

    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    # finds the determinate using gram schmidt
    Q, R = qr_gram_schmidt(A)
    det_R = np.abs(np.prod(np.diag(R)))
    return det_R

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    # follows the standard of computing Q and R and then figures out y and then uses
    # back substituation to solve
    Q, R = qr_gram_schmidt(A)
    y = Q.T @ b
    x = np.linalg.solve(R, y)
    return x

    


    


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    # making the parts of the householder algorithm
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.eye(m)
    # we are going to loop over all the colomns and then 0 out the entries below the diagonal
    for k in range(min(m,n)):
        # makes the reflection
        u = np.copy(R[k:, k])
        norm_u = np.linalg.norm(u)
        sign = 1.0 if u[0] >= 0 else -1.0
        u[0] += sign * norm_u

        #normalize th housolder vector so that H is orthagonal
        norm_u_new = np.linalg.norm(u)
        u = u / norm_u_new
        R[k:, k:] -= 2 * np.outer(u, u @ R[k:, k:])
        Q[k:, :] -= 2 * np.outer(u, u @ Q[k:, :])
    return Q.T, R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    # initialize hessenburg
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.eye(m,m)

    for k in range(n-3):
        # makes the reflection
        u = np.copy(H[k+1:, k])
        sign = 1.0 if u[0] >= 0 else -1.0
        u[0] = u[0] + sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)


        H[k+1:, k:] -= -2 * np.outer(u, u.T @ H[k+1:, k:])
        H[:, k+1:] -= -2 * np.matmul((np.matmul(H[:,k+1:], u)), u.T)
        Q[k+1:,:] -= -2 * np.outer(u.T, Q[k+1:,:])

        return H, Q.T





if __name__ == "__main__":
    A = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)

    Q, R = qr_gram_schmidt(A)

    print("Q =\n", Q)
    print("R =\n", R)
    print("Check: QR =\n", Q @ R)

    A = np.random.random((6, 4))
    Q,R = la.qr(A, mode="economic") # Use mode="economic" for reduced QR.
    print(A.shape, Q.shape, R.shape)
    (6, 4), (6, 4), (4, 4)

    # Verify that R is upper triangular, Q is orthonormal, and QR = A.
    np.allclose(np.triu(R), R)
    True
    np.allclose(Q.T @ Q, np.identity(4))
    True
    np.allclose(Q @ R, A)
    True

    A = np.array([[2, 1], [1, 3]], dtype=float)

    print("QR determinant:", abs_det(A))
    print("np.linalg.det:", np.linalg.det(A))  

    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)

    x = solve(A, b)
    print(f"Answer for Ax = b {x}")
    
    A = np.random.random((5, 3))
    Q,R = la.qr(A)                  # Get the full QR decomposition.
    print(A.shape, Q.shape, R.shape)
    (5, 3), (5, 5), (5, 3)

    np.allclose(Q @ R, A)
    True

    # test householder
    A = np.random.randn(4, 4)
    Q, R = qr_householder(A)

    A = np.random.randn(3, 5)
    Q, R = qr_householder(A)
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)

    print("All tests passed!")
    # Tests for Hessenburg
    # Generate a random matrix and get its upper Hessenberg form via SciPy.
    A = np.random.random((8, 8))
    H, Q = la.hessenberg(A, calc_q=True)

    # Verify that H has all zeros below the first subdiagonal and QHQ\trp = A.
    np.allclose(np.triu(H, -1), H)
    True
    np.allclose(Q @ H @ Q.T, A)
    True



    

    
    