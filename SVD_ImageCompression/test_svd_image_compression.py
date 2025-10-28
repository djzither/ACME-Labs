"""Unit testing file svd_image_compression.py"""


import svd_image_compression as sv
import numpy as np
import numpy.linalg as nla

def test_compact_svd(): # problem 1
        """Unit test for the algorithm to compute the compact SVD of a matrix"""
        m = 7 # change m and n as you see fit
        n = 6
        A = np.random.randint(1, 10, (m, n)).astype(float)
        U, sigma, V =  sv.compact_svd(A) 

        assert np.allclose(U@np.diag(sigma)@V, A) is True, "Incorrect truncated SVD"
        assert np.allclose(U.T @ U, np.identity(n)) is True, "U is not orthonormal"
        assert np.allclose(V.T @ V, np.identity(n)) is True, "V is not orthonormal"
        assert nla.matrix_rank(A) == len(sigma), "Number of nonzero singular values is not equal to rank of A"
        
def test_svd_approx(): # problem 3
    """Unit test for approximating the rank S SVD approximation of a matrix A"""
    A = np.array([[3, 1], [1, 3]])
    s = 1
    A_s, entries = sv.svd_approx(A, s)

    assert A_s.shape == A.shape
    expected_entries = s * (A.shape[0] + A.shape[1] + 1)
    assert entries == expected_entries, f"expected {expected_entries} entries"

    assert np.linalg.matrix_rank(A_s) <= s
    fro_error = np.linalg.norm(A - A_s, 'fro')
    assert fro_error >= 0, "Frobenius norm error should be non-negative"

    print("All tests passed!")

    