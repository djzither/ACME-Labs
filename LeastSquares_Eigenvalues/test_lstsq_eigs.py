"""Unit testing file for Least Squares and Computing Eigenvalues problem 6"""


import leastsquares_eigenvalues as lstsq_eigs
import pytest
import numpy as np
from scipy import linalg as la

def test_qr_algorithm():
    """
    Write at least one unit test for problem 6, the qr algorithm function.
    """
    A = np.diag([1, 2, 3])
    eig_vals = lstsq_eigs.qr_algorithm(A)
    expected = np.array([1, 2, 3])
    assert np.allclose(sorted(eig_vals), sorted(expected), atol=1e-10)


def test_qr_algorithm_complex():
    """
    Test qr_algorithm on a random non-symmetric matrix
    to ensure it correctly handles complex eigenvalues.
    """
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n)  # possible complex eigs

    eig_vals = lstsq_eigs.qr_algorithm(A)  
    expected = np.linalg.eigvals(A)       

    # real and imag
    expected_sorted = sorted(expected, key=lambda x: (x.real, x.imag))
    eig_vals_sorted = sorted(eig_vals, key=lambda x: (x.real, x.imag))

    assert np.allclose(eig_vals_sorted, expected_sorted, atol=1e-4), \
        f"Expected {expected_sorted}, got {eig_vals_sorted}"


def test_power_method():
    #Sets up test cases
    A = np.array([[1, 1], [1, 1]])
    B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    C = np.array([[2, 2], [1, 3]])
    
    Aval, Avec = lstsq_eigs.power_method(A)
    Bval, Bvec = lstsq_eigs.power_method(B)
    Cval, Cvec = lstsq_eigs.power_method(C)
    
    #Checks if it finds the appropriate eigenvalue
    assert abs(Aval - 2) < 1e-5, "Incorrect eigenvalue"
    assert abs(Bval - 3) < 1e-5, "Incorrect eigenvalue"
    assert abs(Cval - 4) < 1e-5, "Incorrect eigenvalue"
    
    #Checks if it finds an eigenvector that works
    assert np.linalg.norm(A @ Avec - Aval * Avec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(B @ Bvec - Bval * Bvec) < 1e-3, "Incorrect vector"
    assert np.linalg.norm(C @ Cvec - Cval * Cvec) < 1e-3, "Incorrect vector"