"""Unit testing file for Image Segmentation"""

import image_segmentation
import pytest
import numpy as np
from scipy import sparse as sp

@pytest.fixture
def set_up_matrices():
    # Sets up test cases
    A = np.array([[0, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 1],
                  [1, 1, 0, 1, 0, 0],
                  [1, 0, 0, 1, 0, 0]], dtype=float)

    B = np.array([[0, 3, 0, 0, 0, 0],
                  [3, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 2, .5],
                  [0, 0, 0, 2, 0, 1],
                  [0, 0, 0, .5, 1, 0]], dtype=float)

    C = np.array([[0, 4, 2, 100],
                  [4, 0, 23, 54],
                  [2, 23, 0, 23],
                  [100, 54, 23, 0]], dtype=float)

    return A, B, C


def test_laplacian(set_up_matrices):
    A, B, C = set_up_matrices

    # Checks for the correct laplacian
    assert (image_segmentation.laplacian(A) == sp.csgraph.laplacian(A)).all(), "Incorrect Laplacian"
    assert (image_segmentation.laplacian(B) == sp.csgraph.laplacian(B)).all(), "Incorrect Laplacian"
    assert (image_segmentation.laplacian(C) == sp.csgraph.laplacian(C)).all(), "Incorrect Laplacian"


def test_connectivity(set_up_matrices):
    """
    Write at least one unit test for the connectivity function.
    """
    A, B, C = set_up_matrices
    #connected components
    expected_num_comp, _ = sp.csgraph.connected_components(A, directed=False, return_labels=True)
    num_comp, alg_con = image_segmentation.connectivity(A)
    assert  num_comp == expected_num_comp, "incorrect num comp"
    assert round(alg_con, 2) == 1.59, "algebraic not correct"

    expected_num_comp, _ = sp.csgraph.connected_components(B, directed=True, return_labels=True)
    num_comp, alg_con = image_segmentation.connectivity(B)
    assert num_comp == expected_num_comp, "incorrect num comp"


def test_adjacency_heart():
    #tests adjancecy matrix
    A_ref = sp.load_npz("HeartMatrixA.npz")
    D_ref = np.load("HeartMatrixD.npy")
    seg = image_segmentation.ImageSegmenter("blue_heart.png")
    A, D = seg.adjacency()

    assert np.allclose((A - A_ref).data, 0, atol=1e-12)
    assert np.allclose(D.diagonal(), D_ref, atol=1e-12)


    


