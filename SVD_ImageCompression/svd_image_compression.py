"""Volume 1: The SVD and Image Compression."""

import numpy as np
from imageio.v3 import imread
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #svd algorithm
    eigs, V = la.eig(A.conj().T @ A)
    sigma = np.sqrt(np.real(eigs))
    #greatest to least
    idxs = np.argsort(sigma)[::-1]
    sigma = sigma[idxs]
    V = V[:, idxs]

    r = np.sum(sigma > tol)
    sigma = sigma[:r]
    V = V[:, :r]
    U = A @ V / sigma
    return U, sigma, V.conj().T


    

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    S = np.array([np.cos(theta), np.sin(theta)])
    E = np.array([[1, 0, 0], [0, 0, 1]])
    AS = A @ S
    AE = A @ E
    
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()

    axes[0].axhline(0, color='gray', linewidth=1)
    axes[0].axvline(0, color='gray', linewidth=1)
    axes[0].plot(S[0], S[1], 'b')

    
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].axvline(0, color='gray', linewidth=1)
    axes[1].plot(AS[0], AS[1], 'b')

    
    axes[2].axhline(0, color='gray', linewidth=1)
    axes[2].axvline(0, color='gray', linewidth=1)
    axes[2].plot(E[0], E[1], 'b')

    
    axes[3].axhline(0, color='gray', linewidth=1)
    axes[3].axvline(0, color='gray', linewidth=1)
    axes[3].plot(AE[0], AE[1], 'b')
    
    
    # Formatting
    plt.axis("equal")
    plt.legend()
    plt.title("Effect of A on E and S")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("SVD_on_SE.png")
    
 




# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    u, sigma, v_conj = compact_svd(A)
    #this strips off the columns and entries that don't matter
    U_s = u[:, :s]
    Sigma_s = sigma[:s]
    vh_s = v_conj[:s, :]
    





# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    u, sigma, v = compact_svd(A)
    


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    raise NotImplementedError("Problem 5 Incomplete")

if __name__ == "__main__":
    visualize_svd([[3, 1], [1, 3]])