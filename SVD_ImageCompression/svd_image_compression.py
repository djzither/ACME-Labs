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
    #svd
    U, sigma, VT = la.svd(A)
    Sigma = np.diag(sigma)

    #all the transforms that it wants
    VS = VT @ S
    VSE = VT @ E
    SVS = Sigma @ VS
    SVE = Sigma @ VSE
    AVS = U @ SVS
    AVE = U @ SVE

    #create subplots
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    datasets = [(S, E), (VS, VSE), (SVS, SVE), (AVS, AVE)]
    titles = [r"$S$", r"$V^TS$", r"$\Sigma V^TS$", r"$U\Sigma V^TS$"]

    for ax, (S_now, E_now), title in zip(axes, datasets, titles):
        ax.plot(S_now[0], S_now[1], 'b')
        ax.plot([0, E_now[0, 0]], [0, E_now[1, 0]])
        ax.plot([0, E_now[0, 2]], [0, E_now[1, 2]])
        ax.set_title(title)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('svd_visualization.png')
    
 




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
    # Best rank-s approximation svd formula
    A_s = U_s @ np.diag(Sigma_s) @ vh_s
    

    # Number of entries to store truncated SVD
    entries = s * (A.shape[0] + A.shape[1] + 1)

    return A_s, entries

    





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

    if err <= sigma[-1]: #the value of thei index that is smaller than the error
        raise ValueError("error is too small")
    
    # find smallest s such that sigma s+1 < err

    s = np.argmax(sigma < err)
    #compute the rank-s approx
    A_s, entries = svd_approx(A, s)

 
    return A_s, entries




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
    #we are going to compress the image
    #grey
    image = imread(filename) / 255
    if image.ndim == 2:
        A_s, entries = lowest_rank_approx(image, s)
        compressed = np.clip(A_s, 0, 1)
    #color problems baby
    elif image.ndim == 3 and image.shape[2] == 3:
        channels = []
        total_entries = 0
        #each channel separately
        for i in range(3):
            A_s, entries = svd_approx(image[:, :, i], s)
            channels.append(A_s)
            total_entries += entries
    #put em bakc to gether
        compressed = np.dstack(channels)
        entries = total_entries

    else:
        raise ValueError("Image grayscale or RGb")
    #how many pixel values minues origional
    original_entries = np.prod(image.shape)
    saved = original_entries - entries

    #we will plot each one
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(compressed, cmap='gray' if image.ndim == 2 else None)
    axes[1].set_title(f"Rank- {s} Approx")
    axes[1].axis('off')

    plt.suptitle(f"Compression: saved {saved:,} entries out of {original_entries:,}")
    plt.savefig('done_compressed.png')





# if __name__ == "__main__":
#     visualize_svd([[3, 1], [1, 3]])
#     A = np.array([[3, 1],
#                     [1, 3]])
#     err = 2.5

#     A_s, entries = lowest_rank_approx(A, err)
#     print("A_s:\n", A_s)
#     print("Entries:", entries)

    
#     compress_image('hubble.jpg', 30)
