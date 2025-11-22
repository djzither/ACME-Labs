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
    B = A.conj().T @ A                  # Step 1
    eigvals, V = la.eigh(B)             # Step 2 (ascending)

    eigvals = np.maximum(eigvals, 0)     # Numerical fix
    s = np.sqrt(eigvals)

    # sorting decreasingg
    idx = np.argsort(s)[::-1]
    s = s[idx]
    V = V[:, idx]

    # don't want zero sing vals
    r = np.sum(s > tol)
    s = s[:r]
    V = V[:, :r]

    # u array brodcasting

    U = (A @ V) / s[np.newaxis, :]

    return U, s, V.conj().T



    

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
    
    U, sigma, Vh = compact_svd(A)

    if s < 1 or s > len(sigma):
        raise ValueError("s must be between 1 and the rank of A.")
    U_s = U[:, :s]
    Sigma_s = np.diag(sigma[:s])
    Vh_s = Vh[:s, :]

   
    A_s = U_s @ Sigma_s @ Vh_s

    
    m, n = A.shape
    entries = s * (m + n + 1)

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

    if err <= sigma[-1]:
        raise ValueError("error is too small")
    
    # find smallest s such that sigma s+1 < err

    
    idx_list = np.where(sigma < err)[0]
    s = idx_list[0]   
    

    return svd_approx(A, s)




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
        A_s, entries = svd_approx(image, s)
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

#     A = np.random.random((10, 5))
#     U2, s2, Vh2 = compact_svd(A, tol=1e-12)


#     print(np.allclose(U2.T @ U2, np.eye(len(s2))))
#     print(np.allclose(Vh2 @ Vh2.T, np.eye(len(s2))))


#     print(np.allclose(U2 @ np.diag(s2) @ Vh2, A))


#     U, s, Vh = la.svd(A, full_matrices=False)
#     print(np.allclose(sorted(s2, reverse=True), sorted(s, reverse=True)))


#     print(np.linalg.matrix_rank(A) == len(s2))
