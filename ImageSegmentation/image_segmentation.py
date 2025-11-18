# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import sparse
from imageio.v3 import imread
from scipy import linalg as la
from scipy.sparse import linalg as spla
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #gets the weights of everything leaving each node represented by a new row
    degrees = np.sum(A, axis=1)
    degree_matrix = np.diag(degrees)
    laplacian_m = degree_matrix - A
    return laplacian_m


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    eigs = np.real(la.eigvals(L))
    eigs.sort()
    #connected components
    num_comp = np.sum(eigs < tol)

    if num_comp < len(eigs):
        #we want the min eigs
        alg_con = eigs[num_comp]
    else:
        alg_con = 0.0
    return num_comp, alg_con

    

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #reading and saving the image with its brightness vals as a flat array
        self.image = imread(filename).astype(np.float64)
        if self.image.max() > 1:
            self.image = self.image / 255.0
        if self.image.ndim == 3:
        #gets brighness if it is color
            self.brightness = self.image.mean(axis=2)
        else:
            self.brightness = self.image
        #shape
        self.shape = self.brightness.shape
        self.N = self.shape[0] * self.shape[1]
        self.flattened = self.brightness.flatten()
        self.flattened = self.brightness.flatten()
        


    # Problem 3
    def show_original(self):
        """Display the original image."""
        # displays the image correctly with gray and color
        plt.imshow(self.image, cmap="gray" if self.image.ndim == 2 else None)
        plt.axis("off")
        plt.savefig("the_og_image.png")
        

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        A = sparse.lil_matrix((self.N, self.N))

        #gonna calculate the adjacency matrix     
        for i in range(self.N):
            neighbors, dist = get_neighbors(i, r, self.shape[0], self.shape[1])
            for j, d in zip(neighbors, dist):
                w = np.exp(-np.abs((self.flattened[i] - self.flattened[j])) / sigma_B2
                        - abs(d) / sigma_X2)
                A[i, j] = w
                A[j, i] = w
        #change to csc to make faster
        A = A.tocsc()
        D = np.array(A.sum(axis=1)).flatten()


        return A, D

    def cut(self, A, D):
        """we are going to segment the image"""
        # lapalacian
        D_mat = sparse.diags(D)
        L = D_mat - A

        #the d^-1/2 matrox
        D_safe = D.copy()
        D_safe[D_safe == 0] = 1.0
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(D_safe))
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt

        #smallest eig
        vals, vecs = spla.eigsh(L_norm, k=2, which="SM")

        #reshape to ge the mask
        fiedler_vector = vecs[:, 1] 
        mask = fiedler_vector > 0   
        mask = mask.reshape(self.shape) 
        
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Display the original image and its segments."""
        # adjacency and degree matrices
        A, D = self.adjacency(r=r, sigma_B2=sigma_B2, sigma_X2=sigma_X2)

        # segmetation mask
        mask = self.cut(A, D)
        #fix grays
        if self.image.ndim == 2: 
            positive = self.image * mask
            nnegative = self.image * (~mask)
        #normal colors
        else: 
            mask_3d = np.stack([mask] * 3, axis=-1)
            positive = self.image * mask_3d
            negative = self.image * (~mask_3d)
        fig, axes = plt.subplots(1, 3)
        #pos segment
        axes[0].imshow(self.image, cmap='gray' if self.image.ndim == 2 else None)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        #neg segment
        axes[1].imshow(positive, cmap='gray' if self.image.ndim == 2 else None)
        axes[1].set_title("Positive Segment")
        axes[1].axis('off')
        #neg segment
        axes[2].imshow(negative, cmap='gray' if self.image.ndim == 2 else None)
        axes[2].set_title("Negative Segment")
        axes[2].axis('off')
        plt.savefig("segment_graph.png")

if __name__ == '__main__':
    ImageSegmenter("dream_gray.png").segment()
    ImageSegmenter("dream.png").segment()

    from scipy.stats import norm
    z = 2.53
    prob = norm.cdf(z)
    print(prob)

    
