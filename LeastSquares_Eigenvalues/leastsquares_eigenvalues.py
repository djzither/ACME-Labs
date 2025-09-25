# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

import numpy as np
from cmath import sqrt
from scipy import linalg as la
from matplotlib import pyplot as plt



# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    #we are doing qr decomp with scipy
    Q, R = la.qr(A, mode="economic")

    #going to figure out the answer
    Qtb = Q.T @ b
    x = la.solve_triangular(R, Qtb)
    return x

# Problem 2
def line_fit(dataset="housing.npy"):
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    data = np.load(dataset)
    years = data[:, 0] 
    prices = data[:, 1]

    #making the A matrix to solve
    A = np.column_stack((years, np.ones_like(years)))
    b = prices

    #we are going to plot it
    beta = least_squares(A, b)
    slope, intercept = beta
    plt.scatter(years, prices, label="Housing price data")
    x_line = np.linspace(years.min(), years.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, "r-", label = "Least squares line")
    plt.xlabel("year (0 = 2000)")
    plt.ylabel("housing price index")
    plt.legend()
    plt.savefig("housingprices.png")

# Problem 3
def polynomial_fit(dataset="housing.npy"):
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    data = np.load(dataset)
    years = data[:, 0]
    housing_price = data[:, 1]
    refined_dom = np.linspace(years.min(), years.max(), 400)

    plt.cla()

    degrees = [3, 6, 9, 12]
    for i, deg in enumerate(degrees):
        #this is the A vandermonde matrix
        A = np.vander(years, deg + 1, increasing=True)
        #solve least sqr
        coeffs = la.lstsq(A, housing_price)[0]
        #find the dence points to eval at
        A_dense = np.vander(refined_dom, deg + 1, increasing = True)
        y_fit = A_dense @ coeffs
        
        plt.subplot(2, 2, i+1)
        plt.scatter(years, housing_price, label = "Data")
        plt.plot(refined_dom, y_fit, label = f"degree {deg}")
        plt.title(f"least squares fit with degree: {deg}")
        plt.xlabel("Year")
        plt.xlabel("year")
        plt.legend()
    plt.tight_layout()
    plt.savefig("least squares better.png")

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit(dataset="ellipse.npy"):
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    plt.cla()
    data = np.load(dataset)
    x = data[:, 0]
    y = data[:, 1]
    #make matrix
    A = np.column_stack((x**2, x, x*y, y, y**2))
    b = np.ones_like(x)

    params = la.lstsq(A, b)[0]

    plt.scatter(x, y, label = "Data Points")
    plot_ellipse(*params)
    plt.legend()
    plt.title("ellipse fit")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.savefig("ellipse_fit.png")

# Problem 5
def power_method(A, N=20, tol=1e-12):
    
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #get matrix size
    m, n = np.shape(A)
    #chooses the vector needed for power method
    x = np.random.rand(n)
    x = x/la.norm(x)
    
    for k in range(N):
        #current vector by A
        x = A @ x
        #normalize
        x = x/la.norm(x)
    #gives dom eig val and vect
    return x.T @ A @ x, x

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    m, n = np.shape(A)
    S = la.hessenberg(A)

    for _ in range(N):
        Q, R = la.qr(S)
        S = R @ Q
    eigs = []
    i = 0
    while i < n:
        if i == n-1 or abs(S[i+1, i]) < tol:
            #get these eigan vals
            eigs.append(S[i, i])
            i += 1
        # 2x2 eig vals == hard!!
        else:
            a, b, c, d = S[i, i+1], S[i+1, i], S[i+1, i+1]
            trace = a + d
            det = a * d - b * c
            disc = trace**2 - 4 * det
            #eig vals
            root1 = (trace + sqrt(disc)) / 2
            root2 = (trace - sqrt(disc)) / 2
            eigs.extend([root1, root2])
            i += 2 #2 rows this time
    return np.array(eigs)

