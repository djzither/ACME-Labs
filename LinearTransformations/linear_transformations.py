# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

import time
import numpy as np
from random import random
from matplotlib import pyplot as plt


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #stretch matrix
    S = np.array([[a, 0], [0, b]])
    return S @ A

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #shear matrix
    S = np.array([[1, a], [b, 1]])
    return A.T @ S 


def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #reflect matrix
    scaler = 1/(a**2 + b**2)
    R = ([[(a**2 - b**2), (2 * a * b)], [(2 * a * b), (b**2 - a**2)]])
    return R @ scaler @ A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    #rotate matrix
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]) @ A


# Problem 2

def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    times = np.linspace(0, T, 1000)
    #get origional places
    p_e0 = [x_e, 0]
    p_m0 = [x_m, 0]
    rel_m0 = [p_m0[0] - p_e0[0], p_m0[1] - p_e0[1]]
    #to store x and y coordinates
    earth_x, earth_y = [], []
    moon_x, moon_y = [], []

    for t in times:
        #clockwise
        p_e = rotate(p_e0, omega_e * t)
        
        #pos moon cntr clockwise
        rel_m = rotate(rel_m0, omega_m * t)

        #have to translate
        p_m = [p_e[0] - rel_m[0], p_e[1] - rel_m[1]]


        earth_x.append(p_e[0])
        earth_y.append(p_e[1])
        moon_x.append(p_m[0])
        moon_y.append(p_m[1])
    
    #plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(earth_x, earth_y, label="Earth Orbit")
    ax.plot(moon_x, moon_y, label="Moon Orbit")
    ax.scatter([0], [0], c="yellow", s=200, label="Sun")
    ax.set_aspect("equal")
    ax.legend()
    
    plt.savefig("solar_system.png", dpi=300)


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p)]
                                    for i in range(m)]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # gonna time the times
    ns = [50, 100, 150, 200, 250]
    vec_times = []
    mat_times = []
    for n in ns:
        A = random_matrix(n)
        x = random_vector(n)
        B = random_matrix(n)

        start = time.time()
        matrix_vector_product(A, x)
        vec_time = (time.time() - start)
        vec_times.append(vec_time)
       
        start_mat = time.time()
        matrix_matrix_product(A, B)
        mat_time = (time.time() - start_mat)
        mat_times.append(mat_time)
    #plot the results
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(ns, vec_times, marker='o')
    axs[0].set_title("Matrix Vector Product Timing")
    axs[0].set_xlabel("matrix size")
    axs[0].set_ylabel("Time (s)")

    axs[1].plot(ns, mat_times, marker='o', color='orange')
    axs[1].set_title("Matrix Matrix Product Timing")
    axs[1].set_xlabel("matrix size")
    axs[1].set_ylabel("Time (s)")

    plt.tight_layout()
    plt.savefig("timings.png", dpi=300)

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    #gonna do times
    ns = [2**2, 2**4, 2**6, 2**8]
    vec_times = []
    mat_times = []
    np_vec = []
    np_mat = []
    for n in ns:
        A = random_matrix(n)
        x = random_vector(n)
        B = random_matrix(n)

        start = time.time()
        matrix_vector_product(A, x)
        vec_time = (time.time() - start)
        vec_times.append(vec_time)
       
        start_mat = time.time()
        matrix_matrix_product(A, B)
        mat_time = (time.time() - start_mat)
        mat_times.append(mat_time)


        start_np = time.time()
        np.dot(A, x)
        np_dot_vec_time = (time.time() - start_np)
        np_vec.append(np_dot_vec_time)

        start_np_mat = time.time()
        np.dot(A, B)
        np_dot_mat_time = (time.time() - start_np_mat)
        np_mat.append(np_dot_mat_time)
    #plot the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  

    axs[0].plot(ns, vec_times, marker='o', label='Python mat-vec')
    axs[0].plot(ns, mat_times, marker='o', color='orange', label='Python mat-mat')
    axs[0].plot(ns, np_vec, marker='o', color='red', label='NumPy mat-vec')
    axs[0].plot(ns, np_mat, marker='o', color='green', label='NumPy mat-mat')
    axs[0].set_title("lin-lin Plot")
    axs[0].set_xlabel("Matrix Size")
    axs[0].set_ylabel("Seconds")
    axs[0].legend()

    
    axs[1].loglog(ns, vec_times, marker='o', label='Python mat-vec')
    axs[1].loglog(ns, mat_times, marker='o', color='orange', label='Python mat-mat')
    axs[1].loglog(ns, np_vec, marker='o', color='red', label='NumPy mat-vec')
    axs[1].loglog(ns, np_mat, marker='o', color='green', label='NumPy mat-mat')
    axs[1].set_title("log-log Plot")
    axs[1].set_xlabel("Matrix Size")
    axs[1].set_ylabel("Seconds")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("linlinvsloglogtimings.png", dpi=300)





# if __name__ == "__main__":
#     solar_system(T = 3*np.pi/2, x_e = 10, x_m = 11, omega_e = 1, omega_m = 13)
#     prob3()
#     prob4()

    
#     solar_system(
#     T=3*np.pi/2,  # final time
#     x_e=10,       # Earth’s initial x position
#     x_m=11,       # Moon’s initial x position
#     omega_e=1,    # Earth angular velocity
#     omega_m=13    # Moon angular velocity
# )
#     prob4()

